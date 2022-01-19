import gc
import random
import copy
import os
import traceback
import numpy
import pandas as pd
from tqdm.notebook import tqdm as tqdmn
import math
import statistics
import xgboost as xgb
import scipy

import matplotlib.pyplot as plt

MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=BIGGER_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def treat_errors(raw_dict, max_time, min_action_lenght, debug=True):
    """Function aimed at removing errors from the series and splitting
    after the max time, cutting sessions either on max time only
    or on both max time and errors.

    Arguments:
        raw_dict {dict} -- Untreated dict of series data, user ids as keys
        max_time {int} -- Max time between keystrokes for series splitting

    Returns:
        dict -- Treated dict of series data, user ids as keys
    """
    treated_dict = {}
    if debug:
        userbar = tqdmn(total=len(raw_dict), desc="Treating Errors")
    for user_key in raw_dict:
        user_sessions = []
        for session_array in raw_dict[user_key]:
            action = []
            for event_line_index, event_line in enumerate(session_array):
                while (event_line_index > 0 and
                        session_array[event_line_index-1][0] >= event_line[0]
                    ):
                    event_line = [event_line[0]+0.0001,*event_line[1:]]
                    session_array[event_line_index] = event_line
                if (type(event_line[0]) != float or
                    type(event_line[1]) != str or
                    type(event_line[2]) != str or
                    type(event_line[3]) != int or
                    type(event_line[4]) != int or
                    event_line[0] < 0 or
                    event_line[1] == 'Scroll' or
                    event_line[3] >= 4000 or
                    event_line[4] >= 4000 or
                    (
                        event_line_index > 0 and 
                        [*session_array[event_line_index-1][1:]] == [*event_line[1:]]
                    )
                    ):
                    continue
                if (action and
                    event_line[0]-action[-1][0] > max_time and
                    action[-1][2] != 'Drag'                          
                    ):
                    if len(action) >= min_action_lenght:
                        user_sessions.append((action,1)) #1=MouseMove
                    action = []
                elif (action and event_line[2] == 'Released'):
                    action.append(event_line)
                    if len(action) >= min_action_lenght:
                        if session_array[event_line_index - 1][2] == 'Pressed':
                            user_sessions.append((action,2)) #2=PointClick
                        elif session_array[event_line_index - 1][2] == 'Drag':
                            user_sessions.append((action,3)) #3=DragDrop
                    action = []
                    continue
                action.append(event_line)
            if len(action) >= min_action_lenght:
                user_sessions.append((action,1))
        treated_dict[user_key] = user_sessions
        if debug:
            userbar.update(1)
    if debug:
        userbar.close()
    return treated_dict

def pull_features(samples_dict):
    """Pulling function for multiple features, using callable feature functions.

    Arguments:
        samples_dict {tuple} -- The samples dict, users for keys
        feature_functions {list} -- List of tuples, each tuple is a feature to be pulled;
                                    fist value of the tuple should be the callable function
                                    for the feature, second value should be another tuple
                                    with the other function parameters. eg: 
                                    [(callable_feature_function,(argument,)),]

    Returns:
        dict -- Users dict of pulled features; every user has a list of features, each feature is a list
                of its model ready samples, multiple features always have their lists ordered the same.
    """
    pulled_features_dict = {}
    userbar = tqdmn(total=len(samples_dict), desc="User progress")
    samplebar = tqdmn(desc="Featuring Samples")
    for user_key in samples_dict:
        samplebar.reset(total=len(samples_dict[user_key]))
        pulled_features_dict[user_key] = []
        for sample in samples_dict[user_key]: 
            sample_copy = copy.deepcopy(sample)
            pulled_features_dict[user_key].append(featuring_function(sample_copy))
            samplebar.update(1)
        userbar.update(1)
    userbar.close()
    samplebar.close()
    return pulled_features_dict

def featuring_function(sample):
    """Feature function for the original Intrusion Detection Using Mouse Dynamics article

    Arguments:
        sample {list} -- A list of samples to pull features from

    Returns:
        list -- The features pulled data
    """
    movement_type = sample[1]
    data_lines = numpy.array(sample[0])
    t = data_lines[:,0]
    x = data_lines[:,3]
    y = data_lines[:,4]
    trajectory, sumOfAngles = 0, 0
    angles, s, vx, vy, v = [0], [0], [0], [0], [0]
    numPoints = len(x)
    omega = [0]
    a = [0]
    j = [0]
    c = []
    for i in range(1, numPoints):
        dx = int(x[i]) - int(x[i - 1])
        dy = int(y[i]) - int(y[i - 1])
        dt = float(t[i]) - float(t[i-1])
        vx_val = dx/dt
        vy_val = dy/dt
        vx.append(vx_val)
        vy.append(vy_val)
        v.append(math.sqrt(math.pow(vx_val, 2) + math.pow(vy_val, 2)))
        angle = math.atan2(dy, dx)
        angles.append(angle)
        sumOfAngles = sumOfAngles + angle
        distance = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        trajectory = trajectory + distance
        s.append(trajectory)
    dif_s = []
    for i in s:
        if i not in dif_s:
            dif_s.append(i)
    if len(dif_s) <= 1:
        raise("todos os S são iguais")
    mean_vx = statistics.mean(vx)
    sd_vx = numpy.std(vx)
    max_vx = max(vx)
    min_vx = numpy.ma.masked_equal(vx, 0.0, copy=False).min()
    min_vx = 0 if type(min_vx) == numpy.ma.core.MaskedConstant else min_vx
    mean_vy = statistics.mean(vy)
    sd_vy = numpy.std(vy)
    max_vy = max(vy)
    min_vy = numpy.ma.masked_equal(vy, 0.0, copy=False).min()
    min_vy = 0 if type(min_vy) == numpy.ma.core.MaskedConstant else min_vy
    mean_v = statistics.mean(v)
    sd_v = numpy.std(v)
    max_v = max(v)
    min_v = numpy.ma.masked_equal(v, 0.0, copy=False).min()
    min_v = 0 if type(min_v) == numpy.ma.core.MaskedConstant else min_v
    n_angles = len(angles)
    for i in range(1, n_angles):
        dtheta = angles[i] - angles[i-1]
        dt = float(t[i]) - float(t[i - 1])
        omega.append(dtheta/dt)
    mean_omega = statistics.mean(omega)
    sd_omega = numpy.std(omega)
    max_omega = max(omega)
    min_omega = numpy.ma.masked_equal(omega, 0.0, copy=False).min()
    min_omega = 0 if type(min_omega) == numpy.ma.core.MaskedConstant else min_omega
    accTimeAtBeginning = 0
    cont = True
    for i in range(1, numPoints - 1):
        dv = v[i] - v[i-1]
        dt = float(t[i]) - float(t[i - 1])
        if cont and dv > 0:
            accTimeAtBeginning += dt
        else:
            cont = False
        a.append(dv/dt)
    mean_a = statistics.mean(a)
    sd_a = numpy.std(a)
    max_a = max(a)
    min_a = numpy.ma.masked_equal(a, 0.0, copy=False).min()
    min_a = 0 if type(min_a) == numpy.ma.core.MaskedConstant else min_a
    n_a = len(a)
    for i in range(1, n_a):
        da = a[i] - a[i - 1]
        dt = float(t[i]) - float(t[i - 1])
        j.append(da / dt)
    mean_jerk = statistics.mean(j)
    sd_jerk = numpy.std(j)
    max_jerk = max(j)
    min_jerk = numpy.ma.masked_equal(j, 0.0, copy=False).min()
    min_jerk = 0 if type(min_jerk) == numpy.ma.core.MaskedConstant else min_jerk
    sharp_angles = 0
    nn = len(s)
    for i in range(1, nn):
        dp = s[i] - s[i-1]
        if dp == 0:
            continue
        dangle = angles[i] - angles[i - 1]
        curv = dangle/dp
        c.append(curv)
        if abs(curv) < 0.0005:
            sharp_angles = sharp_angles + 1
    mean_curv = statistics.mean(c)
    sd_curv = numpy.std(c)
    max_curv = max(c)
    min_curv = numpy.ma.masked_equal(c, 0.0, copy=False).min()
    min_curv = 0 if type(min_curv) == numpy.ma.core.MaskedConstant else min_curv
    elapsed_time = float(t[numPoints - 1]) - float(t[0])
    theta = math.atan2(int(y[numPoints - 1]) - int(y[0]), int(x[numPoints - 1]) - int(x[0]))
    direction = 0
    if theta >= 0 and theta < (math.pi/4):
        direction = 0
    elif theta >= (math.pi/4) and theta < (math.pi/2):
        direction = 1
    elif theta >= (math.pi/2) and theta < ((3*math.pi)/4):
        direction = 2
    elif theta >= ((3*math.pi)/4) and theta < math.pi:
        direction = 3
    elif theta >= (-math.pi)/4 and theta < 0:
        direction = 7
    elif theta >= ((-math.pi)/2) and theta < ((-math.pi)/4):
        direction = 6
    elif theta >= ((-3*math.pi)/4) and theta < ((-math.pi)/2):
        direction = 5
    elif theta >= (-math.pi) and theta < ((-3*math.pi)/4):
        direction = 4
    distEndToEndLine = math.sqrt(math.pow((int(x[0]) - int(x[numPoints-1])), 2) + math.pow((int(y[0]) - int(y[numPoints-1])), 2))
    if trajectory == 0:
        straightness = 0
    else:
        straightness = distEndToEndLine / trajectory
    if straightness > 1:
        straightness = 1
    a = float(x[numPoints-1]) - float(x[0])
    b = float(y[0]) - float(y[numPoints-1])
    c = float(x[0]) * float(y[numPoints-1]) - float(x[numPoints-1]) * float(y[0])
    max_deviation = 0
    den = math.sqrt((a*a)+(b*b))
    for i in range(1, numPoints-1):
        d = math.fabs(a*float(x[i])+b*float(y[i])+c)
        if d > max_deviation:
            max_deviation = d
    if den > 0:
        max_deviation /= den
    pulled_sample = [movement_type, elapsed_time, trajectory,
                        distEndToEndLine, direction, straightness, numPoints,
                        sumOfAngles, max_deviation, sharp_angles,
                        accTimeAtBeginning, mean_vx, sd_vx, max_vx, min_vx,
                        mean_vy, sd_vy, max_vy, min_vy, mean_v, sd_v, max_v,
                        min_v, mean_a, sd_a, max_a, min_a, mean_jerk, sd_jerk,
                        max_jerk, min_jerk, mean_omega, sd_omega, max_omega,
                        min_omega, mean_curv, sd_curv, max_curv, min_curv]
    return pulled_sample

def create_single_user_data(users_dict,user_key):   
    X_true = users_dict[user_key]
    y = [1 for _ in X_true]
    other_users = list(users_dict.keys())
    other_users.remove(user_key)
    X_false = []
    for _ in X_true:
        X_false.append(random.choice(users_dict[random.choice(other_users)]))
        y.append(0)
    X = numpy.concatenate((X_true,X_false),axis=0)
    return X,y

def train_standard_xgb(X_train,y_train,X_val,y_val): #Parameters searched
    classifier = xgb.XGBClassifier(
                                tree_method='exact',
                                # gpu_id=0,
                                seed=seed,
                                eval_metric='mlogloss',
                                use_label_encoder=False,
                                max_depth=4,
                                min_child_weight=3,
                                subsample=0.875,
                                learning_rate=0.05,
                                gamma=3,
                                colsample_bytree=0.88,
                                )
    classifier = classifier.fit(X_train, y_train)
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_val, y_val)
    return train_score,test_score,classifier

def arrayify_data_helper(data,shuffle=False,shuffle_seed=False,feature_index=False,reshape=False):
    """Helper function aimed at transforming standardized list data into array

    Arguments:
        data {list} -- Data to transform to array

    Keyword Arguments:
        shuffle {bool} -- Whether to shuffle the data before returning (default: {False})
        shuffle_seed {int} -- Numpy shuffle seed to use, only valid if shuffle == True (default: {False})
        feature_index {int} -- Used if feature is indexed besides other features (default: {False})
        reshape {tuple} -- If passed will try to reshape the final array into the given shape (default: {False})

    Returns:
        numpy.array -- The arrayified data; or,
        numpy.array, int -- The arrayified data, the seed used for the shuffle.
    """
    if type(feature_index) == bool and not feature_index:
        array_data = numpy.array(data)
    elif type(feature_index) == int:
        array_data = numpy.array(data)[:,feature_index]
    else:
        raise IndexError("Array feature indexing should be int. Either data isn't standardized or the indexing expected isn't implemented.")
    if shuffle:
        if type(shuffle_seed) == bool and not shuffle_seed:
            shuffle_seed = numpy.random.randint(100000)
        elif type(shuffle_seed) != int:
            raise ValueError("Shuffle seed must be int. Expected to be reused from a previous function call with only shuffle=True.")
        numpy.random.seed(shuffle_seed)
        numpy.random.shuffle(array_data)
    if reshape:
        array_data = array_data.reshape(reshape)
    if shuffle:
        return array_data, shuffle_seed
    return array_data

def combine_for_average(users_dict,max_combinations_per_user=False,random_pick=False):
    """Function that combines each 'other' sample of an user with every 'true' sample
    of that user, while at the same time combining it with an equal number of random
    'true' samples from other users. Intended for an average evaluation aplication.

    Arguments:
        users_dict {dict} -- Samples dict, user ids as keys, to form combinations.

    Keyword Arguments:
        max_combinations_per_user {int} -- Max combinations per user, users with less
                                           samples than specified defaults to all
                                           samples being used (default: {False})
        random_pick {bool} -- Whether to randomly pick from available samples or pull
                              max combinations sequentially, only used max combinations
                              is specified (default: {False})

    Returns:
        list, list, list, list -- X part 1 and 2, followed by y and then user references
    """
    truth_dict = {}
    normal_dict = {}
    for user_key in users_dict:
        truth_samples = []
        normal_samples = []
        for sample in users_dict[user_key]:
            if sample[1] == True:
                truth_samples.append(sample)
            else:
                normal_samples.append(sample)
        if truth_samples:
            truth_dict[user_key] = truth_samples
        if normal_samples:
            normal_dict[user_key] = normal_samples
    user_reference = []
    x_part1 = []
    x_part2 = []
    y = []
    tagged_user_keys = list(truth_dict.keys())
    for user_key in truth_dict:
        if user_key not in list(normal_dict.keys()):
            continue
        if type(max_combinations_per_user) == int and len(normal_dict[user_key]) > max_combinations_per_user:
            if random_pick:
                other_list = random.sample(normal_dict[user_key], max_combinations_per_user)
            else:
                other_list = normal_dict[user_key][:max_combinations_per_user]
        else:
            other_list = normal_dict[user_key]
        positive_list_1 = []
        positive_list_2 = []
        positive_reference = []
        negative_list_1 = []
        negative_list_2 = []
        negative_reference = []
        while len(other_list) > 0:
            current_sample = other_list.pop(0)
            for truth_sample in truth_dict[user_key]:
                positive_list_1.append(current_sample[0])
                positive_list_2.append(truth_sample[0])
                positive_reference.append((user_key,user_key))
                user2_key = user_key
                while user2_key == user_key:
                    user2_key = tagged_user_keys[random.randint(0,len(tagged_user_keys)-1)]
                negative_list_1.append(current_sample[0])
                negative_list_2.append(truth_dict[user2_key][random.randint(0,len(truth_dict[user2_key])-1)][0])
                negative_reference.append((user_key,user2_key))
        x_part1 += positive_list_1
        x_part2 += positive_list_2
        user_reference += positive_reference
        x_part1 += negative_list_1
        x_part2 += negative_list_2
        user_reference += negative_reference
        y += [1 for positive_sample in positive_list_1]
        y += [0 for negative_sample in negative_list_1]
    return x_part1, x_part2, y, user_reference

def average_evaluate(model, samples_x, samples_y, threshold=0.5):
    """Average evaluating function, groups multiple predicts into one value,
    intended to be used with combine_for_average.

    Arguments:
        model {tensorflow.python.keras.engine.functional.Functional} -- Model to be evaluated
        samples_x {array-like} -- Array-like input for a model predict
        samples_y {list} -- List of true values

    Keyword Arguments:
        threshold {float} -- Threshold to consider a positive predict (default: {0.5})

    Returns:
        float, (float,float,float,float) -- general_accuracy, (tp, fp, fn, tn)
    """
    predictions = model.predict(samples_x)
    past_label = samples_y[0]
    pred_sum = []
    true_predictions = []
    true_labels = []
    for index, label in enumerate(samples_y):
        if label == past_label:
            pred_sum.append(predictions[index])
        else:
            true_predictions.append(numpy.mean(pred_sum))
            true_labels.append(samples_y[index-1])
            pred_sum = [predictions[index]]
            past_label = label
    true_predictions.append(numpy.mean(pred_sum))
    true_labels.append(samples_y[index])
    eval_results = evaluate_predicted_values(true_predictions,true_labels,threshold)
    return eval_results

def evaluate_predicted_values(predictions,true_labels,threshold=0.5):
    """Evaluates a given predictions list to its true labels list, returning the general
    accuracy of the predictions, as well as the ratios of true/false - positives/negatives.

    Arguments:
        predictions {list} -- List of predictions
        true_labels {list} -- List of true labels

    Keyword Arguments:
        threshold {float} -- Threshold to consider a prediction as true (default: {0.5})

    Returns:
        float, (float,float,float,float) -- general_accuracy, (tp, fp, fn, tn)
    """
    true_accepted = 0
    false_accepted = 0
    true_rejected = 0
    false_rejected = 0
    for i in zip([0 if (pred < threshold) else 1 for pred in predictions],true_labels):
        if i[0] == 1 and i[1] == 1:
            true_accepted += 1
        elif i[0] == 1 and i[1] == 0:
            false_accepted += 1
        elif i[0] == 0 and i[1] == 0:
            true_rejected += 1
        elif i[0] == 0 and i[1] == 1:
            false_rejected += 1
        else:
            raise
    total_samples = len(true_labels)
    general_accuracy = (true_accepted+true_rejected)/total_samples
    tp = true_accepted/total_samples
    fp = false_accepted/total_samples
    fn = false_rejected/total_samples
    tn = true_rejected/total_samples
    return general_accuracy, (tp, fp, fn, tn)

def create_single_user_average_data(users_dict,user_key,n_samples_for_average=5):
    working_feature = copy.deepcopy(users_dict[user_key])
    X = []
    y = []
    other_users = list(users_dict.keys())
    other_users.remove(user_key)
    while len(working_feature) >= n_samples_for_average:
        other_user = random.choice(other_users)
        max_index = len(users_dict[other_user]) - n_samples_for_average
        other_index = random.randint(0,max_index)
        for i in range(n_samples_for_average):
            X.append(working_feature.pop(0))
            y.append(1)
        for i in range(n_samples_for_average):
            X.append(users_dict[other_user][other_index + i])
            y.append(0)
    return X,y

def number_of_samples_analysis(untreated_raw_dict):
    max_time_results = []
    min_action_lenght_results = []
    max_times = numpy.linspace(0.5, 2.5, num=11)
    min_action_lenghts = list(range(5,61,1))
    analysisbar = tqdmn(total=len(max_times)+len(min_action_lenghts), desc="Analysing Number of Samples")
    for max_time in max_times:
        working_dict = copy.deepcopy(untreated_raw_dict)
        treated_dict = treat_errors(working_dict,max_time,28,debug=False)
        prov_list = []
        for user_key in treated_dict:
            prov_list.append(len(treated_dict[user_key]))
        max_time_results.append(numpy.mean(prov_list))
        analysisbar.update(1)
    for min_action_lenght in min_action_lenghts:
        working_dict = copy.deepcopy(untreated_raw_dict)
        treated_dict = treat_errors(working_dict,1.325,min_action_lenght,debug=False)
        prov_list = []
        for user_key in treated_dict:
            prov_list.append(len(treated_dict[user_key]))
        min_action_lenght_results.append(numpy.mean(prov_list))
        analysisbar.update(1)
    analysisbar.close()
    return max_time_results, min_action_lenght_results

def analysis_plot(f,results_list,labels,title=False,xlabel=False,ylabel=False,xlim=False,ylim=False,tight=False):
    ax = fig.add_axes([0,0,1,1])
    ax.set_axisbelow(True)
    ax.scatter(labels,results_list,s=65,marker='D',color='k')
    if tight:
        ticks = labels[::2]
    else:
        ticks = labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title,pad=15)
    if xlabel:
        ax.set_xlabel(xlabel,labelpad=15)
    if ylabel:
        ax.set_ylabel(ylabel,labelpad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return f, ax