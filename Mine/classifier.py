from trainNN import train_nn
from trainML import train_RF,train_GBDT,train_AdaBoost,train_SVM,train_xgboost
from evaluation import get_results, evaluate_performance_two, evaluate_performance,get_max_avg_results
import numpy as np

def classifiers(perf_all,args,X_train, Y_train, X_test, Y_test):
    # ml : 'RF', 'GBDT', 'AdaBoost', 'SVM', 'xgboost'
    x_train = X_train.flatten()[:, np.newaxis]
    y_train = Y_train.flatten()
    asa = np.where(y_train == 1)
    x_test = X_test.flatten()[:, np.newaxis]
    y_test = Y_test.flatten()
    for classifier in args.classifiers:
        if classifier == 'nn':
            y_score = train_nn(X_train, Y_train, X_test, Y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['nn'].append(perf)
        if classifier == 'RF':
            y_score = train_RF(X_train, Y_train, X_test, Y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['RF'].append(perf)
        if classifier == 'GBDT':
            y_score = train_GBDT(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['GBDT'].append(perf)
        if classifier == 'AdaBoost':
            y_score = train_AdaBoost(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['AdaBoost'].append(perf)
        if classifier == 'xgboost':
            y_score = train_xgboost(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['xgboost'].append(perf)
        if classifier == 'SVM':
            y_score = train_SVM(x_train, y_train, x_test, y_test)
            y_pred = [int(item > args.threshold) for item in y_score.flatten()]
            perf = evaluate_performance_two(y_test, y_pred)
            print(perf)
            perf_all['SVM'].append(perf)

    return perf_all