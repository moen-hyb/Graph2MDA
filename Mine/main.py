import argparse
import numpy as np
import json
import os
from inits import load_data
from trainGcn import train_gcn
from trainNN import train_nn
from evaluation import get_results, evaluate_performance_two, evaluate_performance,get_max_avg_results

import os
os.environ['KMP_WARNINGS'] = '0'
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #global parameters
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")], default=('net1', 'net2'), help="lists of graphs to use.")
    parser.add_argument('--attributes', type=lambda s: [item for item in s.split(",")], default=('features','similarity'),help=" attributes ： features similarity")

    parser.add_argument('--data_path', type=str, default='MDAD', help="lists of dataset : MDAD  , DrugVirus, aBiofilm")
    #parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.0005, help="Initial learning rate. best 0.0005")
    parser.add_argument('--epochs_net1', type=int, default=300, help="Number of epochs to train net1. best 150")
    parser.add_argument('--epochs_net', type=int, default=200, help="Number of epochs to train net.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1. best 800 400")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--threshold', type=float, default=0.02, help="threshold for two classes.")
    parser.add_argument('--k_folds', type=int, default=10, help="kflod")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
    args = parser.parse_args()
    print(args)

    print("Training", args.data_path)
    dataset = args.data_path

    for time in range(5):
        print("#############################")
        graph = 'net1'
        for dataset in args.data_path:
            print("#############################")
            print("Training", graph)
            adj, features, labels, num_drug, num_microbe = load_data(dataset)
            embeddings = train_gcn(features, adj, args, graph, num_drug, num_microbe)

            k_folds = args.k_folds

            # split data into train and test
            num_test = int(np.floor(labels.shape[0] / k_folds))
            num_train = labels.shape[0] - num_test
            all_idx = list(range(labels.shape[0]))

            perf_list = []
            print("###################################")
            for k in range(k_folds):
                print("------this is %dth cross validation------" % (k + 1))
                np.random.seed(k)
                np.random.shuffle(all_idx)

                train_idx = all_idx[:num_train]
                test_idx = all_idx[num_train:(num_train + num_test)]

                Y_train = labels[train_idx]

                Y_test = labels[test_idx]

                X_train = embeddings[train_idx]
                X_test = embeddings[test_idx]

                print('....train_nn....')
                y_score = train_nn(X_train, Y_train, X_test, Y_test)

                print("---------------------------------")
                print('....evalution....')

                y_pred = [int(item > args.threshold) for item in y_score.flatten()]
                perf = evaluate_performance_two(Y_test.flatten(), y_pred)
                print(perf)
                perf_list.append(perf)
            print("###################################")
            perf_result = get_max_avg_results(perf_list)

            save_path = os.path.join("results_" + str(args.model))
            version = '0506'
            with open("results/" + version + '/' + save_path + "_" + version + ".json", "w") as f:
                json.dump(perf_result, f)



if __name__ == '__main__':
    main()