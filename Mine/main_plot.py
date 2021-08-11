import argparse
import csv

import numpy as np
import json
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import io
from inits import load_data
from trainGcn import train_gcn
from trainNN import train_nn
from evaluation import get_results, evaluate_performance_two, evaluate_performance,get_max_avg_results,get_metrics

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

    save_path = os.path.join("ppv_10k")
    version = '0809'

    if os.path.exists("results/" + version + '/' + save_path + "_" + version + ".mat") == False:
        print("Training", args.data_path)

        embeddings_list = []
        for graph in args.graphs:
            print("#############################")
            print("Training", graph)
            adj, features, labels, num_drug, num_microbe = load_data(graph, args)
            embeddings = train_gcn(features, adj, args, graph, num_drug, num_microbe)
            embeddings_list.append(embeddings)

        embeddings = np.hstack(embeddings_list)
        io.savemat("results/" + version + '/' + save_path + "_" + version + ".mat", {'embeddings': embeddings, 'labels': labels})
    else:
        mat = io.loadmat("results/" + version + '/' + save_path + "_" + version + ".mat")
        embeddings = mat['embeddings']
        labels = mat['labels']

    k_folds = args.k_folds

    # split data into train and test
    num_test = int(np.floor(labels.shape[0] / k_folds))
    num_train = labels.shape[0] - num_test
    all_idx = list(range(labels.shape[0]))

    perf_list = []
    y_score_list = []
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
        perf = evaluate_performance_two(Y_test.flatten(), y_pred, y_score.flatten().tolist())
        print(perf)
        perf_list.append(perf)

        y_score_list.append(y_score.tolist())

    print("###################################")
    perf_result = get_max_avg_results(perf_list)


    with open("results/" + version + '/' + save_path + "_150_" + version + ".json", "w") as f:
        json.dump(perf_result, f)


def plot_embedding(input_vector, k):
    print("....plot tsne....")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # tsne = TSNE(n_components=2)
    tsne.fit_transform(input_vector)
    X_tsne = tsne.embedding_

    # 可以画出低维散点图：
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    x = [xy[1] for xy in X_norm]
    y = [xy[0] for xy in X_norm]

    plt.scatter(x, y, c='#000000', s=100, marker='.')
    plt.show()

    # plot_ATC(x,y)
def plot_ATC(x,y):
    drugs = []
    with open('../tool/atc.csv', 'r', encoding='UTF-8-sig') as f:
        reader = csv.reader(f)
        for i in reader:
            drugs.append(i)
    f.close()

    atc = ['#808080' for i in range(len(x))]
    colormap = {'A': '#FFEBCD',
                'B': '#FF1493',
                'C': '#008000',
                 'D': '#7FFFD4',
                 'E': '#006400',
                 'F': '#FFA500',
                 'G': '#F5F5DC',
                 'H': '#FFE4C4',
                 'I': '#483D8B',
                 'J': '#FFFF00',
                 'K': '#0000FF',
                 'L': '#8A2BE2',
                 'M': '#A52A2A',
                 'N': '#DEB887',
                 'O': '#5F9EA0',
                 'P': '#7FFF00',
                 'Q': '#D2691E',
                 'R': '#FF7F50',
                 'S': '#6495ED',
                 'T': '#FFF8DC',
                 'U': '#DC143C',
                 'V': '#00FFFF',
                 'W': '#00008B',
                 'X': '#008B8B',
                 'Y': '#B8860B',
                 'Z': '#AD5FD7'}

    for drug in drugs:
        atc[int(drug[0])-1] = colormap.get(drug[2][0])
    plt.scatter(x, y, c=atc, s=100, marker='.')
    plt.show()

if __name__ == '__main__':
      main()
