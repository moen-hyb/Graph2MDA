

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf           
from tqdm import tqdm

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_label():
    print('loading labels...')
    labels = np.loadtxt("../data/adj.txt")

    temp_label = np.zeros((1373, 173))
    for temp in labels:
        temp_label[int(temp[0]) - 1, int(temp[1]) - 1] = int(temp[2])
    labels = temp_label


def load_data(graph_type,args):

    print('loading adj...')
    P = {}
    P_v ={}

    if graph_type == "net1":
        P = sio.loadmat('../data/net1.mat')
        P_v = P['interaction']
    elif graph_type == "net2":
        P = sio.loadmat('../data/net2.mat')
        P_v = P['net2']

        P = np.vstack((np.hstack((np.zeros(shape=(1373,1373),dtype=int), P_v)),np.hstack((P_v.transpose(),np.zeros(shape=(173,173),dtype=int)))))
        interaction = preprocess_adj(P)


    interaction = sp.csr_matrix(interaction)
    attributes_list = []
    print('loading attributes...')
    for attribute in args.attributes:
        if attribute =='features':
            F1 = np.loadtxt("../data/drug_features.txt")
            F2 = np.loadtxt("../data/microbe_features.txt")
            feature = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                                  np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))
            attributes_list.append(feature)
        elif attribute =='similarity':
            F1 = np.loadtxt("../data/drug_similarity.txt")
            F2 = np.loadtxt("../data/microbe_similarity.txt")
            similarity = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                                     np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))
            attributes_list.append(similarity)

    features = np.hstack(attributes_list)
    features = normalize_features(features)
    features = sp.csr_matrix(features)

    num_drug = F1.shape[0]
    num_microbe =F2.shape[0]

    print('loading labels...')
    labels = np.loadtxt("../data/adj.txt")

    temp_label = np.zeros((num_drug, num_microbe))
    for temp in labels:
        temp_label[int(temp[0])-1, int(temp[1])-1] = int(temp[2])
    labels = temp_label

    return interaction, features, labels, num_drug, num_microbe


def load_dataset(dataset):

    print('loading adj...')
    P = {}
    P_v ={}

    print('loading attributes...')

    F1 = np.loadtxt("../data/"+dataset+"/drug_similarity.txt")
    F2 = np.loadtxt("../data/"+dataset+"/virus_similarity.txt")
    similarity = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                             np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))

    similarity = sp.csr_matrix(similarity)

    num_drug = F1.shape[0]
    num_microbe =F2.shape[0]

    print('loading labels...')
    labels = np.loadtxt("../data/"+dataset+"/adj.txt")

    P_v = labels
    P = np.vstack((np.hstack((np.zeros(shape=(num_drug, num_drug), dtype=int), P_v)),
                   np.hstack((P_v.transpose(), np.zeros(shape=(num_microbe, num_microbe), dtype=int)))))
    interaction = preprocess_adj(P)
    interaction = sp.csr_matrix(interaction)

    return interaction, similarity, labels, num_drug, num_microbe

def generate_mask(labels,N):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((1*N,2)) 
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            label_neg[num,0]=a
            label_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])  
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()  
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(      [0] +shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    # adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    adj_normalized = adj +np.eye(adj.shape[0])

    return adj_normalized

