#!/usr/bin/env python2.7

import numpy as np
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations

np.random.seed(1982)


def generate_data(adj, adj_train, feats, labels, mask, shuffle=True):
    adj = adj.tocsr()
    adj_train = adj_train.tocsr()
    feats = feats.tocsr()
    zipped = zip(adj, adj_train, feats, labels, mask)
    while True: # this flag yields an infinite generator
        if shuffle:
            print('Shuffling data')
            np.random.shuffle(zipped)
        for data in zipped:
            a, t, f, y, m = data
            yield (a.toarray(), t.toarray(), f.toarray(), y, m)


def batch_data(data, batch_size):
    while True: # this flag yields an infinite generator
        a, t, f, y, m = zip(*[next(data) for i in xrange(batch_size)])
        a = np.vstack(a)
        t = np.vstack(t)
        f = np.vstack(f)
        y = np.vstack(y)
        m = np.vstack(m)
        yield map(np.float32, (a, t, f, y, m))


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    from keras import backend as K
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)


def load_mat_data(dataset_str):
    """ dataset_str: protein, metabolic, conflict, powergrid """
    dataset_path = 'data/' + dataset_str + '.mat'
    mat = loadmat(dataset_path)
    if dataset_str == 'powergrid':
        adj = sp.lil_matrix(mat['G'], dtype=np.float32)
        feats = None
        return adj, feats
    adj = sp.lil_matrix(mat['D'], dtype=np.float32)
    feats = sp.lil_matrix(mat['F'].T, dtype=np.float32)
    # Return matrices in scipy sparse linked list format
    return adj, feats
        

def split_train_test(dataset_str, adj, fold=0):
    if fold not in range(10):
        raise Exception('Choose fold in range [0,9]')
    upper_inds = [ind for ind in combinations(range(adj.shape[0]), r=2)]
    np.random.shuffle(upper_inds)
    test_inds = []
    for ind in upper_inds:
        rand = np.random.randint(0, 10)
        if dataset_str == 'powergrid':
        # Select 10% of {powergrid} dataset for test split
            boolean = (rand == fold)
        else:
        # Select 90% of {protein, metabolic, conflict} datasets
        # for test split
            boolean = (rand != fold)
        if boolean:
            test_inds.append(ind)
    return np.asarray(test_inds)


def compute_masked_accuracy(y_true, y_pred, mask):
    correct_preds = np.equal(np.argmax(y_true, 1), np.argmax(y_pred, 1))
    num_examples = float(np.sum(mask))
    correct_preds *= mask
    return np.sum(correct_preds) / num_examples

