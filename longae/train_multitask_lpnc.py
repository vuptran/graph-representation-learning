#!/usr/bin/env python2.7
"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs multi-task
learning for link prediction and semi-supervised node classification
using latent features learned from local graph topology and available
explicit node features. The following datasets are supported:
{cora, citeseer, pubmed}.

Usage: python train_multitask_lpnc.py <dataset_str> <gpu_id>
"""

import sys
if len(sys.argv) < 3:
    print('\nUSAGE: python %s <dataset_str> <gpu_id>' % sys.argv[0])
    sys.exit()
dataset = sys.argv[1]
gpu_id = sys.argv[2]

import numpy as np
from keras import backend as K
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import average_precision_score as ap_score

from utils_gcn import load_citation_data, split_citation_data
from utils import generate_data, batch_data
from utils import compute_masked_accuracy
from longae.models.ae import autoencoder_multitask

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


print('\nLoading dataset {:s}...\n'.format(dataset))
try:
    adj, feats, y_train, y_val, y_test, mask_train, mask_val, mask_test = load_citation_data(dataset)
except IOError:
    sys.exit('Supported strings: {cora, citeseer, pubmed}')
feats = MaxAbsScaler().fit_transform(feats).tolil()
train = adj.copy()

test_inds = split_citation_data(adj)
test_inds = np.vstack({tuple(row) for row in test_inds})
test_r = test_inds[:, 0]
test_c = test_inds[:, 1]
labels = []
labels.extend(np.squeeze(adj[test_r, test_c].toarray()))
labels.extend(np.squeeze(adj[test_c, test_r].toarray()))

multitask = True
if multitask:
    # If multitask, simultaneously perform link prediction and
    # semi-supervised node classification on incomplete graph with
    # 10% held-out positive links and same number of negative links.
    # If not multitask, perform node classification with complete graph.
    train[test_r, test_c] = -1.0
    train[test_c, test_r] = -1.0
    adj[test_r, test_c] = 0.0
    adj[test_c, test_r] = 0.0

adj.setdiag(1.0)
if dataset != 'pubmed':
    train.setdiag(1.0)

print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder_multitask(dataset, adj, feats, y_train)
adj = sp.hstack([adj, feats]).tolil()
train = sp.hstack([train, feats]).tolil()
print ae.summary()

# Specify some hyperparameters
epochs = 100
train_batch_size = 64
val_batch_size = 256

print('\nFitting autoencoder model...\n')
train_data = generate_data(adj, train, feats,
                           y_train, mask_train, shuffle=True)
batch_data = batch_data(train_data, train_batch_size)
num_iters_per_train_epoch = adj.shape[0] / train_batch_size
y_true = y_val
mask = mask_val
for e in xrange(epochs):
    print('\nEpoch {:d}/{:d}'.format(e+1, epochs))
    print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
    curr_iter = 0
    train_loss = []
    for batch_a, batch_t, batch_f, batch_y, batch_m in batch_data:
        # Each iteration/loop is a batch of train_batch_size samples
        batch_y = np.concatenate([batch_y, batch_m], axis=1)
        res = ae.train_on_batch([batch_a, batch_f], [batch_t, batch_y])
        train_loss.append(res)
        curr_iter += 1
        if curr_iter >= num_iters_per_train_epoch:
            break
    train_loss = np.asarray(train_loss)
    train_loss = np.mean(train_loss, axis=0)
    print('Avg. training loss: {:s}'.format(str(train_loss)))
    print('\nEvaluating validation set...')
    lp_scores, nc_scores, predictions = [], [], []
    for step in xrange(adj.shape[0] / val_batch_size + 1):
        low = step * val_batch_size
        high = low + val_batch_size
        batch_adj = adj[low:high].toarray()
        batch_feats = feats[low:high].toarray()
        if batch_adj.shape[0] == 0:
            break
        decoded = ae.predict_on_batch([batch_adj, batch_feats])
        decoded_lp = decoded[0] # link prediction scores
        decoded_nc = decoded[1] # node classification scores
        lp_scores.append(decoded_lp)
        nc_scores.append(decoded_nc)
    lp_scores = np.vstack(lp_scores)
    predictions.extend(lp_scores[test_r, test_c])
    predictions.extend(lp_scores[test_c, test_r])
    print('Val AUC: {:6f}'.format(auc_score(labels, predictions)))
    print('Val AP: {:6f}'.format(ap_score(labels, predictions)))
    nc_scores = np.vstack(nc_scores)
    node_val_acc = compute_masked_accuracy(y_true, nc_scores, mask)
    print('Node Val Acc {:f}'.format(node_val_acc))
print('\nAll done.')

