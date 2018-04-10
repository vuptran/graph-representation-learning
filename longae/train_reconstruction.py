#!/usr/bin/env python2.7
"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs reconstruction
using only latent features learned from local graph topology.

Usage: python train_reconstruction.py <dataset_str> <gpu_id>
"""

import sys
if len(sys.argv) < 3:
    print('\nUSAGE: python %s <dataset_str> <gpu_id>' % sys.argv[0])
    sys.exit()
dataset = sys.argv[1]
gpu_id = sys.argv[2]

import numpy as np
import scipy.sparse as sp
from keras import backend as K

from utils import create_adj_from_edgelist, compute_precisionK
from utils import generate_data, batch_data, split_train_test
from longae.models.ae import autoencoder

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


print('\nLoading dataset {:s}...\n'.format(dataset))
try:
    adj = create_adj_from_edgelist(dataset)
except IOError:
    sys.exit('Supported strings: {arxiv-grqc, blogcatalog}')

original = adj.copy()
train = adj.copy()
missing_edges = split_train_test(dataset, adj, ratio=0.0)
if len(missing_edges) > 0:
    r = missing_edges[:, 0]
    c = missing_edges[:, 1]
    train[r, c] = -1.0
    train[c, r] = -1.0
    adj[r, c] = 0.0
    adj[c, r] = 0.0

print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder(dataset, adj)
print ae.summary()

# Specify some hyperparameters
epochs = 50
train_batch_size = 8
val_batch_size = 256

print('\nFitting autoencoder model...\n')
dummy = np.empty(shape=(adj.shape[0], 1))
feats = sp.csr_matrix(dummy.copy())
y_true = dummy.copy()
mask = dummy.copy()

train_data = generate_data(adj, train, feats, y_true, mask, shuffle=True)
batch_data = batch_data(train_data, train_batch_size)
num_iters_per_train_epoch = adj.shape[0] / train_batch_size
for e in xrange(epochs):
    print('\nEpoch {:d}/{:d}'.format(e+1, epochs))
    print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
    curr_iter = 0
    train_loss = []
    for batch_adj, batch_train, dummy_f, dummy_y, dummy_m in batch_data:
        # Each iteration/loop is a batch of train_batch_size samples
        res = ae.train_on_batch([batch_adj], [batch_train])
        train_loss.append(res)
        curr_iter += 1
        if curr_iter >= num_iters_per_train_epoch:
            break
    train_loss = np.asarray(train_loss)
    train_loss = np.mean(train_loss, axis=0)
    print('Avg. training loss: {:6f}'.format(train_loss))
print('\nEvaluating reconstruction performance...')
reconstruction = np.empty(shape=adj.shape, dtype=np.float32)
for step in xrange(adj.shape[0] / val_batch_size + 1):
    low = step * val_batch_size
    high = low + val_batch_size
    batch_adj = adj[low:high].toarray()
    if batch_adj.shape[0] == 0:
        break
    reconstruction[low:high] = ae.predict_on_batch([batch_adj])
print('Computing precision@k...')
k = [10, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
if dataset == 'blogcatalog':
    k = k[:1] + [item*10 for item in k[1:]]
precisionK = compute_precisionK(original, reconstruction, np.max(k))
for index in k:
    if index == 0:
        index += 1
    print('Precision@{:d}: {:6f}'.format(index, precisionK[index-1]))
print('\nAll Done.')        

