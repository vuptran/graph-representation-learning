# Multi-Task Graph Autoencoders

This is a Keras implementation of the symmetrical autoencoder architecture with parameter sharing for the tasks of link prediction and semi-supervised node classification, as described in the following:

Tran, Phi Vu. [Learning to Make Predictions on Graphs with Autoencoders.](https://arxiv.org/abs/1802.08352) Proceedings of the 5th IEEE International
Conference on Data Science and Advanced Analytics (2018). Full oral paper.

Tran, Phi Vu. [Multi-Task Graph Autoencoders.](https://arxiv.org/abs/1811.02798) NIPS 2018 Workshop on Relational Representation Learning. Short poster paper.

![schematic](figure1.png?raw=true)

## Requirements
The code is tested on Ubuntu 16.04 with the following components:

### Software

* Python 2.7
* Keras 2.0.6 using TensorFlow GPU 1.1.0 backend
* CUDA 8.0 with CuDNN 5.1
* NetworkX 1.11
* NumPy 1.11
* SciPy 0.17.0
* Scikit-Learn 0.18.1

### Hardware

* Intel Xeon CPU with 32 cores
* 64GB of system RAM
* NVIDIA GeForce GTX TITAN X GPU with 12GB of VRAM

### Datasets

Citation networks from [Thomas Kipf and Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks](https://github.com/tkipf/gcn):

* `Cora`, `Citeseer`, `Pubmed`

Collaboration and social networks from [Wang et al. 2016. Structural Deep Network Embedding](https://github.com/suanrong/SDNE):

* `Arxiv-GRQC`, `BlogCatalog`

Miscellaneous networks from [Aditya Krishna Menon and Charles Elkan. 2011. Link Prediction via Matrix Factorization](http://users.cecs.anu.edu.au/~akmenon/papers/link-prediction/index.html):

* `Protein`, `Metabolic`, `Conflict`, `PowerGrid`

For custom graph datasets, the following are required:

* N x N adjacency matrix (N is the number of nodes) [required for link prediction],
* N x F matrix of node features (F is the number of features per node) [optional for link prediction],
* N x C matrix of one-hot label classes (C is the number of classes) [required for node classification].

For an example of how to prepare the input dataset, take a look at the `load_citation_data()` function in `utils_gcn.py`.

## Usage
For training and evaluation, execute the following `bash` commands in the same directory where the code resides:

```bash
# Set the PYTHONPATH environment variable
$ export PYTHONPATH="/path/to/this/repo:$PYTHONPATH"

# Train the autoencoder model for network reconstruction
# using only latent features learned from local graph topology.
$ python train_reconstruction.py <dataset_str> <gpu_id>

# Train the autoencoder model for link prediction using
# only latent features learned from local graph topology.
$ python train_lp.py <dataset_str> <gpu_id>

# Train the autoencoder model for link prediction using
# both latent graph features and available explicit node features.
$ python train_lp_with_feats.py <dataset_str> <gpu_id>

# Train the autoencoder model for the multi-task
# learning of both link prediction and semi-supervised
# node classification, simultaneously.
$ python train_multitask_lpnc.py <dataset_str> <gpu_id>
```

The flag `<dataset_str>` refers to one of the following nine supported dataset strings:
`protein`, `metabolic`, `conflict`, `powergrid`, `cora`, `citeseer`, `pubmed`, `arxiv-grqc`, `blogcatalog`. The flag `<gpu_id>` denotes the GPU device ID, `0` by default if only one GPU is available.

## Citation
If you find this work useful, please cite the following:

```
@inproceedings{Tran-LoNGAE:2018,
  author={Tran, Phi Vu},
  title={Learning to Make Predictions on Graphs with Autoencoders},
  booktitle={5th IEEE International Conference on Data Science and Advanced Analytics},
  year={2018}
}
