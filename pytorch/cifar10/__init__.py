"""Dataset cifar-10"""
import pickle
import os
import numpy as np
import torch
from matplotlib import pyplot as plt


def where_am_i():
    return os.path.dirname(__file__)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_single(integer):
    """Unpickle a single batch, 0 for test batch"""
    if integer == 0:
        d = unpickle(where_am_i() + "/test_batch")
    else:
        d = unpickle(where_am_i() + "/data_batch_" + str(integer))

    data = d[b"data"].reshape(10000, 3, 32, 32).astype("uint8")
    data = data.astype("float32")
    data -= 127.5
    data /= -127.5

    labels = np.asarray(d[b"labels"])
    labels.reshape(-1, 1)

    labNames = unpickle(where_am_i() + "/batches.meta")
    labNames = labNames[b"label_names"]

    return labels, data, labNames


def load_single_torch(integer):
    l, d, ln = load_single(integer)

    l = torch.from_numpy(l)
    d = torch.from_numpy(d)

    return l, d, ln


def load_multiple():
    """Load all the data"""
    dd = np.empty((0, 3, 32, 32))
    ll = np.empty(0, dtype=int)

    for i in range(1, 6):
        l, d, ln = load_single(i)
        dd = np.vstack((dd, d))
        ll = np.hstack((ll, l))

    dtest, ltest, _ = load_single(0)

    return ll, dd, ltest, dtest, ln


def load_multiple_torch():
    ll, dd, ltest, dtest, ln = load_multiple()

    dd = torch.from_numpy(dd)
    ll = torch.from_numpy(ll)
    dtest = torch.from_numpy(dtest)
    ltest = torch.from_numpy(ltest)

    return ll, dd, ltest, dtest, ln


def display_image(integ, data, labels, ln, transpose=True):
    if transpose:
        plt.imshow(data[integ].transpose(1, 2, 0) * 127.5)
    else:
        plt.imshow(data[integ] * 127.5)

    plt.title("It is a " + ln[labels[integ]].decode())
    plt.show()
