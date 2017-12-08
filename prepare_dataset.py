"""Control script for using the module"""
import pickle
import sentiment_toolkit.model_rnn as st_rnn
import sentiment_toolkit.model_gboost as st_gboost
import torch


def prepare_GloVe_data():
    filename = "dataset/dataset_pickled.pkl"
    with open(filename, "rb") as fp:
        data, labels, labels2names = pickle.load(fp)

    test_data, test_labels = st_rnn.embed_dataset(data[0: 7000], labels[0: 7000])
    print("Test:", len(test_data), test_data[0].size(), test_labels.size())

    train_data, train_labels = st_rnn.embed_dataset(data[7000:], labels[7000:])
    print("Train:", len(train_data), train_data[0].size(), train_labels.size())

    filename = "dataset/train_dataset.pkl"
    with open(filename, "wb") as fp:
        pickle.dump((train_data, train_labels, labels2names), fp)

    filename = "dataset/test_dataset.pkl"
    with open(filename, "wb") as fp:
        pickle.dump((test_data, test_labels, labels2names), fp)

    num_labels = [0 for item in labels2names]
    for item in test_labels.numpy()[:, 0]:
        num_labels[item] += 1
    print("\nLabels to names test:\n", labels2names, "\n", num_labels, "\n")

    num_labels = [0 for item in labels2names]
    for item in train_labels.numpy()[:, 0]:
        num_labels[item] += 1
    print("\nLabels to names train:\n", labels2names, "\n", num_labels, "\n")

    print(torch.sum((test_data[0][0, 0] - st_rnn.word2vector(data[0][0]))))
    print(test_data[0].size())


def prepare_BoW_data():
    filename = "dataset/dataset_pickled.pkl"
    with open(filename, "rb") as fp:
        data, labels, labels2names = pickle.load(fp)

    text2features = st_gboost.Text2features(data)
    data_final, lebels = text2features.dataset2bow(data, labels)

    test_data, test_labels = text2features.dataset2bow(data[0: 7000], labels[0: 7000])
    print("Test:", test_data.shape, test_labels.shape)

    train_data, train_labels = text2features.dataset2bow(data[7000:], labels[7000:])
    print("train:", train_data.shape, train_labels.shape)

    filename = "dataset/train_dataset_bow.pkl"
    with open(filename, "wb") as fp:
        pickle.dump((train_data, train_labels, labels2names, text2features), fp)

    filename = "dataset/test_dataset_bow.pkl"
    with open(filename, "wb") as fp:
        pickle.dump((test_data, test_labels, labels2names, text2features), fp)

    num_labels = [0 for item in labels2names]
    for item in test_labels:
        num_labels[item] += 1
    print("\nLabels to names test:\n", labels2names, "\n", num_labels, "\n")

    num_labels = [0 for item in labels2names]
    for item in train_labels:
        num_labels[item] += 1
    print("\nLabels to names train:\n", labels2names, "\n", num_labels, "\n")


if (__name__ == "__main__"):
    # prepare_GloVe_data()
    prepare_BoW_data()
