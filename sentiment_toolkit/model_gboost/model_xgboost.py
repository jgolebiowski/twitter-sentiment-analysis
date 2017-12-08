"""Model using Gradient boosted trees with XGboost"""
import xgboost as xgb
import numpy as np


class GBTrees(object):
    """Model using GBTrees"""

    def __init__(self):
        self.initialize_default_parameters()
        self.model = None

    def initialize_default_parameters(self):
        """Set the parameters to default values

        Parameters to be set
        ----------
        self.params : dict
            Dictionary with xgboost parameters
        self.num_trees : int
            Number of trees ot be trained
        """

        # Set num rounds
        self.num_trees = 200

        # ------ Main parameters
        self.params = {}

        """booster [default=gbtree]
        which booster to use, can be gbtree, gblinear or dart.
        gbtree and dart use tree based model while gblinear uses linear function."""
        self.params["booster"] = "gbtree"

        # Set number of threads
        self.params["nthread"] = 10

        # set versobity
        self.params["silent"] = True
        # set number of classes
        self.params["num_class"] = 6

        # ------ Tree Boosting params
        """eta [default=0.3, alias: learning_rate]
        step size shrinkage used in update to prevents overfitting. After each
        boosting step, we can directly get the weights of new features. and
        eta actually shrinks the feature weights to make the boosting process
        moreconservative"""
        self.params["eta"] = 0.3

        """max_depth [default=6]
        maximum depth of a tree, increase this value will make the model more
        complex / likely to be overfitting. 0 indicates no limit, limit is required
        for depth-wise grow policy."""
        self.params["max_depth"] = 5

        # ------ Additional parameters
        """objective [default=reg:linear]
        "reg:linear" --linear regression
        "reg:logistic" --logistic regression
        "binary:logistic" --logistic regression for binary classification, output
            probability
        "binary:logitraw" --logistic regression for binary classification, output
            score before logistic transformation
        "count:poisson" --poisson regression for count data, output mean of poisson
            distribution. Max_delta_step is set to 0.7 by default in poisson regression
            (used to safeguard optimization)
        "multi:softmax" --set XGBoost to do multiclass classification using the softmax
            objective, you also need to set num_class(number of classes)
        "multi:softprob" --same as softmax, but output a vector of ndata * nclass,
            which can be further reshaped to ndata, nclass matrix. The result contains
            predicted probability of each data point belonging to each class.
        "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
        "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution.
            It might be useful, e.g., for modeling insurance claims severity, or for any
            outcome that might be gamma-distributed
        "reg:tweedie" --Tweedie regression with log-link. It might be useful,
            e.g., for modeling total loss in insurance, or for any outcome that might
            be Tweedie-distributed."""
        self.params["objective"] = "multi:softmax"

    def train(self, X_train, Y_train, X_test=None, Y_test=None):
        """Train the model

        Parameters
        ----------
        X_train : np.array
            Training data with dims (n_examples, n_features)
        Y_train : np.array
            Training labels with dims (n_examples, )
        X_test : np.array
            Testing data with dims (n_examples, n_features)
        Y_test : np.array
            Test labels with dims (n_examples, )
        """

        # Wrap data into XBboost vars
        xg_train = xgb.DMatrix(X_train, label=Y_train)
        xg_test = xgb.DMatrix(X_test, label=Y_test)

        # Train
        watchlist = [(xg_train, "training_set"), (xg_test, "test_set")]
        self.model = xgb.train(self.params, xg_train, self.num_trees, watchlist)

    def predict(self, X_test):
        """Get predictions from a model

        Parameters
        ----------
        X_test : np.array
            Testing data with dims (n_examples, n_features)

        Returns
        -------
        np.array
            Prediction labels with dims (n_examples, )
        """

        xg_test = xgb.DMatrix(X_test)
        return self.model.predict(xg_test)

    def get_accuracy(self, X_test, Y_test):
        """Get the accuracy of the model

        Parameters
        ----------
        X_test : np.array
            Testing data with dims (n_examples, n_features)
        Y_test : np.array
            Test labels with dims (n_examples, )

        Returns
        -------
        float
            Accuracy
        """
        pred = self.predict(X_test)
        error_rate = np.sum(pred != Y_test) / Y_test.shape[0]

        return error_rate

    def predict_from_sentence(self, document, text2features):
        """Make prediction based on a given document

        Parameters
        ----------
        document : list of strings
            document to run predictions on given as a list of words
        text2features : Text2Features
            Class storing conversion routines for df-idf bow model
            from this toolkit, initialized for current dataset

        Returns
        -------
        int
            Predicted class
        """

        bow_vector = text2features.raw_document2bow(document)
        return int(self.predict(bow_vector))
