import numpy as np
import itertools
#import matplotlib.pyplot as plt
#from imblearn import over_sampling
#from sklearn import cluster, base, preprocessing, metrics
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.svm import SVC

"""
Methods:
0 - geometric integration
1 - voting
2 - support accumulation
"""

class CWS(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=2, random_state=None, method=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.method = method

    def fit(self, X, y):
        # Store basic informations about training set
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        # Clusters
        # Prepare storage for clusters
        training_samples = [[] for i in range(self.n_classes)]
        training_labels = [[] for i in range(self.n_classes)]
        self.cluster_centroids = [[] for i in range(self.n_classes)]

        # Iterate every class in training set
        for j, label in enumerate(self.classes):
            # Get training samples only from class
            class_X_train = X[y == label,:]
            class_y_train = y[y == label]

            # Clusterize them
            clusters = KMeans(n_clusters=self.n_clusters,
                                      random_state=self.random_state).fit(
                                          class_X_train).predict(
                                              class_X_train)

            # Iterate every cluster
            for i in range(self.n_clusters):
                #print("\tCluster %i" % i)
                cluster_X_train = class_X_train[clusters==i]
                cluster_y_train = class_y_train[clusters==i]

                centroid = np.median(cluster_X_train, axis = 0)

                # And store them
                self.cluster_centroids[j].append(centroid)
                training_samples[j].append(cluster_X_train)
                training_labels[j].append(cluster_y_train)

        self.cluster_centroids = np.array(self.cluster_centroids)
        # print(self.cluster_centroids.shape)

        # Possible combinations
        self.products = list(itertools.product(list(range(self.n_clusters)),
                                          repeat=self.n_classes))

        self.ensemble = []
        self.centroids = []

        self.X_ = []
        self.y_ = []

        # Train models
        for product in self.products:
            TS = []
            TL = []
            for i in range(self.n_classes):
                TS.append(training_samples[i][product[i]])
                TL.append(training_labels[i][product[i]])
            TS = np.concatenate(TS)
            TL = np.concatenate(TL)

            centroid = np.mean(TS, axis = 0)


            clf = SVC(gamma="scale", kernel="linear").fit(TS, TL)
            #clf = base.clone(self.base_clf)
            #clf.fit(TS, TL)

            self.X_.append(TS)
            self.y_.append(TL)

            self.ensemble.append(clf)
            self.centroids.append(centroid)

        self.centroids = np.array(self.centroids)

    def decision_function(self, X):
        # Calculate and transpone supports
        supports = np.array([clf.decision_function(X)
                             for clf in self.ensemble]).T

        # Prepare weights
        weights = np.zeros((len(supports), len(self.products)))

        for i, support in enumerate(supports):
            x = X[i]
            w = np.zeros(len(self.products))
            for j, product in enumerate(self.products):
                cent_a = self.cluster_centroids[0, product[0], :]
                cent_b = self.cluster_centroids[1, product[1], :]

                d_a = np.sum(abs(x - cent_a))
                d_b = np.sum(abs(x - cent_b))
                d_c = np.abs(support[j])

                #print(j, d_a, d_b, d_c)

                #print("a", d_a)
                #print("b", d_b)
                #print("c", d_c)

                w[j] = (d_a + d_b + d_c) / 3
                #w[j] = d_c

            # Weight calculation
            w -= np.min(w)
            if np.max(w) != 0:
                w /= np.max(w)
            w = 1 - w
            weights[i] = w


        weighted_supports = np.multiply(supports, weights)
        supports = np.sum(weighted_supports, axis=1)

        return supports

    def predict(self, X):
        """
        Methods:
        0 - geometric integration
        1 - voting
        2 - support accumulation
        """
        if self.method == 0:
            supports = self.decision_function(X)
            y_pred = (supports > 0).astype(int)
            return y_pred
        elif self.method==1:
            pred_mat = np.array([clf.predict(X) for clf in self.ensemble])
            y_pred = np.median(pred_mat, axis=0)
            tie_mask = y_pred == .5
            random_decision = np.random.choice(self.classes, size=len(y_pred))
            y_pred[tie_mask] = random_decision[tie_mask]
            y_pred = y_pred.astype(int)

            return y_pred
        elif self.method==2:
            esm = np.array([clf.decision_function(X) for clf in self.ensemble])
            #print(esm, esm.shape)
            sv = np.mean(esm, axis=0)
            #print(sv, sv.shape)
            y_pred = (sv > 0).astype(int)
            #print(y_pred)
            return y_pred
            #print("WRONG")
            #exit()

    def score(self, X, y):
        supports = self.decision_function(X)

        y_pred = (supports > 0).astype(int)
        score = metrics.balanced_accuracy_score(y, y_pred)

        return score
