import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from strlearn import metrics
from scipy.spatial.distance import cityblock

"""
Methods:
0 - geometric integration
1 - voting
2 - support accumulation
"""


class CWS_ENC(BaseEstimator, ClassifierMixin):
    def __init__(self, max_clusters=5, min_samples=25, random_state=None, method=0, metric=cityblock):
        self.max_clusters = max_clusters
        self.min_samples = min_samples
        self.random_state = random_state
        self.method = method
        self.metric = metric

    def fit(self, X, y):
        # Store basic informations about training set
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        # Prepare storage for clusters
        training_samples = [[] for i in range(self.n_classes)]
        training_labels = [[] for i in range(self.n_classes)]
        self.cluster_centroids = [[] for i in range(self.n_classes)]
        self.combinations = []
        self.class_n_clusters = []
        self.cluster_models = []

        # Iterate every class in training set
        for j, label in enumerate(self.classes):
            # Get training samples only from class
            class_X_train = X[y == label, :]
            class_y_train = y[y == label]
            if class_X_train.shape[0] < self.min_samples:
                samples = [np.array(class_X_train)]
                labels = [np.array(class_y_train)]
                cluster_centroids = np.mean(samples, axis=0)
                n_clusters = 1
                model = None
            else:
                samples, labels, cluster_centroids, n_clusters, model = self._best_number_of_clusters(
                    class_X_train, class_y_train, self.max_clusters)

            training_samples[j] = samples
            training_labels[j] = labels
            self.cluster_centroids[j] = cluster_centroids
            self.class_n_clusters.append(n_clusters)
            self.cluster_models.append(model)

        # training_samples = np.array(training_samples)
        # print(training_samples[1])
        self.ensemble = []
        self.centroids = []

        self.X_ = []
        self.y_ = []

        for idx in range(self.class_n_clusters[0]):
            for jdx in range(self.class_n_clusters[1]):
                self.combinations.append((idx, jdx))

        self.classifiers = {}

        # Train models
        for comb in self.combinations:
            TS = []
            TL = []
            for i in range(self.n_classes):
                # print(training_samples[i][comb[i]])
                # print(i)
                # print(comb[i])
                TS.append(training_samples[i][comb[i]])
                TL.append(training_labels[i][comb[i]])
            TS = np.concatenate(TS)
            TL = np.concatenate(TL)

            centroid = np.mean(TS, axis=0)

            clf = clone(SVC(gamma="scale", kernel="linear", random_state=self.random_state)).fit(TS, TL)
            # clf = base.clone(self.base_clf)
            # clf.fit(TS, TL)

            self.X_.append(TS)
            self.y_.append(TL)

            self.classifiers[comb] = clf
            self.ensemble.append(clf)
            self.centroids.append(centroid)

        self.centroids = np.array(self.centroids)

    def _best_number_of_clusters(self, data, y, kmax=10):
        score_vals = []
        clusters = []
        cluster_models = []
        if len(data) == 0:
            return None, None, 0
        for k in range(1, kmax + 1):
            try:
                cluster_model = KMeans(n_clusters=k, random_state=self.random_state)
                labels = cluster_model.fit_predict(data)
                clusters.append(labels)
                cluster_models.append(cluster_model)

                if k == 1:
                    score_vals.append(0)
                else:
                    score_vals.append(silhouette_score(data, labels))
            except Exception as ex:
                # print(ex)
                break

        best_number = np.argmax(np.array(score_vals))
        n_clusters = best_number + 1
        best_cluster = clusters[best_number]
        samples = []
        new_y = []
        for i in range(n_clusters):
            samples.append(data[best_cluster == i])
            new_y.append(y[best_cluster == i])

        if hasattr(cluster_models[best_number], "cluster_centers_"):
            cluster_centers = cluster_models[best_number].cluster_centers_
        else:
            cluster_centers = []
            for sp in samples:
                cluster_centers.append(np.mean(sp, axis=0))
            cluster_centers = np.array(cluster_centers)

        return samples, new_y, cluster_centers, n_clusters, cluster_models[best_number]


    def decision_function(self, X):
        # Calculate and transpone supports
        supports = np.array([clf.decision_function(X)
                             for clf in self.ensemble]).T

        # Prepare weights
        weights = np.zeros((len(supports), len(self.combinations)))

        for i, support in enumerate(supports):
            x = X[i]
            w = np.zeros(len(self.combinations))
            for j, comb in enumerate(self.combinations):
                cent_a = self.cluster_centroids[0][comb[0], :]
                cent_b = self.cluster_centroids[1][comb[1], :]

                d_a = self.metric(x, cent_a)
                d_b = self.metric(x, cent_b)

                d_c = np.abs(support[j])

                w[j] = (d_a + d_b + d_c) / 3

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
        elif self.method == 1:
            pred_mat = np.array([clf.predict(X) for clf in self.ensemble])
            y_pred = np.median(pred_mat, axis=0)
            tie_mask = y_pred == .5
            random_decision = np.random.choice(self.classes, size=len(y_pred))
            y_pred[tie_mask] = random_decision[tie_mask]
            y_pred = y_pred.astype(int)

            return y_pred
        elif self.method == 2:
            esm = np.array([clf.decision_function(X) for clf in self.ensemble])
            sv = np.mean(esm, axis=0)
            y_pred = (sv > 0).astype(int)
            return y_pred

    def score(self, X, y):
        supports = self.decision_function(X)

        y_pred = (supports > 0).astype(int)
        score = metrics.balanced_accuracy_score(y, y_pred)

        return score
