#!/usr/bin/env python
# This script calculates effectiveness of all reference algorithms wrapped from
# skl and saves them to reference.csv.

import numpy as np
import json
from tqdm import tqdm
import ksienie as ks
from sklearn import svm
from sklearn import base
from sklearn import model_selection

from strlearn import metrics
import method_old as mo
import method as mt

from scipy.spatial import distance

rs = 111

# Initialize classifiers
classifiers = {
    "EUC": mt.CWS_ENC(method=0, metric=distance.euclidean, random_state=rs),
    "MAN": mt.CWS_ENC(method=0, metric=distance.cityblock, random_state=rs),
    "CHE": mt.CWS_ENC(method=0, metric=distance.chebyshev, random_state=rs),
    "JSH": mt.CWS_ENC(method=0, metric=distance.jensenshannon, random_state=rs),
    "BRY": mt.CWS_ENC(method=0, metric=distance.braycurtis, random_state=rs),
    "CAN": mt.CWS_ENC(method=0, metric=distance.canberra, random_state=rs),
    "COR": mt.CWS_ENC(method=0, metric=distance.correlation, random_state=rs),
    "SQE": mt.CWS_ENC(method=0, metric=distance.sqeuclidean, random_state=rs),

    # "SNE": mt.CWS_ENC(method=0, metric=distance.seuclidean, random_state=rs),
    # "MAH": mt.CWS_ENC(method=0, metric=distance.mahalanobis, random_state=rs),

    # "CWS": mo.CWS(method=0, random_state=rs),
    # "SVC": svm.SVC(gamma="scale", kernel="linear"),
    # "CVM": mo.CWS(method=1, random_state=rs),
    # "CSA": mo.CWS(method=2, random_state=rs),
}

# Choose metrics
used_metrics = {
    "BAC": metrics.balanced_accuracy_score,
    "F-1": metrics.f1_score,
    "REC": metrics.recall,
    "PRE": metrics.precision,
    "SPE": metrics.specificity,
    "GMN": metrics.geometric_mean_score_1,

}

# Gather all the datafiles and filter them by tags
files = ks.dir2files("datasets/")
files.sort()

tag_filter = ["imbalanced"]  # , "multi-class"]
datasets = []
for file in files:
    X, y, dbname, tags = ks.csv2Xy(file)
    intersecting_tags = ks.intersection(tags, tag_filter)
    if len(intersecting_tags):
        datasets.append((X, y, dbname))


# Prepare results cube
print(
    "# Experiment on %i datasets, with %i estimators using %i metrics."
    % (len(datasets), len(classifiers), len(used_metrics))
)
rescube = np.zeros((len(datasets), len(classifiers), len(used_metrics), 5))

# Iterate datasets
for i, dataset in enumerate(tqdm(datasets, desc="DBS", ascii=True, leave=False)):
    # load dataset
    X, y, dbname = dataset

    # Folds
    skf = model_selection.StratifiedKFold(n_splits=5, random_state=rs)
    for fold, (train, test) in enumerate(
        tqdm(skf.split(X, y), desc="FLD", ascii=True, total=5, disable=False, leave=False)
    ):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for c, clf_name in enumerate(tqdm(classifiers, desc="CLF", ascii=True, disable=False, leave=False)):
            clf = base.clone(classifiers[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            for m, metric_name in enumerate(tqdm(used_metrics, desc="MET", ascii=True, disable=False, leave=False)):
                try:
                    score = used_metrics[metric_name](y_test, y_pred)
                    rescube[i, c, m, fold] = score
                except:
                    rescube[i, c, m, fold] = np.nan

np.save("results/rescube", rescube)
with open("results/legend.json", "w") as outfile:
    json.dump(
        {
            "datasets": [obj[2] for obj in datasets],
            "classifiers": list(classifiers.keys()),
            "metrics": list(used_metrics.keys()),
            "folds": 5,
        },
        outfile,
        indent="\t",
    )

print("\n")
