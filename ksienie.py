# A little module with some useful methods by Paweł Ksieniewicz.
import numpy as np
import os
import re
import json
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def json2object(path):
    with open(path) as json_data:
        return json.load(json_data)

def dat2Xy(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for cl in df.select_dtypes(include="object").columns:
        if cl != ("Class" or "class"):

            # ohe = OneHotEncoder()
            # data = df[cl].values.astype('U13').reshape(-1, 1)
            # ohe_res = ohe.fit_transform(data)
            # df2 = pd.DataFrame(ohe_res.toarray(), columns=ohe.categories_)
            # df = pd.concat((df2, df), axis=1, join="inner")
            # df.pop(cl)

            ord = OrdinalEncoder()
            data = df[cl].values.astype('U13').reshape(-1, 1)
            df[cl] = ord.fit_transform(data)

            # print(df)
        else:
            df[cl] = df[cl].values.astype('U13')
            df.loc[df[cl] == "positive", cl] = 1
            df.loc[df[cl] == "negative", cl] = 0

    X = df.iloc[:, 0:-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    dbname = path.split("/")[-1].split(".")[0]
    tags = tags4Xy(X, y)
    print(dbname)
    # print(df)
    return X, y, dbname, tags

def csv2Xy(path):
    ds = np.genfromtxt(path, delimiter=",")
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)
    dbname = path.split("/")[-1].split(".")[0]
    tags = tags4Xy(X, y)
    return X, y, dbname, tags


def dir2files(path, extention="csv"):
    return [
        path + x
        for x in os.listdir(path)
        if re.match("^([a-zA-Z0-9-_])+\.%s$" % extention, x)
    ]


def tags4Xy(X, y):
    tags = []
    numberOfFeatures = X.shape[1]
    numberOfSamples = len(y)
    numberOfClasses = len(np.unique(y))
    if numberOfClasses == 2:
        tags.append("binary")
    else:
        tags.append("multi-class")
    if numberOfFeatures >= 8:
        tags.append("multi-feature")

    # Calculate ratio
    ratio = [0.0] * numberOfClasses
    for y_ in y:
        ratio[y_] += 1
    ratio = [int(round(i / min(ratio))) for i in ratio]
    if max(ratio) > 4:
        tags.append("imbalanced")

    return tags


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def datasets_for_groups(ds_groups):
    datasets = []
    ds_dir = "keel"
    for group_idx, ds_group in enumerate(ds_groups):
        group_path = "%s/%s" % (ds_dir, ds_group)
        ds_list = sorted(os.listdir(group_path))
        for ds_idx, ds_name in enumerate(ds_list):
            if ds_name[0] == "." or ds_name[0] == "_":
                continue
            datasets.append((group_path, ds_name))
    return datasets


def load_dataset(dataset):
    group_path, ds_name = dataset
    # Load full dataset
    X, y = load_keel("%s/%s/%s.dat" % (group_path, ds_name, ds_name))
    X_, y_ = [], []
    # Load and process folds
    for i in range(1, 6):
        X_train, y_train = load_keel(
            "%s/%s/%s-5-fold/%s-5-%itra.dat"
            % (group_path, ds_name, ds_name, ds_name, i)
        )
        X_test, y_test = load_keel(
            "%s/%s/%s-5-fold/%s-5-%itst.dat"
            % (group_path, ds_name, ds_name, ds_name, i)
        )
        X_.append((X_train, X_test))
        y_.append((y_train, y_test))
    return (X, y, X_, y_)


def load_keel(string, separator=","):
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        raise Exception

    s = filter(lambda e: e[0] != "@", s)
    s = [v.strip().split(separator) for v in s]
    df = np.array(s)
    X = np.asarray(df[:, :-1], dtype=float)
    d = {"positive": 1, "negative": 0}
    y = np.asarray(
        [d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip() for v in s]
    )

    return X, y


def load_arff(filename):

    print("Start")

    with open(filename) as file:
        attr_count = 0
        for line in file:
            if line.startswith("@attribute Class"):
                continue
            elif line.startswith("@attribute"):
                attr_count += 1
            elif line.startswith("@data"):
                break

        df = pd.read_csv(file)
        features = df.iloc[:, 0:-1].values.astype(float)
        labels = df.iloc[:, -1].values.astype(str)
        labels[labels == 'positive'] = 1
        labels[labels == 'negative'] = 0
        labels = labels.astype(int)
        dbname = filename.split("/")[-1].split(".")[0]
        tags = tags4Xy(X, y)
        return features, labels, dbname, tags
