import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv("input/train.csv", index_col='id')
test_data = pd.read_csv("input/test.csv", index_col='id')
submission = pd.read_csv("input/sample_submission.csv")
print submission.shape
colnames = train_data.columns

def label(df):
    le = LabelEncoder()
    le.fit(df.values)
    return le.transform(df.values)

def get_train_data(df):
    df.drop('species', axis=1, inplace=True)
    return df.values


def reduce_dimension(train, dim = None):
    pca = PCA()
    pca.fit(train)
    score = 0
    scores = pca.explained_variance_ratio_
    num = 0
    for variance in scores:
        score += variance
        num += 1
        if score > 0.99:
            break
    if dim is None:
        return PCA(n_components=num).fit_transform(train)
    else:
        return PCA(n_components=dim).fit_transform(train)

def do_ml(X_train, y_train, X_test):
    clf = SVC()
    clf.fit(X, y)
    return clf.predict(X_test)

label = label(train_data[colnames[0]])

x_train = get_train_data(train_data)
reduce_x = reduce_dimension(x_train)
X, xval, y, yval = train_test_split(reduce_x, label, test_size = 0.3, random_state=0)

clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(X, y)
print clf.score(xval, yval)
print clf.predict_proba(xval).shape

clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, random_state=0)
scores = cross_val_score(clf, reduce_x, label)
print scores, scores.mean()
#x_test = test_data.values
#reduce_test = reduce_dimension(x_test, 88)


print X.shape
print reduce_x.shape
#print train_data.head()
