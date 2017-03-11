import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA

mul_clf = False
log_reg = True
use_pca = True
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

######################
# read the data
######################
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

######################
# define encode function
######################
def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings  **to numerical
    classes = list(le.classes_)  # save column names for submission  **like unique
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1) # drop off col
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
# print train.shape
# print test.shape
# print train.head(3)
# print len(labels) = 990
# print len(classes) = 99
if use_pca:
    # # We initialise pca choosing Minka's MLE to guess the minimum number of output components necessary
    # # to maintain the same information coming from the input descriptors and we ask to solve SVD in full
    pca = PCA(n_components='mle', svd_solver='full')
    train = pca.fit_transform(train)
    test = pca.fit_transform(test)
######################
# Stratified Train/Test Split
######################
sss = StratifiedShuffleSplit(10, test_size=0.2, random_state=23)  # random_state --- reproducible

for train_index, test_index in sss.split(train, labels):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
# print X_train.shape
# print X_test.shape
# print y_train.shape

if mul_clf:
    ######################
    # looping through classifiers
    ######################
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LogisticRegression(solver='lbfgs', multi_class='multinomial')]

    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        # print("=" * 30)
        # print(name)

        # print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)  # from sklearn
        # print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        # print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)
    # print("=" * 30)
    print log
    # LogisticRegression 59.090909 4.173216

######################
# I would like to choose
# LogisticRegression
# for future improvements by tuning their hyper-parameters
######################
if log_reg:
# We initialise the Exhaustive Grid Search, we leave the scoring as the default function of
# the classifier singe log loss gives an error when running with K-fold cross validation
# add n_jobs=-1 in a parallel computing calculation to use all CPUs available
# cv=3 increasing this parameter makes it too difficult for kaggle to run the script
    # kfold = KFold(n_splits=5, shuffle=True, random_state=4)
    params = {'C': [100, 1000], 'tol': [0.001, 0.0001]}
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clf = GridSearchCV(lr, params, scoring='neg_log_loss', refit='True', cv=5)
    # gs_validation = clf.fit(X_train, y_train).score(X_test, y_test)
    clf.fit(X_train, y_train)
    # print gs_validation
    # print("best params: " + str(clf.best_params_))
    # print clf.best_score_
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print acc
    print ll
    # print acc = 0.974747474747 for original
    # print ll = 0.268471158191

# 0.969696969697 for pca
# 0.269494464678

######################
# Submission
######################
#Predict Test Set
test_predictions = clf.predict_proba(test)
# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission.csv', index = False)
submission.tail()
