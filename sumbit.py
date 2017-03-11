import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
# submission = pd.read_csv("sample_submission.csv")
# print submission.shape
test_ids = test['id']


def label(df):
    le = LabelEncoder()
    return le.fit_transform(df.values), le.classes_

def get_train_data(df, df1):
    df.drop(['species', 'id'], axis=1, inplace=True)
    df1.drop('id', axis=1, inplace=True)
    return df.values, df1.values


def reduce_dimension(train, test):
    # n_train = train.shape[0]
    # n_test = test.shape[0]
    # data = np.concatenate((train, test))
    # pca = PCA()
    # pca.fit(data)
    # score = 0
    # scores = pca.explained_variance_ratio_
    # num = 0
    # for variance in scores:
    #     score += variance
    #     num += 1
    #     if score > 0.99:
    #         break
    # reduce_data = PCA(n_components=num).fit_transform(data)
    # return reduce_data[:n_train, ], reduce_data[n_train:, ]
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train, test


label, colnames = label(train['species'])


x_train, x_test = get_train_data(train_data, test_data)
reduce_x, reduce_x_test = reduce_dimension(x_train, x_test)
X, xval, y, yval = train_test_split(reduce_x, label, test_size = 0.3, random_state=0)


## Test and Test the results of sevaral Classifiers
clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(X, y)
print clf.score(xval, yval)
print clf.predict_proba(xval).shape

clf2 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, random_state=0)
scores = cross_val_score(clf2, reduce_x, label)
print scores, scores.mean()

clf1 = KNeighborsClassifier(n_neighbors=4)
scores = cross_val_score(clf1, reduce_x, label)
print scores, scores.mean()

clf3 = LinearDiscriminantAnalysis()
scores = cross_val_score(clf3, reduce_x, label)
print scores, scores.mean()

clf4 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
scores = cross_val_score(clf4, reduce_x, label)
print scores, scores.mean()


#build an ensemble learning by GridSeachCV
eclf = VotingClassifier(estimators=[('SVM', clf),
                                    ('rf', clf2),
                                    ('KNN', clf1),
                                    #('ld', clf3),
                                    ('lgr', clf4)],
                                    voting='hard')
scores = cross_val_score(eclf, reduce_x, label)
print scores, scores.mean()
params = {'rf__n_estimators':[20, 200], 'KNN__n_neighbors': [2,5], 'lgr__C':[100, 1000], 'lgr__tol': [0.001, 0.0001]}
grid = GridSearchCV(estimator=eclf, param_grid=params, scoring='neg_log_loss', refit='True', cv = 5)
grid = grid.fit(reduce_x, label)



# print("best params: " + str(grid.best_params_))
# for params, mean_score, scores in grid.grid_scores_:
#   print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))
#   print(scores)
# print grid.best_estimator_
# print grid.cv_results_

#Some not good Classifier/ Dropped
# clf1 = AdaBoostClassifier(n_estimators=100, learning_rate=0.2)
# scores = cross_val_score(clf1, reduce_x, label)
# print scores, scores.mean()
#
# clf1 = GradientBoostingClassifier(n_estimators=100, max_depth=None, learning_rate=1, random_state=0)
# scores = cross_val_score(clf1, reduce_x, label)
# print scores, scores.mean()
#x_test = test_data.values
#reduce_test = reduce_dimension(x_test, 88)


#-------------#-------------#-------------#-------------#-------------#-------------#-------------
test_predictions = grid.predict_proba(reduce_x_test)
print test_predictions.shape, colnames.shape
# #Format DataFrame
# submission = pd.DataFrame(test_predictions, columns=colnames)
# submission.insert(0, 'id', test_ids)
# submission.reset_index()
#
# # Export Submission
# submission.to_csv('submission.csv', index = False)
# submission.tail()