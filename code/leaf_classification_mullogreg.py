import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

np.random.seed(0)

train = pd.read_csv('../input/train.csv')
# print train.head()
x_train = train.drop(['id', 'species'], axis=1).values

le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

x_train = StandardScaler().fit(x_train).transform(x_train)

mul_logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

params = {'C': [100, 1000], 'tol': [0.001, 0.0001]}
clf = GridSearchCV(mul_logreg, params, scoring='neg_log_loss', refit='True', n_jobs=1, cv=5)
clf.fit(x_train, y_train)

print("best params: " + str(clf.best_params_))
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f for %r" % (mean_score, params))
    print(scores)

test = pd.read_csv('../input/test.csv')
test_ids = test.pop('id')
x_test = test.values
x_test = StandardScaler().fit(x_test).transform(x_test)
y_test = clf.predict_proba(x_test)
print y_test.mean()


# submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
# submission.to_csv('submission_log_reg.csv')