from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/feature03.csv')
Y = df['loan_status']
X = df.drop('loan_status', 1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lr = LogisticRegression()
start = time.time()
lr.fit(x_train, y_train)
train_predict = lr.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train)
train_acc = metrics.accuracy_score(train_predict, y_train)
train_rec = metrics.recall_score(train_predict, y_train)
print("逻辑回归模型上的效果如下：")
print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
print("在训练集上的准确率的值为%.4f" % train_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % train_rec)
test_predict = lr.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test)
test_acc = metrics.accuracy_score(test_predict, y_test)
test_rec = metrics.recall_score(test_predict, y_test)
print("在测试集上f1_mean的值为%.4f" % test_f1, end=' ')
print("在测试集上的准确率的值为%.4f" % test_acc, end=' ')
print("在测试集上的查全率的值为%.4f" % test_rec)
end = time.time()
print(end - start)

print("随机森林" + "=" * 30)
rf = RandomForestClassifier()
start1 = time.time()
rf.fit(x_train, y_train)
train_predict = rf.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train)
train_acc = metrics.accuracy_score(train_predict, y_train)
train_rec = metrics.recall_score(train_predict, y_train)
print("随机森林模型上的效果如下：")
print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
print("在训练集上的准确率的值为%.4f" % train_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % train_rec)
test_predict = rf.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test)
test_acc = metrics.accuracy_score(test_predict, y_test)
test_rec = metrics.recall_score(test_predict, y_test)
print("在测试集上f1_mean的值为%.4f" % test_f1, end=' ')
print("在测试集上的准确率的值为%.4f" % test_acc, end=' ')
print("在测试集上的查全率的值为%.4f" % test_rec)
end1 = time.time()
print(end1 - start1)

# 特征程度排名
feature_importance = rf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
index = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[index], color='dodgerblue', alpha=0.4)
print(np.array(X.columns)[index])
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[index])
plt.xlabel('Relative importance')
plt.title('Top 10 Importance Variable')
plt.show()

# Gradient Boosting Regression Tree
# param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [1,2,3,4],
#               'min_samples_split': [50,100,200,400],
#               'n_estimators': [100,200,400,800]
#               }

param_grid = {'learning_rate': [0.1],
              'max_depth': [2],
              'min_samples_split': [50, 100],
              'n_estimators': [100, 200]
              }
# param_grid = {'learning_rate': [0.1],
#               'max_depth': [4],
#               'min_samples_leaf': [3],
#               'max_features': [1.0],
#               }

est = GridSearchCV(ensemble.GradientBoostingRegressor(),
                   param_grid, n_jobs=4, refit=True)

est.fit(x_train, y_train)

best_params = est.best_params_
print(best_params)

est = ensemble.GradientBoostingRegressor(min_samples_split=50, n_estimators=300,
                                         learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(x_train,
                                                                                                        y_train)
est.score(x_test, y_test)

est = ensemble.GradientBoostingRegressor(min_samples_split=50, n_estimators=100,
                                         learning_rate=0.1, max_depth=2, random_state=0, loss='ls').fit(x_train,
                                                                                                        y_train)
est.score(x_test, y_test)


def compute_ks(data):
    sorted_list = data.sort_values(['predict'], ascending=[True])

    total_bad = sorted_list['label'].sum(axis=None, skipna=None, level=None, numeric_only=None) / 3
    total_good = sorted_list.shape[0] - total_bad

    # print "total_bad = ", total_bad
    # print "total_good = ", total_good

    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():
        if row['label'] == 3:
            bad_count += 1.0
        else:
            good_count += 1.0

        val = bad_count / total_bad - good_count / total_good
        max_ks = max(max_ks, val)

    return max_ks


test_pd = pd.DataFrame()
test_pd['predict'] = est.predict(x_test)
test_pd['label'] = y_test
# df['predict'] = est.predict(x_test)
print(compute_ks(test_pd[['label', 'predict']]))

# Top Ten
feature_importance = est.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

import xgboost as xgb

# XGBoost
clf2 = xgb.XGBClassifier(n_estimators=50, max_depth=1,
                         learning_rate=0.01, subsample=0.8, colsample_bytree=0.3, scale_pos_weight=3.0,
                         silent=True, nthread=-1, seed=0, missing=None, objective='binary:logistic',
                         reg_alpha=1, reg_lambda=1,
                         gamma=0, min_child_weight=1,
                         max_delta_step=0, base_score=0.5)

clf2.fit(x_train, y_train)
print(clf2.score(x_test, y_test))
test_pd2 = pd.DataFrame()
test_pd2['predict'] = clf2.predict(x_test)
test_pd2['label'] = y_test
print(compute_ks(test_pd[['label', 'predict']]))
print(clf2.feature_importances_)
# Top Ten
feature_importance = clf2.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices], color='dodgerblue', alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# RFR
clf3 = RandomForestRegressor(n_jobs=-1, max_depth=10,random_state=0)
clf3.fit(x_train, y_train)
print (clf3.score(x_test, y_test))
test_pd3 = pd.DataFrame()
test_pd3['predict'] = clf3.predict(x_test)
test_pd3['label'] = y_test
print (compute_ks(test_pd[['label','predict']]))
print (clf3.feature_importances_)
# Top Ten
feature_importance = clf3.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])

# XTR
clf4 = ExtraTreesRegressor(n_jobs=-1, max_depth=10,random_state=0)
clf4.fit(x_train, y_train)
print (clf4.score(x_test, y_test))
test_pd4 = pd.DataFrame()
test_pd4['predict'] = clf4.predict(x_test)
test_pd4['label'] = y_test
print (compute_ks(test_pd[['label','predict']]))
print( clf4.feature_importances_)
# Top Ten
feature_importance = clf4.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')



# 特征工程方法1：histogram
def get_histogram_features(full_dataset):
    def extract_histogram(x):
        count, _ = np.histogram(x, bins=[0, 10, 100, 1000, 10000, 100000, 1000000, 9000000])
        return count
    column_names = ["hist_{}".format(i) for i in range(8)]
    hist = full_dataset.apply(lambda row: pd.Series(extract_histogram(row)), axis=1)
    hist.columns= column_names
    return hist
# 特征工程方法2：quantile
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
column_names = ["quantile_{}".format(i) for i in q]
# print pd.DataFrame(train_x)
quantile = pd.DataFrame(x_train).quantile(q=q, axis=1).T
quantile.columns = column_names
# 特征工程方法3：cumsum
# def get_cumsum_features(all_features):
#     column_names = ["cumsum_{}".format(i) for i in range(len(all_features))]
#     cumsum = full_dataset[all_features].cumsum(axis=1)
#     cumsum.columns = column_names
#     return cumsum
# 特征工程方法4：特征归一化
# from sklearn.preprocessing import MinMaxScaler
# Scaler = MinMaxScaler()
# x_train_normal = Scaler.fit_transform(x_train_normal)
