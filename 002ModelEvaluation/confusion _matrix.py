import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# 混淆矩阵
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
#                 header=None)
df = pd.read_csv('wdbc.data', header=None)
# (569, 32)
print(df.head())
ss = StandardScaler()
print("原始数据")
X = df.loc[:, 2:].values
tr3 = ss.fit_transform(X)
print(tr3)

print(df.shape)
# loc 选取行和列的值


print(df.loc[:, 2:].shape)
y = df.loc[:, 1].values
print(y.shape)
# fit()：训练算法，设置内部参数。
# transform()：数据转换。
# fit_transform()：合并fit和transform两个方法。


# 把字符串类型的数据转化为整型
le = LabelEncoder()
Y = le.fit_transform(y)  # 类标整数化
print(Y)

# 为使各特征的均值为0，方差为1
print(StandardScaler().fit_transform(df.loc[:, 2:]))

# 划分训练集合测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
# 建立pipeline
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
# fit 训练
pipe_svc.fit(X_train, y_train)
y_predict = pipe_svc.predict(X_test)
# 混淆矩阵并可视化
confmat = confusion_matrix(y_true=y_test, y_pred=y_predict)  # 输出混淆矩阵
print(confmat)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
# 召回率、准确率、F1
print('precision:%.3f' % precision_score(y_true=y_test, y_pred=y_predict))
print('recall:%.3f' % recall_score(y_true=y_test, y_pred=y_predict))
print('F1:%.3f' % f1_score(y_true=y_test, y_pred=y_predict))
