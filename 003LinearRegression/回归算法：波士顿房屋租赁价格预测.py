import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


def notEmpty(s):
    return s != ''


mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
path = "datas/boston_housing.data"
# 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再按照每行数据进行处理
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)

x, y = np.split(data, (13,), axis=1)
y = y.ravel()

print("样本数据量:%d, 特征个数：%d" % x.shape)
print("target样本数据量:%d" % y.shape[0])

# 数据的分割，
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))

# 标准化
ss = MinMaxScaler()

x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

print("原始数据各个特征属性的调整最小值:", ss.min_)
print("原始数据各个特征属性的缩放数据值:", ss.scale_)

# 构建模型（回归）
model = DecisionTreeRegressor(criterion='mae', max_depth=7)
# 模型训练
model.fit(x_train, y_train)
# 模型预测
y_test_hat = model.predict(x_test)

# 评估模型
score = model.score(x_test, y_test)
print("Score：", score)

# 构建线性回归
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_test_hat = lr.predict(x_test)
lr_score = lr.score(x_test, y_test)
print("lr:", lr_score)
# 构建lasso
lasso = LassoCV(alphas=np.logspace(-3, 1, 20))
lasso.fit(x_train, y_train)
lasso_y_test_hat = lasso.predict(x_test)
lasso_score = lasso.score(x_test, y_test)
print("lasso:", lasso_score)
# 构建岭回归
ridge = RidgeCV(alphas=np.logspace(-3, 1, 20))
ridge.fit(x_train, y_train)
ridge_y_test_hat = ridge.predict(x_test)
ridge_score = ridge.score(x_test, y_test)
print("ridge:", ridge_score)

# 7. 画图
plt.figure(figsize=(12, 6), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'实际值')
plt.plot(ln_x_test, lr_y_test_hat, 'b-', lw=2, label=u'Linear回归，$R^2$=%.3f' % lr_score)
plt.plot(ln_x_test, lasso_y_test_hat, 'y-', lw=2, label=u'Lasso回归，$R^2$=%.3f' % lasso_score)
plt.plot(ln_x_test, ridge_y_test_hat, 'c-', lw=2, label=u'Ridge回归，$R^2$=%.3f' % ridge_score)
plt.plot(ln_x_test, y_test_hat, 'g-', lw=4, label=u'回归决策树预测值，$R^2$=%.3f' % score)
plt.xlabel(u'数据编码')
plt.ylabel(u'租赁价格')
plt.legend(loc='lower right')
plt.grid(True)
plt.title(u'波士顿房屋租赁数据预测')
plt.show()
