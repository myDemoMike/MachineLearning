import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split  # 数据划分的类
from sklearn.preprocessing import StandardScaler  # 数据标准化

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

path = "datas/breast-cancer-wisconsin.data"
names = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
         'Mitoses', 'Class']

df = pd.read_csv(path, header=None, names=names)

data = df.replace('?', np.nan).dropna(how='any')  # 只要有列为空，就进行删除操作

print(data.head(5))

# 1.数据提取以及数据分割
# 提取
X = data[names[1:10]]
Y = data[names[10]]

# 分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 2.数据格式化(归一化)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)  # 训练模型及归一化数据

# 3.模型构建及训练

lr = LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2',
                          solver='lbfgs', tol=0.01)
re = lr.fit(X_train, Y_train)

r = re.score(X_train, Y_train)

# 4.模型效果获取
print("R值(准确率):", r)
print("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("参数：", re.coef_)
print("截距：", re.intercept_)
print(re.predict_proba(X_test))  # 获取sigmoid函数返回的概率值

# 5.模型相关信息保存


joblib.dump(lr, "result/ss.model")

oss = joblib.load("result/ss.model")

# # 数据预测
#  a.预测数据格式化（归一化）
X_test = ss.transform(X_test)  # 使用模型进行归一化操作

# b.结果数据预测
Y_predict = oss.predict(X_test)

# 图标展示
x_len = range(len(X_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(0, 6)
plt.plot(x_len, Y_test, 'ro', markersize=8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize=15, zorder=2, label=u'预测值，$R^2$=%.3f' % re.score(X_test, Y_test))
plt.legend(loc='upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.xlabel(u'乳腺癌类型', fontsize=18)
plt.title(u'Logistic回归算法对数据进行分类', fontsize=20)
plt.show()
