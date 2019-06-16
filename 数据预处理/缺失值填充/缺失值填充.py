import numpy as np
from sklearn.preprocessing import Imputer

X = [
    [2, 2, 4, 1],
    [np.nan, 3, 4, 4],
    [1, 1, 1, np.nan],
    [2, 2, np.nan, 3]
]

X2 = [
    [2, 6, np.nan, 1],
    [np.nan, 5, np.nan, 1],
    [4, 1, np.nan, 5],
    [np.nan, np.nan, np.nan, 1]
]

# 按照列进行填充值   mean median 中位数 most_frequent 众数
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
# 按照行进行计算填充值（如果按照行进行填充的话，那么是不需要进行模型fit的，直接使用X现有的行信息进行填充）
imp2 = Imputer(missing_values='NaN', strategy='mean', axis=1)

imp1.fit(X)
imp2.fit(X)
print(imp1.statistics_)

print(imp1.transform(X2))
print("-----------------------------")
print(imp2.transform(X2))
print(X2)


