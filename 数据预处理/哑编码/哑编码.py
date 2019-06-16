from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
import numpy as np

enc = OneHotEncoder()
n = np.array([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 5], [1, 1, 1]])
enc.fit(n)
print(n)

enc.transform([[0, 1, 2]]).toarray()

# sparse:最终产生的结果是狗是稀疏矩阵，默认为True，一般不改动
dv = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2.2}, {'foo': 3, 'baz': 2}]
X = dv.fit_transform(D)
print(X)

# 直接把字典中的key作为特征，value作为特征值，然后构建特征矩阵
print(dv.get_feature_names())
print(dv.transform({'foo': 4, 'bar': 3}))

h = FeatureHasher(n_features=3)
M = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
f = h.transform(M)
print(f.toarray())