import numpy as np
from sklearn.preprocessing import Binarizer

arr = np.array([[1.5, 2.3, 1.9], [0.5, 0.5, 1.6], [1.1, 2, 0.2]])
binarizer = Binarizer(threshold=2.0).fit(arr)

binarizer.transform(arr)

### 一般情况下，对于每个特征需要使用不同的阈值进行操作，所以一般我们会拆分成为几个DataFrame进行二值化操作，再将数据合并
### 一般情况下，对数据进行划分的时候，不是进行二值化，而是进行多值化（分区化/分桶化）。

