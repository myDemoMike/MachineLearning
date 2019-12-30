import os

from surprise import NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNBaseline, SVD, SVDpp, NMF
from surprise import Reader, Dataset, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

'''
基础算法/baseline algorithms
基于近邻方法(协同过滤)/neighborhood methods
矩阵分解方法/matrix factorization-based (SVD, PMF, SVD++, NMF)

'''

# 指定文件路径
file_path = os.path.expanduser('./data/popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)

# 使用NormalPredictor

algo = NormalPredictor()
trainset, testset = train_test_split(music_data, test_size=.25)
algo.fit(trainset)
predictions = algo.test(testset)
print(accuracy.rmse(predictions))

# 使用BaselineOnly

algo = BaselineOnly()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用基础版协同过滤

algo = KNNBasic()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用均值协同过滤

algo = KNNWithMeans()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用协同过滤baseline

algo = KNNBaseline()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用SVD

algo = SVD()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用SVD++

algo = SVDpp()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])

# 使用NMF

algo = NMF()
perf = cross_validate(algo, music_data, measures=['RMSE', 'MAE'])
