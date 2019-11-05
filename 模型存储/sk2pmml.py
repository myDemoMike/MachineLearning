#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import linear_model

import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, FunctionTransformer
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper

heart_data = pandas.read_csv("heart.csv")
# 用Mapper定义特征工程
mapper = DataFrameMapper([
    (['sbp'], MinMaxScaler()),
    (['tobacco'], MinMaxScaler()),
    ('ldl', None),
    ('adiposity', None),
    (['famhist'], LabelBinarizer()),
    ('typea', None),
    ('obesity', None),
    ('alcohol', None),
    (['age'], FunctionTransformer(np.log))
])

# 用pipeline定义使用的模型，特征工程等
pipeline = PMMLPipeline([
    ('mapper', mapper),
    ("classifier", linear_model.LinearRegression())
])

pipeline.fit(heart_data[heart_data.columns.difference(["chd"])], heart_data["chd"])
sklearn2pmml(pipeline, "lrHeart.xml", with_repr=True)
