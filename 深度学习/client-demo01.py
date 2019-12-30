"""
分布式集群使用
"""

import tensorflow as tf
import numpy as np

# 1.构造图
with tf.device('/job:ps/task:0/gpu:0'):
    # 2.构造数据
    x = tf.constant(np.random.rand(100).astype(tf.float32))

# 3.使用另外一台机器
with tf.device('/job:ps/task:1'):
    y = x * 0.1 + 0.3

# 4.运行
with tf.Session(target='grpc://127.0.0.1:33335',
                config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    print(sess.run(y))
