import tensorflow as tf

# 1.定义常量矩阵a和矩阵b
a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')
print(type(a))
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')
print(type(b))
# 2.以a和b作为输入，进行矩阵的乘法操作
c = tf.matmul(a, b, name='matmul')

# 矩阵相加
g = tf.add(a, c, name='add')

# 矩阵相减
h = tf.subtract(b, a, name='subtract')
# 矩阵乘法  dot
l = tf.matmul(h, c)
r = tf.add(g, l)

# 会话的构建和启动
sess = tf.Session()
print(sess)

# 调用sess的run方法来执行矩阵的乘法，得到c的结果值（所以将c作为参数传递进去）。
# 不需要考虑图中间的运算，在运行的时候只需要关注最终结果对应的对象以及所需要的输入数据值。
# 只需要传递进去所需要得到的结果对象，会自动的根据图中的依賴关系触发所有相关的op操作的执行。
# 如果op之间美欧依賴关系,tensorflow底层会并行的执行op（又资源）--> 自动进行
# 如果传递的fetches 是一个列表，那么返回值是一个list集合
result = sess.run(fetches=[c, r])
print("type:{},value{}".format(type(result), result))

# 会话的关闭。当一个会话关闭后，不能再使用。
sess.close()

# sess2 = tf.Session()
# with sess2.as_default(): 没有自动关闭
# with tf.Session() as sess2:   可以自动关闭
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
with tf.Session() as sess2:
    print(sess2)
    # 获取张量c的结果：通过Session的run方法获取
    print("sess2 run {}".format(sess2.run(c)))
    # 获取张量c的结果：通过张量对象eval方法获取，和Session的run方法一直
    print("c eval {}".format(c.eval()))

# 交互式会话构建
sess3 = tf.InteractiveSession()
print(r.eval)