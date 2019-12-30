import tensorflow as tf

#  定义一个变量，必须给定初始值
w1 = tf.Variable(tf.random_normal(shape=[10], stddev=0.5, seed=28, dtype=tf.float32), name='w1')
a = tf.Variable(initial_value=3.0, dtype=tf.float32)
w2 = tf.Variable(w1.initialized_value() * a, dtype=tf.float32)

# 定义一个张量
b = tf.constant(value=2.0, dtype=tf.float32)
c = tf.add(a, b)

# 进行初始化操作(推荐：使用全局所有变量初始化API)
init_op = tf.global_variables_initializer()
print(type(init_op))

# 图的运行
with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    # 运行init op进行变量初始化，一定要放在所有运行操作之前
    sess.run(init_op)
    # init_op.run() 运行代码也是初始化运行操作，但是要求明确给定当前代码块对应的默认session(tf.get_default_session())来运行
    #
    print("result:{}".format(sess.run(w1)))
    print("result:{}".format(w2.eval()))

# 构建一个矩阵的乘法，但是矩阵再运行的时候给定
m1 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_m1')
m2 = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_m2')
m3 = tf.matmul(m1, m2)

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    print("result:\t{}".format(
        sess.run(fetches=[m3], feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})
    ))
    print("result:\n{}".format(m3.eval(feed_dict={m1: [[1, 2, 3],   [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]})))
