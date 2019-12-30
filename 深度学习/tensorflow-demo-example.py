import tensorflow as tf

# 需求一
"""
使用已介绍的相关TensorFlow相关知识点，实现以下三个功能(变量更新)#
1. 实现一个累加器，并且每一步均输出累加器的结果值。
2. 编写一段代码，实现动态的更新变量的维度数目
3. 实现一个求解阶乘的代码
"""

# 1.定义一个变量
x = tf.Variable(0, dtype=tf.int32, name='v_x')

# 2.变量的更新
assign_op = tf.assign(ref=x, value=x + 1)

# 变量初始化操作
x_init_op = tf.global_variables_initializer()

# 3.运行
with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(x_init_op)

    # 模拟迭代更新累加器
    for i in range(5):
        r_x = sess.run(x)
        print(r_x)
        sess.run(assign_op)

# 需求二
# 1.定义一个不定形状的变量
y = tf.Variable(
    initial_value=[],  # 给定一个空值
    dtype=tf.float32,
    trainable=False,
    validate_shape=False  # 设置为True，表示在变量更新的时候，进行shape的检查，默认为True。如果你想要在之后改变变量的形状,就设置为false 默认为True
)

# 2.变量更改
concat = tf.concat([y, [0.0, 0.0]], axis=0)
assign_op_y = tf.assign(y, concat, validate_shape=False)
# 变量初始化操作
y_init_op = tf.global_variables_initializer()

# 3.运行
with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(y_init_op)

    # 模拟迭代更新累加器
    for i in range(5):
        r_y = sess.run(y)
        print(r_y)
        sess.run(assign_op_y)

# 阶乘
# 定义一个变量
sum = tf.Variable(1, dtype=tf.int32)

# 定义一个占位符
i = tf.placeholder(dtype=tf.int32)

# 更新操作
tmp_sum = sum * i
# tmp_sum = tf.multiply(sum,i)
assign_op = tf.assign(sum, tmp_sum)

# 控制依赖
with tf.control_dependencies([assign_op]):
    # 如果需要执行这个代码块中的内容，必须先执行control_dependencies中给定的操作/tensor
    sum = tf.Print(sum, data=[sum, sum.read_value()], message='sum:')

# 变量的初始化操作
v_init_op = tf.global_variables_initializer()

# 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 变量初始化
    sess.run(v_init_op)
    for j in range(1, 6):
        # 通过control_dependencies可以指定依赖关系，这样的话，就不用管内部的更新操作了
        r = sess.run(sum, feed_dict={i: j})
    print("5!={}".format(r))
