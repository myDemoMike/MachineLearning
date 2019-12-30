import tensorflow as tf

# 变量
# a = tf.constant([1, 2, 3], dtype=tf.float32)
# b = tf.constant(2.0, dtype=tf.float32)
# c = tf.add(a, b)
# # init_op = tf.initialize_all_variables()
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     # sess.run(init_op)
#     print("a={}".format(sess.run(a)))
#     print("c={}".format(c.eval()))
#
#
# # 变量作用域 方式一
# def my_func(x):
#     w1 = tf.Variable(tf.random_normal([1]))[0]
#     b1 = tf.Variable(tf.random_normal([1]))[0]
#     r1 = w1 * x + b1
#
#     w2 = tf.Variable(tf.random_normal([1]))[0]
#     b2 = tf.Variable(tf.random_normal([1]))[0]
#     r2 = w2 * x + b2
#
#     return r1, w1, b1, r2, w2, b2
#
#
# # 下面2行代码还是属于图的构建。所以变量必须放在初始化之前
# x = tf.constant(3, dtype=tf.float32)
# r = my_func(x)
#
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(r))
#
#
# # 变量作用域 方式二
# def my_func2(x):
#     # initializer 初始化器
#     w3 = tf.get_variable(name='w', shape=[1], initializer=tf.random_normal_initializer())[0]
#     b3 = tf.get_variable(name='b', shape=[1], initializer=tf.random_normal_initializer())[0]
#     r3 = w3 * x + b3
#     return r3, w3, b3
#
#
# def fun(x):
#     #  可重复使用
#     with tf.variable_scope('op1', reuse=tf.AUTO_REUSE):
#         r4 = my_func2(x)
#     with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):
#         r5 = my_func2(r4[0])
#     return r4, r5
#
#
# x5 = tf.constant(3, dtype=tf.float32)
# x6 = tf.constant(4, dtype=tf.float32)
# r6 = fun(x5)
# r7 = fun(x6)
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(r6))
#     print(sess.run(r7))

# 可视化 TensorBoard   tensorboard --logdir E:\workspace2\MachineLearning\深度学习\result
# with tf.device("/cpu:0"):
#     with tf.variable_scope("foo"):
#         x_init1 = tf.get_variable('init_x', [10], tf.float32, initializer=tf.random_normal_initializer())[0]
#         x = tf.Variable(initial_value=x_init1, name='x')
#         y = tf.placeholder(dtype=tf.float32, name='y')
#         z = x + y
#     with tf.variable_scope("bar"):
#         a = tf.constant(3.0) + 4.0
#     w = z * a
#
# # 开始记录信息(需要展示信息的输出)
# tf.summary.scalar('scalar_init_x', x_init1)
# tf.summary.scalar('scalar_x', x)
# tf.summary.scalar('scalar_y', y)
# tf.summary.scalar('scalar_z', z)
# tf.summary.scalar('scalar_w', w)
#
# # update x
# assign_op = tf.assign(x, x + 1)
#
# with tf.control_dependencies([assign_op]):
#     with tf.device("/gpu:0"):
#         out = x * y
#     tf.summary.scalar('scalar_out', out)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
#     # merge all summary
#     merged_summary = tf.summary.merge_all()
#     # 得到输出到文件的对象
#     writer = tf.summary.FileWriter('./result', sess.graph)
#     # 初始化
#     sess.run(tf.global_variables_initializer())
#     # print
#     for i in range(1, 5):
#         summary, r_out, r_x, r_w = sess.run([merged_summary, out, x, w], feed_dict={y: i})
#         writer.add_summary(summary, i)
#         print("{},{},{}".format(r_out, r_x, r_w))

# v1 = tf.Variable(tf.constant(3.0), name='v11')
# v2 = tf.Variable(tf.constant(5.0), name='v22')
# result = v1 + v2
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # 模型保存到model文件下，文件前缀为：./model.ckpt
#     saver.save(sess, './model/model.ckpt')
#
# # 模型的提取（1.完整提取：需要完整恢复保存之前的数据格式）
# with tf.Session() as sess:
#     # 会从对应的文件中加载变量、图等信息
#     saver.restore(sess,'./model/model.ckpt')
#     print(sess.run([result]))

# 2.直接加载图，不需要定义变量了
# saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
# with tf.Session() as sess:
#     # 会从对应的文件中加载变量、图等信息
#     saver.restore(sess,'./model/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


# 3.
a = tf.Variable(tf.constant(3.0), name='a')
b = tf.Variable(tf.constant(999.0), name='b')
result = a + b
saver = tf.train.Saver({"v11": a, "v22": b})
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run([result]))
