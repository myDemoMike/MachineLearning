

"""
运行命令
python server-demo.py --job_name=ps --task_index=0
python server-demo.py --job_name=ps --task_index=1
python server-demo.py --job_name=work --task_index=0
python server-demo.py --job_name=work --task_index=1
python server-demo.py --job_name=work --task_index=2

"""


import tensorflow as tf



# 1.配置服务器相关信息
# 因为tensorflow底层代码中，默认就是使用ps和work分别表示两类不同的工作节点
# ps:变量/张量的初始化，存储相关节点
# work:变量/张量的计算/运算的相关节点
ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']
work_hosts = ['127.0.0.1:33333', '127.0.0.1:33334', '127.0.0.1:33335']
cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'work': work_hosts})

# 2.定义一些运行参数
tf.app.flags.DEFINE_string('job_name', default_value='work', docstring="One Of 'ps' or 'work'")
tf.app.flags.DEFINE_integer('task_index', default_value=0, docstring='Index Of Task Within The Job')
FLAGS = tf.app.flags.FLAGS


# 2.启动服务
def main(_):
    print(FLAGS.job_name)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    server.join()



if __name__ == '__main__':
    # 底层默认会调用main方法
    tf.app.run()
