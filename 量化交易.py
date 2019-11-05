# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    # 在context中保存全局变量
    context.s1 = "000001.XSHE"
    # 实时打印日志
    #　logger.info("RunInfo: {}".format(context.run_info))
    # 获取计算机通信行业的一些公司股票代码
    context.stock_list = industry("C39")
    # 版块
    context.sector_list = sector("energy")
    #　经常会调用指数成分股的接口
    #  获取沪深300的指数股  实时更新   指数股？
    # 股票价格指数为度量和反映股票市场总体价格水平及其变动趋势而编制的股价统计相对数。通常是报告期的股票平均价格或股票市值与选定的基期股票平均价格或股票市值相比，并将两者的比值乘以基期的指数值，即为该报告期的股票价格指数。
    # 相当于股票池
    context.index_list = index_components("000300.XSHG")


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    #　print(context.stock_list)
    #　print(context.sector_list)
    print(context.index_list)


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    pass
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # TODO: 开始编写你的算法吧！
    # order_shares(context.s1, 1000)

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass