import jieba.analyse

words_a = "上海自来水来自海上，所以吃葡萄不吐葡萄皮"

seg_a = jieba.cut(words_a, cut_all=True)
print("全模式：", "/".join(seg_a))

seg_b = jieba.cut(words_a)
print("准确模式：", "/".join(seg_b))

seg_c = jieba.cut_for_search(words_a)
print("搜索引擎模式：", "/".join(seg_c))

seg_d = jieba.cut(words_a)
for w in seg_d:
    print(w)

words_al = "我为机器学习疯狂打call"
print("自定义前：", "/".join(jieba.cut(words_al)))
jieba.del_word("打call")
jieba.add_word("打call")
print("加入‘打call’后：", "/".join(jieba.cut(words_al)))

# 导入自定义词典
jieba.del_word("打call")
words_a2 = "在正义者联盟的电影里，嘻哈侠和蝙蝠侠联手打败了大boss,我高喊666，为他们疯狂打call"
print("加载自定义词库前：", "/".join(jieba.cut(words_a2)))
jieba.load_userdict("../data/mydict.txt")
print("---------------------")
print("加载自定义词库后：", "/".join(jieba.cut(words_a2)))

# 获得切词后的数据
ls1 = []
for item in jieba.cut(words_a2):
    ls1.append(item)
print(ls1)

# 用lcut直接获得切词后的lise列表数据

ls2 = jieba.lcut(words_a2)
print(ls2)

# 调整词典，关闭HMM发现新词功能

jieba.suggest_freq(("中", "将"), False)
print('/'.join(jieba.cut("如果放到post中将出错。", HMM=False)))
jieba.suggest_freq(("中", "将"), True)  # suggest_freq 可调节单个词语的词频，使其能（或不能）被分出来  默认为false，会被切开
print('/'.join(jieba.cut("如果放到post中将出错。", HMM=False)))
print('/'.join(jieba.cut("[台中]正确应该不会被切开", HMM=False)))
jieba.suggest_freq('台中', True)
print('/'.join(jieba.cut("[台中]正确应该不会被切开", HMM=False)))

# 获取TF-IDF最大的20个单词
print(jieba.analyse.extract_tags(words_a2, topK=20, withWeight=True))
