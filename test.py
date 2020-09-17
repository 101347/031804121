import re
import numpy as np
import jieba
import jieba.analyse
from collections import Counter


# 文件转字符串
def string(file):
    with open(file, 'r', encoding='utf-8') as File:
        # 读取
        lines = File.readlines()
        line = ''.join(lines)
        # 去特殊符号
        punctuation = '!@#$%^&*(),，.?;:-”、。；《》'
        s = re.sub(r"[%s]+" % punctuation, "", line)
    return s


def get_vec(str1, str2):
    # 获取词向量
    str1_info = jieba.analyse.extract_tags(str1, withWeight=True)
    str2_info = jieba.analyse.extract_tags(str2, withWeight=True)
    # 为排除0的情况而转成counter
    str1_dict = Counter({i[0]: i[1] for i in str1_info})
    str2_dict = Counter({i[0]: i[1] for i in str2_info})
    #
    bags = set(str1_dict.keys()).union(set(str2_dict.keys()))
    # 进行从小到大的重新排序
    bags = sorted(list(bags))
    vec_str1 = [str1_dict[i] for i in bags]
    vec_str2 = [str2_dict[i] for i in bags]
    # 将结构数据转化为ndarray,因asarray不会占用新内存而不选择array
    vec_str1 = np.asarray(vec_str1, dtype=np.float)
    vec_str2 = np.asarray(vec_str2, dtype=np.float)
    return vec_str1, vec_str2


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    #
    vec1, vec2 = np.asarray(vec1, dtype=np.float), np.asarray(vec2, dtype=np.float)
    up = np.dot(vec1, vec2)
    down = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    # 四舍五入保留2位小数
    return round(up / down, 2)


def test(doc_name):
    doc_file = "D:/sim_0.8/" + doc_name
    str_test = string(doc_file)
    similarity = cosine_similarity(*get_vec(doc, str_test))
    print("%s 的相似度 = %.2f" % (doc_name, similarity))


doc = "D:/sim_0.8/orig.txt"
doc = string(doc)
test("orig_0.8_add.txt")
test("orig_0.8_del.txt")
test("orig_0.8_dis_1.txt")
test("orig_0.8_dis_3.txt")
test("orig_0.8_dis_7.txt")
test("orig_0.8_dis_10.txt")
test("orig_0.8_dis_15.txt")
test("orig_0.8_mix.txt")
test("orig_0.8_rep.txt")
test("test1.txt")
test("test2.txt")
test("test3.txt")
