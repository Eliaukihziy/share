import codecs
import json
import operator
import re

import numpy as np
import torch
import matplotlib

# 导入train.py的文件处理函数、实体和关系的字典
from transE import data_loader, entity2id, relation2id


# 测试文件处理
def dataloader(entity_file,relation_file,test_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    # 训练生成的实体文件处理
    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            # 实体编号、嵌入的向量
            entity,embedding = line.strip().split('\t')
            embedding = json.loads(embedding) # 将字符串转换为python对象，这里是列表
            entity_dict[entity] = embedding

    # 训练生成的关系文件处理
    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation,embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding

    # 测试文件处理
    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')
            if len(triple) != 3: # 字符列表的长度不为3是错误数据
                continue
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            test_triple.append(tuple((h_,t_,r_))) # 存储着ID

    return entity_dict,relation_dict,test_triple

# 距离计算，利用numpy.array进行计算向量
def distance(h,r,t):
    # 向量构建
    h = np.array(h)
    r=np.array(r)
    t = np.array(t)
    s=h+r-t
    return np.linalg.norm(s)

def consistence(triplelist, dimsize):
    res = 0
    for i in range(1,len(triplelist)):
        res += torch.nn.softmax(triplelist, dimsize) * (1 - np.sign(abs(triplelist[0] - triplelist[i])))
    return res / dimsize

# 测试类设计
class Test:

    # 测试集实体集合、测试集关系集合、测试集三元组列表、训练集生成三元组列表、是否过滤
    def __init__(self,entity_dict,relation_dict,test_triple,train_triple,isFit = False):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit

    # 排名计算
    
    def entity_rank(self):
        k = 10

        for triple in self.test_triple:
            rank_dict = {}

            #维度大小
            dim_size = len(self.entity_dict[0])
            for w in self.entity_dict.keys():
                corrupted_relation = (triple[0],w,triple[2])
                if self.isFit and corrupted_relation in self.train_triple: # 是否过滤
                    continue
                h_emb = self.entity_dict[corrupted_relation[0]]
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                rank_dict[w]=distance(h_emb, r_emb, t_emb)# 向量

            rank_sorted = sorted(rank_dict.items(),key = operator.itemgetter(1))

            k_sorted = {}
            #将前k个排序结果放到k_sorted里
            for i in range(k):
                k_sorted.update(rank_sorted[i].key(), rank_sorted[i].value())
                

            #每一个关系对应的一致性值
            entity_csis = {}
            #i代表k_sorted中的第i个
            #j代表第j个维度
            for i in range(k):
                #每一行的每一个维度更改后的排序
                row_sorted = [i]
                for j in range(dim_size):
                    entity_id = k_sorted[i].key()
                    #保存未修改的第j个维度值,注意copy
                    save_dim = entity_dict[entity_id][j]
                    save_sorted = k_sorted.copy()
                    entity_dict[entity_id][j] = 0
                    h_emb = self.entity_dict[corrupted_relation[0]]
                    r_emb = self.relation_dict[corrupted_relation[2]]
                    t_emb = self.entity_dict[entity_id]
                    k_sorted[i].value=distance(h_emb, r_emb, t_emb)# 向量

                    #待排序的k_sorted
                    cur_sort = k_sorted.copy()
                    cur_sorted = sorted(cur_sort.items(),key = operator.itemgetter(1))

                    #将该次修改，该行所在的排名添加
                    for k in range(k):
                        if entity_id == cur_sorted[k].key():
                            row_sorted.append(k)
                            break
                    #重置
                    k_sorted = save_sorted
                    relation_dict[entity_id][j] = save_dim

                #计算最终得分
                consis = consistence(row_sorted,dim_size)
                entity_csis.update(entity_id, consis)

            #输出最后的打分结果
            for i in entity_csis:
                print(i.key, '\t', i.value)

if __name__ == '__main__':

    _, _, train_triple = data_loader(
        "C:/Users/Y/Desktop/transe/PyTransE/FB15k/")

    entity_dict, relation_dict, test_triple = \
        dataloader("C:/Users/Y/Desktop/transe/PyTransE/FB15k/entity_50dim_batch400",\
         "C:/Users/Y/Desktop/transe/PyTransE/FB15k/relation50dim_batch400", \
                   "C:/Users/Y/Desktop/transe/PyTransE/FB15k/test.txt")


    test = Test(entity_dict,relation_dict,test_triple,train_triple,isFit=False)
    # test.rank()         
    # print("entity hits@10: ", test.hits10)
    # print("entity meanrank: ", test.mean_rank)

    test.entity_rank()
    #print("relation hits@10: ", test.relation_hits10)
    #print("relation meanrank: ", test.relation_mean_rank)

    # f = open("result.txt",'w')
    # f.write("entity hits@10: "+ str(test.hits10) + '\n')
    # f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    # f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    # f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    # f.close()
