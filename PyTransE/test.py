import codecs
import json
import operator

import numpy as np

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

# 测试类设计
class Test:

    # 测试集实体集合、测试集关系集合、测试集三元组列表、训练集生成三元组列表、是否过滤
    def __init__(self,entity_dict,relation_dict,test_triple,train_triple,isFit = False):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    # 排名计算
    '''
    def rank(self):
        hits = 0
        rank_sum = 0
        step = 1
        
        for triple in self.test_triple:
            rank_head_dict = {}
            rank_tail_dict = {}

            for entity in self.entity_dict.keys():
                corrupted_head = [entity,triple[1],triple[2]] # 生成头实体的负样本
                if self.isFit: # 过滤操作                  
                    if corrupted_head not in self.train_triple: # 想法：使用iteration函数会不会加速遍历操作
                        h_emb = self.entity_dict[corrupted_head[0]]
                        r_emb = self.relation_dict[corrupted_head[2]]
                        t_emb = self.entity_dict[corrupted_head[1]]
                        rank_head_dict[tuple(corrupted_head)]=distance(h_emb,r_emb,t_emb) # 计算负样本三元组之间到距离
                else:
                    h_emb = self.entity_dict[corrupted_head[0]]
                    r_emb = self.relation_dict[corrupted_head[2]]
                    t_emb = self.entity_dict[corrupted_head[1]]
                    rank_head_dict[tuple(corrupted_head)] = distance(h_emb, r_emb, t_emb)

                corrupted_tail = [triple[0],entity,triple[2]] # 生成尾实体的负样本
                if self.isFit:
                    if corrupted_tail not in self.train_triple:
                        h_emb = self.entity_dict[corrupted_tail[0]]
                        r_emb = self.relation_dict[corrupted_tail[2]]
                        t_emb = self.entity_dict[corrupted_tail[1]]
                        rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)
                else:
                    h_emb = self.entity_dict[corrupted_tail[0]]
                    r_emb = self.relation_dict[corrupted_tail[2]]
                    t_emb = self.entity_dict[corrupted_tail[1]]
                    rank_tail_dict[tuple(corrupted_tail)] = distance(h_emb, r_emb, t_emb)

            # 将负样本向量的距离按从小到大排序进行排名
            rank_head_sorted = sorted(rank_head_dict.items(),key = operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(),key = operator.itemgetter(1))

            #rank_sum and hits
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i<10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i<10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            step += 1
            if step % 5000 == 0: # 每5000次公布一次结果
                print("step ", step, " ,hits ",hits," ,rank_sum ",rank_sum)
                print()

        # 头尾实体都会查找一遍所以除于测试三元组个数的两倍
        self.hits10 = hits / (2*len(self.test_triple))
        self.mean_rank = rank_sum / (2*len(self.test_triple))
    '''
    def relation_rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            rank_dict = {}
            for r in self.relation_dict.keys():
                corrupted_relation = (triple[0],triple[1],r)
                if self.isFit and corrupted_relation in self.train_triple: # 是否过滤
                    continue
                h_emb = self.entity_dict[corrupted_relation[0]]
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                rank_dict[r]=distance(h_emb, r_emb, t_emb)# 向量

            rank_sorted = sorted(rank_dict.items(),key = operator.itemgetter(1))

            

            rank = 1
            flag = 0
            for i in rank_sorted:
                if(rank<10):
                    print("pridict result：",list(entity2id.keys())[list(entity2id.values()).index(triple[0])], 
                                    " ", list(entity2id.keys())[list(entity2id.values()).index(triple[1])], 
                                    " ", list(relation2id.keys())[list(relation2id.values()).index(i[0])])
                if triple[2] == i[0]:
                    print("*** predict right ***")
                    flag = 1
                    break
                rank += 1
            if flag == 0:
                print("*** predict falut ***")
            if rank<10:
                hits += 1
            rank_sum = rank_sum + rank + 1

            step += 1
            if step % 5000 == 0:
                print("relation step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
                print()

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)

if __name__ == '__main__':

    # file = "/Users/liuyadong/Downloads/PythonLearning/pythoncode/"
    # entity_dict = file + "entity_50dim_batch400"
    # relation_dict = file + "relation50dim_batch400"
    # test_triple = file + "ransE-master/FB15k/test.txt"

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

    test.relation_rank()
    print("relation hits@10: ", test.relation_hits10)
    print("relation meanrank: ", test.relation_mean_rank)

    # f = open("result.txt",'w')
    # f.write("entity hits@10: "+ str(test.hits10) + '\n')
    # f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    # f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    # f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    # f.close()
