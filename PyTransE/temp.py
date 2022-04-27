import codecs
import json
import operator

import numpy as np

# 导入train.py的文件处理函数、实体和关系的字典
from transE import data_loader, entity2id, relation2id


# 测试文件处理
def dataloader(entity_file, relation_file, test_file):
    # entity_file: entity \t embedding
    entity_dict = {}
    relation_dict = {}
    test_triple = []

    # 训练生成的实体文件处理
    with codecs.open(entity_file) as e_f:
        lines = e_f.readlines()
        for line in lines:
            # 实体编号、嵌入的向量
            entity, embedding = line.strip().split('\t')
            embedding = json.loads(embedding)  # 将字符串转换为python对象，这里是列表
            entity_dict[entity] = embedding

    # 训练生成的关系文件处理
    with codecs.open(relation_file) as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')
            embedding = json.loads(embedding)
            relation_dict[relation] = embedding

    # 测试文件处理
    with codecs.open(test_file) as t_f:
        lines = t_f.readlines()
        for line in lines:
            triple = line.strip().split('\t')
            if len(triple) != 3:  # 字符列表的长度不为3是错误数据
                continue
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]

            test_triple.append(tuple((h_, t_, r_)))  # 存储着ID

    return entity_dict, relation_dict, test_triple

# 距离计算，利用numpy.array进行计算向量


def distance(h, r, t):
    h = np.array(h)
    r = np.array(r)
    t = np.array(t)
    s = h+r-t
    return np.linalg.norm(s)

# 测试类设计

class TestRelation:
    def __init__(self, entity_dict, relation_dict, head_entity, tail_entity):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.head_entity = head_entity
        self.tail_entity = tail_entity
        self.test_triple = [self.entity_dict, "null", self.tail_entity]



class Test:
    # 测试集实体集合、测试集关系集合、测试集三元组列表、训练集生成三元组列表、是否过滤
    def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=False):
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.test_triple = test_triple
        self.train_triple = train_triple
        self.isFit = isFit

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    def relation_rank(self):
        hits = 0
        rank_sum = 0
        step = 1

        for triple in self.test_triple:
            if(step >= 1000):
                break
            rank_dict = {}
            for r in self.relation_dict.keys():
                corrupted_relation = (triple[0], triple[1], r)
                if self.isFit and corrupted_relation in self.train_triple:  # 是否过滤
                    continue
                h_emb = self.entity_dict[corrupted_relation[0]]
                r_emb = self.relation_dict[corrupted_relation[2]]
                t_emb = self.entity_dict[corrupted_relation[1]]
                rank_dict[r] = distance(h_emb, r_emb, t_emb)

            rank_sorted = sorted(rank_dict.items(), key=operator.itemgetter(1))

            rank = 1
            for i in rank_sorted:
                if triple[2] == i[0]:
                    print("正确关系: ",rank_dict[i[0]])
                    break
                rank += 1
                print(step," ", rank," 排名靠前的预备关系: ",rank_dict[i[0]])
            
            print("*##############*\n")
            if rank < 10:
                hits += 1

            with codecs.open("TEST", "w+") as file:
                if(rank > 10):
                    file.write("超出前十")
                else:
                    for e in range(rank+1):
                        file.write(rank_dict[i[0]])

            rank_sum = rank_sum + rank + 1

            step += 1
            if step % 1000 == 0:
                print("relation step ", step, " ,hits ",
                      hits, " ,rank_sum ", rank_sum)
                print()

        self.relation_hits10 = hits / len(self.test_triple)
        self.relation_mean_rank = rank_sum / len(self.test_triple)


if __name__ == '__main__':

    # file = "/Users/liuyadong/Downloads/PythonLearning/pythoncode/"
    # entity_dict = file + "entity_50dim_batch400"
    # relation_dict = file + "relation50dim_batch400"
    # test_triple = file + "ransE-master/FB15k/test.txt"

    _, _, train_triple = data_loader(
        "/Users/liuyadong/Downloads/ResearchProject/GraphMachineLearning/codes/TransE/PyTransE/transE-finished/FB15k/")

    entity_dict, relation_dict, test_triple = \
        dataloader("/Users/liuyadong/Downloads/ResearchProject/GraphMachineLearning/codes/TransE/PyTransE/transE-finished/FB15k/entity_50dim_batch400",
                   "/Users/liuyadong/Downloads/ResearchProject/GraphMachineLearning/codes/TransE/PyTransE/transE-finished/FB15k/relation50dim_batch400",
                   "/Users/liuyadong/Downloads/ResearchProject/GraphMachineLearning/codes/TransE/PyTransE/transE-finished/FB15k/test.txt")

    test = Test(entity_dict, relation_dict,
                test_triple, train_triple, isFit=False)
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


# step  5000, hits  2936, rank_sum  2688773
# step  10000, hits  5891, rank_sum  5340548
# step  15000, hits  8852, rank_sum  8194415
# step  20000, hits  11736, rank_sum  10811112
# step  25000, hits  14791, rank_sum  13410539
# step  30000, hits  17780, rank_sum  16019036
# step  35000, hits  20665, rank_sum  18708139
# step  40000, hits  23566, rank_sum  21546574
# step  45000, hits  26609, rank_sum  24138231
# step  50000, hits  29576, rank_sum  26724634
# step  55000, hits  32554, rank_sum  29467612

# entity hits@10:  0.2962198032875692
# entity meanrank:  266.4581351255269

# relation step  5000, hits  4107, rank_sum  427954
# relation step  10000, hits  8201, rank_sum  804424
# relation step  15000, hits  12291, rank_sum  1202761
# relation step  20000, hits  16353, rank_sum  1627425
# relation step  25000, hits  20430, rank_sum  2042220
# relation step  30000, hits  24501, rank_sum  2447066
# relation step  35000, hits  28541, rank_sum  2858711
# relation step  40000, hits  32666, rank_sum  3249984
# relation step  45000, hits  36762, rank_sum  3633002
# relation step  50000, hits  40887, rank_sum  4015066
# relation step  55000, hits  44989, rank_sum  4405508

# relation hits@10:  0.8178463205295322
# relation meanrank:  80.17238577305277


# relation step  5000, hits  4107, rank_sum  427603
# relation step  10000, hits  8201, rank_sum  805194
# relation step  15000, hits  12292, rank_sum  1203625
# relation step  20000, hits  16354, rank_sum  1629134
# relation step  25000, hits  20431, rank_sum  2044065
# relation step  30000, hits  24502, rank_sum  2450558
# relation step  35000, hits  28542, rank_sum  2862752
# relation step  40000, hits  32667, rank_sum  3255339
# relation step  45000, hits  36762, rank_sum  3638338
# relation step  50000, hits  40887, rank_sum  4019642
# relation step  55000, hits  44989, rank_sum  4410877

# relation hits@10:  0.8178463205295322
# relation meanrank:  80.2740769582367
'''
/m/027rn	9447
/m/06cx9	5030
/location/country/form_of_government	352
'''
