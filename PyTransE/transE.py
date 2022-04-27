import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}   #实体、ID字典
relation2id = {}

# 文件操作函数
def data_loader(file):
    file1 = file + "train.txt" #头实体key + 尾实体key + 关系
    file2 = file + "entity2id.txt" #实体key + id编号
    file3 = file + "relation2id.txt" #关系key + id编号

    with open(file2, 'r') as f1, open(file3, 'r') as f2:  # 实体和关系
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            # strip移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。以'\t'（tab键）为界使用split对字符串进行切片，生成子字符串
            line = line.strip().split('\t')
            if len(line) != 2:  # 这里正确格式的数据，被分割为两个子字符串
                continue
            entity2id[line[0]] = line[1]  # 生成字典键值对，实体为key，ID为值

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    '''产生新的空集合,set是一个不允许内容重复的组合
    而且set里的内容位置是随意的 所以不能用索引列出。
    可进行关系测试，删除重复数据
    '''
    entity_set = set()
    relation_set = set()
    triple_list = []
    ''' 使用codecs提供的open方法来指定打开的文件的语言编码
        它会在读取的时候自动转换为内部unicode 不至于出现乱码
    '''
    with codecs.open(file1, 'r') as f:
        """
        这里使用了 with 语句，不管在处理文件过程中是否发生异常，
        都能保证 with 语句执行完毕后已经关闭了打开的文件句柄。
        如果使用传统的 try/finally 范式，则要使用类似如下代码：
        f = open(r'file1')
        try:
            for line in f:
                print line
                # ...more code
        finally:
            f.close()
        """
        content = f.readlines()
        for line in content:
            triple = line.strip().split('\t')
            if len(triple) != 3:
                continue

            # 存入将头尾实体ID和关系ID存入三元组
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]
            triple_list.append([h_, t_, r_])

            # 头尾实体和关系ID存入set集合，不会出现重复的实体或关系
            entity_set.add(h_)
            entity_set.add(t_)
            relation_set.add(r_)
    print("Complete load. entity : %d , relation : %d , triple : %d" % (len(entity_set),len(relation_set),len(triple_list)))
    # 返回头尾实体集合，并且返回三元组列表
    return entity_set, relation_set, triple_list

# L1、L2正则化
# 欧式距离
def distanceL2(h,r,t):
    #为方便求梯度，去掉sqrt开方，只保留square平方
    return np.sum(np.square(h + r - t))
    
# 曼哈顿距离
def distanceL1(h,r,t):  
    return np.sum(np.fabs(h + r - t))

# 算法的实现
class TransE:
    # 初始化，维度设为50，学习率为0.01，边界为1，L1正则化
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim, learning_rate, margin,L1):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1=L1
        self.loss = 0   #损失
    

    # 初始化为向量，最后将ID集合更新为ID向量字典
    def emb_initialize(self):
        # 设置实体和关系两个空字典，临时保存向量化的关系/实体，键为关系/实体ID，值为向量
        relation_dict = {}
        entity_dict = {}

        for relation in self.relation: # 关系ID集合，进行随机向量化
            r_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) ,
                                           6/math.sqrt(self.embedding_dim) ,
                                           self.embedding_dim) # 前两个参数是随机均匀分布的范围，后面是向量得维度
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp,ord=1) # 参数：矩阵、范数类型；FB15k使用L1归一化

        for entity in self.entity: # 实体ID集合，进行随机向量化
            e_emb_temp = np.random.uniform(-6/math.sqrt(self.embedding_dim) ,
                                        6/math.sqrt(self.embedding_dim) ,
                                        self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp,ord=1)

        # 归一化后保存更新向量
        self.relation = relation_dict
        self.entity = entity_dict

    # 训练
    def train(self, epochs): # epochs循环训练次数
        nbatches = 400 # 每次选取三元组的个数

        # 取整 batch_size * nbatches = len(self.triple_list)
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            for k in range(nbatches): # 训练最多为1000次
                # Sbatch:list 每次随机选取batch_size个样本
                Sbatch = random.sample(self.triple_list, batch_size)
                Tbatch = []

                for triple in Sbatch: 
                    # 每个triple选3个负样例
                    # for i in range(3):
                    corrupted_triple = self.Corrupt(triple) # 调用Corrupt函数，生成负样本
                    if (triple, corrupted_triple) not in Tbatch:# 正负样例不重复，则存入混合测试列表
                        Tbatch.append((triple, corrupted_triple))
                self.update_embeddings(Tbatch)# 立即更新样例

            # 输出循环次数和花费时间，并输出损失值
            end = time.time()
            print("epoch: ", epoch , "cost time: %s"%(round((end - start),3)))
            print("loss: ", self.loss)

            #保存临时结果，20次循环写入一次
            if epoch % 20 == 0:
                with codecs.open("entity_temp", "w") as f_e:
                    for e in self.entity.keys():
                        f_e.write(e + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with codecs.open("relation_temp", "w") as f_r:
                    for r in self.relation.keys():
                        f_r.write(r + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")

        print("写入文件...")  # 将更新后的测试集写入文件，ID号作为桥梁
        with codecs.open("entity_50dim_batch400", "w") as f1:  
            for e in self.entity.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with codecs.open("relation50dim_batch400", "w") as f2:
            for r in self.relation.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        print("写入完成")


    # 生成损坏三元组，替换头或尾实体的操作，返回替换完成的三元组
    def Corrupt(self,triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random() # 设置随机替换的条件,seed大于0.5替换头节点，小于替换尾节点
        
        if seed > 0.5:
            # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:
                # 随机抽样，在实体集中随机抽取一个三元组的头实体进行替换
                rand_head = random.sample(self.entity.keys(),1)[0] 
            corrupted_triple[0]=rand_head
        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail

        return corrupted_triple

    # 更新实体
    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            # 计算距离
            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            # 计算损失
            err = self.hinge_loss(dist_correct, dist_corrupt)

            # 根据损失选择更新，向量进行梯度计算的时候，将每个维度看作标量进行更新
            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + relation - t_correct) # 正确三元组差距向量
                grad_neg = 2 * (h_corrupt + relation - t_corrupt) # 错误三元组差距向量

                # 使用L1梯度，更新正确和错误三元组误差向量
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                # 正样本 head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                # corrupt项整体为负，因此符号与correct相反
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg
                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                #relation更新两次
                relation_update -= self.learning_rate*grad_pos
                relation_update -= (-1)*self.learning_rate*grad_neg
                


        #batch norm归一化操作
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    #  损失函数
    def hinge_loss(self,dist_correct,dist_corrupt):
        return max(0,dist_correct-dist_corrupt+self.margin)


if __name__=='__main__':
    file1 = "C:/Users/Y/Desktop/transe/PyTransE/FB15k/"
    entity_set, relation_set, triple_list = data_loader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (len(entity_set),len(relation_set),len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim = 50, learning_rate = 0.01, margin = 1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=400)
    print("train over ！！")
    
