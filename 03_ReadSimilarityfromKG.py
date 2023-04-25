"""
    @Project: rock-image-few-shot-learning
    @File   : ReadSimilarityfromKG.py
    @Author : Chen Zhongliang
    @E-mail : c_mulder@163.com
    @info   : Read the rock type similarity knowledge from the LithoKG and save it as a npy format file.
"""


import os
from py2neo import Graph, RelationshipMatcher, NodeMatcher
import py2neo as py2
import numpy as np

datasets_root = 'E:\\datasets'
data_folder = 'rocks3.0.6_tvt'

print("py2neo:{}".format(py2.__version__))


class RocksGraph:
    def __init__(self):
        self.g = None
        self.id2label = {}
        self.label2id = {}

        try:
            self.g = Graph('http://localhost:11004/', auth=("neo4j", "XXXXXX"))
            print("Connect Neo4j server success！")
        except BaseException:
            print("Failed to connect Neo4j server！")

    def read_similarity_to_file(self, file_path=''):
        num_class = len(self.id2label)
        score = np.zeros((num_class, num_class))

        node_matcher = NodeMatcher(self.g)
        relation_matcher = RelationshipMatcher(self.g)

        for i in range(num_class):
            for j in range(num_class):
                if i == j:
                    score[i][j] = 1.0
                else:
                    subject_name = self.id2label[i]
                    object_name = self.id2label[j]
                    node_subject = node_matcher.match("Rock").where(name=subject_name).first()
                    node_object = node_matcher.match("Rock").where(name=object_name).first()

                    relation_score = list(relation_matcher.match((node_subject, node_object), r_type="SIMILAR"))
                    if len(relation_score) > 0:
                        score[i][j] = relation_score[0]["score"]
        file_path_txt = "%s.txt" % file_path
        np.savetxt(file_path_txt, score, fmt='%0.4f', delimiter=',')
        file_path_npy = "%s.npy" % file_path
        np.save(file_path_npy, score)

    def init_labelid(self):
        dataset_home = os.path.join(datasets_root, data_folder)

        ilabels = os.listdir(os.path.join(dataset_home, 'train'))
        label2id = dict(zip(ilabels, range(len(ilabels))))
        print(label2id)
        self.label2id = label2id
        id2label = dict(zip(label2id.values(), label2id.keys()))
        self.id2label = id2label


if __name__ == '__main__':
    handler = RocksGraph()
    handler.init_labelid()
    str_file_name = "similarity"
    handler.read_similarity_to_file(str_file_name)
