"""
Created on 2017-10-26
class: search_tree, node
@author: fengyue
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

from treelib import Tree
import copy


class node(object):
    def __init__(self):
        self.num = 0.0
        self.Q = 0.0
        self.p = 0.0
        self.doc = []
        
        
class search_tree(object):
    
    def __init__(self, query_id, depth, carpe_diem):
        self.tree = Tree()
        self.tree.create_node(identifier='query_' + query_id, data=node())
        root_node = self.tree.get_node('query_' + query_id)
        root_node.data.num = 1.0
        self.node_map = {}
        self.count = 0.0
        self.carpe_diem = carpe_diem
        self.max_depth = depth
        self.expand(self.tree.get_node(self.tree.root))
        
    def expand(self, leaf_node):
        doc_list = leaf_node.data.doc
        p_doc_id, p_pred = self.carpe_diem.policy(self.tree.root, doc_list)
        for doc in p_doc_id:
            self.node_map[' '.join(doc_list+[doc])] = len(self.node_map)
            new_node = node()
            new_node.doc = doc_list + [doc]
            new_node.p = p_pred[p_doc_id.index(doc)]
            self.tree.create_node(identifier=self.node_map[' '.join(new_node.doc)], data=new_node, parent=leaf_node.identifier)
        
    def update(self, node_list, value):
        for node_id in node_list:
            tmp_node = self.tree.get_node(node_id)
            tmp_node.data.Q = (tmp_node.data.Q * tmp_node.data.num + value) / (tmp_node.data.num + 1)
            tmp_node.data.num += 1
        
    def search(self, start_node_id):
        tmp_node = self.tree.get_node(start_node_id)
        has_visit_num = tmp_node.data.num - 1
        self.count = has_visit_num

        if int(self.carpe_diem.search_time-has_visit_num) > 0:
            start_node_search_time = int(self.carpe_diem.search_time-has_visit_num)
        else:
            start_node_search_time = 0
        
        for time in range(start_node_search_time):
            search_list = [start_node_id]
            tmp_node = self.tree.get_node(start_node_id)
            while not tmp_node.is_leaf():
                max_score = float("-inf")
                max_id = -1
                for child_id in tmp_node.fpointer:
                    child_node = self.tree.get_node(child_id)
                    #score = child_node.data.p
                    score = self.carpe_diem.beta * child_node.data.p * ((tmp_node.data.num-1)**0.5 / (1+child_node.data.num))
                    '''
                    print score
                    print child_node.data.Q
                    print '**************'
                    '''
        
                    score += child_node.data.Q
                    if score > max_score:
                        max_id = child_id
                        max_score = score
                search_list.append(max_id)
                tmp_node = self.tree.get_node(max_id)
            
            query_id_mcts = self.tree.root.split('_')[1]
            if self.tree.depth(tmp_node) == self.max_depth:
                listPermutation = copy.deepcopy(self.carpe_diem.dictQueryPermutaion[query_id_mcts]['permutation'])
                idealScore = self.carpe_diem.alphaDCG(0.5, query_id_mcts, listPermutation, self.max_depth)
                v = self.carpe_diem.alphaDCG(0.5, query_id_mcts, tmp_node.data.doc, self.max_depth)
                v = v / idealScore
            else:
                v = self.carpe_diem.value_function(self.tree.root, tmp_node.data.doc)
            
            self.update(search_list, v)
            self.count += 1
            
            if tmp_node.is_leaf() and (self.tree.depth(tmp_node) < self.max_depth):
                self.expand(tmp_node)

            ###########
            if time % 100 == 0:
                tmp_policy = self.get_policy(start_node_id)
                print tmp_policy.values()
                print sum(tmp_policy.values())
                print time
        
    def take_action(self, start_node_id):
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            prob[child_node.data.doc[-1]] = child_node.data.num / self.count
            if child_node.data.num > max_time:
                max_time = child_node.data.num
                select_doc = child_node.data.doc[-1]
                select_doc_node_id = child_node.identifier
        return prob, select_doc, select_doc_node_id

    def get_policy(self, start_node_id):
        tmp_node = self.tree.get_node(start_node_id)
        max_time = -1
        prob = {}
        for child_id in tmp_node.fpointer:
            child_node = self.tree.get_node(child_id)
            if self.count == 0:
                prob[child_node.data.doc[-1]] = 0.0
            else:
                prob[child_node.data.doc[-1]] = child_node.data.num / self.count
        return prob


