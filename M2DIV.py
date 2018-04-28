"""
Created on 2017-10-26
class: M2DIV
@author: fengyue
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import json
import yaml
import copy
import math
import random
import numpy as np
import tensorflow as tf
import datetime
import subprocess
from MCTStree import search_tree

# tf Graph input
input_query = tf.placeholder(tf.float32, [1, 100])
query_selected = tf.placeholder(tf.float32, [None, 100])
candidate = tf.placeholder(tf.float32, [None, 100])
p = tf.placeholder(tf.float32, [1, None])
v = tf.placeholder(tf.float32, [1, 1])


class M2DIV(object):
    """docstring for M2DIV"""
    def __init__(self, fileQueryPermutaion, fileQueryRepresentation, fileDocumentRepresentation, fileQueryDocumentSubtopics, folder):
        super(M2DIV, self).__init__()

        with open(fileQueryPermutaion) as self.fileQueryPermutaion:
            self.dictQueryPermutaion = json.load(self.fileQueryPermutaion)

        with open(fileQueryRepresentation) as self.fileQueryRepresentation:
            self.dictQueryRepresentation = json.load(self.fileQueryRepresentation)
        for query in self.dictQueryRepresentation:
            self.dictQueryRepresentation[query] = np.matrix([self.dictQueryRepresentation[query]], dtype=np.float)
            self.dictQueryRepresentation[query] = np.transpose(self.dictQueryRepresentation[query])

        with open(fileDocumentRepresentation) as self.fileDocumentRepresentation:
            self.dictDocumentRepresentation = json.load(self.fileDocumentRepresentation)
        for doc in self.dictDocumentRepresentation:
            self.dictDocumentRepresentation[doc] = np.matrix([self.dictDocumentRepresentation[doc]], dtype=np.float)
            self.dictDocumentRepresentation[doc] = np.transpose(self.dictDocumentRepresentation[doc])

        with open(fileQueryDocumentSubtopics) as self.fileQueryDocumentSubtopics:
            self.dictQueryDocumentSubtopics = json.load(self.fileQueryDocumentSubtopics)

        self.query_subtopics = {}
        for query_id, v in self.dictQueryDocumentSubtopics.items():
            subtopics_list = []
            for doc_id, sub in v.items():
                subtopics_list.extend(sub)
            subtopics_set = set(subtopics_list)
            self.query_subtopics[query_id] = len(subtopics_set)

        self.folder = folder
        with open(self.folder + '/config.yml') as self.confFile:
            self.dictConf = yaml.load(self.confFile)
        self.learning_rate = self.dictConf['learning_rate']
        self.listTestSet = self.dictConf['test_set']
        self.lenTrainPermutation = self.dictConf['length_train_permutation']
        self.step = self.dictConf['step']
        self.hidden_dim = self.dictConf['hidden_dim']
        self.search_time = 5000
        self.epoch = 50000
        self.beta = float(sys.argv[3])
        
        self.fileResult = open(self.folder + '/' + sys.argv[1], 'w')
        self.fileReward = open(self.folder + '/' + sys.argv[2], 'w')


    def alphaDCG(self, alpha, query, docList, k):
        DCG = 0.0
        subtopics = []
        for i in xrange(20):
            subtopics.append(0)
        for i in xrange(k):
            G = 0.0
            if docList[i] not in self.dictQueryDocumentSubtopics[query]:
                continue
            listDocSubtopics = self.dictQueryDocumentSubtopics[query][docList[i]]
            if len(listDocSubtopics) == 0:
                    G = 0.0
            else:
                for subtopic in listDocSubtopics:
                    G += (1-alpha) ** subtopics[int(subtopic)-1]
                    subtopics[int(subtopic)-1] += 1
            DCG += G/math.log(i+2, 2)
        return DCG

    def subtopic_recall(self, query, docList, k):
        n = self.query_subtopics[query]
        subtopics_r = []
        for d in docList[:k]:
            if self.dictQueryDocumentSubtopics[query].has_key(d):
                subtopics_r.extend(self.dictQueryDocumentSubtopics[query][d])
        return len(set(subtopics_r))*1.0 / n

    def expected_reciprocal_rank(self, query, docList, k):
        n = self.query_subtopics[query]
        all_doc = len(self.dictQueryPermutaion[query]['permutation'])
        p_topic = [0.0] * n
        topic_map = {}
        for d in self.dictQueryPermutaion[query]['permutation']:
            if self.dictQueryDocumentSubtopics[query].has_key(d):
                for doc_topic in self.dictQueryDocumentSubtopics[query][d]:
                    if topic_map.has_key(doc_topic):
                        p_topic[topic_map[doc_topic]] += 1
                    else:
                        topic_map[doc_topic] = len(topic_map)
                        p_topic[topic_map[doc_topic]] += 1
        err = 0.0
        for id_n, d in enumerate(docList[:k]):
            all_topic = 0.0
            for topic_name, id_t in topic_map.items():
                score = 1.0
                for selected_doc in docList[:id_n]:
                    r = 0.0
                    if self.dictQueryDocumentSubtopics[query].has_key(selected_doc):
                        for doc_t in self.dictQueryDocumentSubtopics[query][selected_doc]:
                            if doc_t == topic_name:
                                r = (2.0**1 - 1) / 2**1
                    score *= (1-r)
                r = 0.0
                if self.dictQueryDocumentSubtopics[query].has_key(docList[id_n]):
                    for doc_t in self.dictQueryDocumentSubtopics[query][docList[id_n]]:
                        if doc_t == topic_name:
                            r = (2.0**1 - 1) / 2**1
                score *= r
                all_topic += p_topic[id_t] / all_doc * score
            err += 1.0 / (id_n+1) * all_topic

        return err

    def value_function(self, query, doc_list):
        query_id = query.split('_')[1]
        query_repr = carpe_diem.dictQueryRepresentation[str(query_id)]
        query_repr = np.reshape(np.asarray(query_repr), -1).tolist()
        listSelecteddoc_repr = []
        for doc_id in doc_list:
            doc_repr = carpe_diem.dictDocumentRepresentation[doc_id]
            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
            listSelecteddoc_repr.append(doc_repr)
        value_p = sess.run(value_pred, feed_dict={input_query: [query_repr], query_selected: listSelecteddoc_repr})
        
        return value_p

    def policy(self, query, doc_list):
        query_id = query.split('_')[1]
        query_repr = carpe_diem.dictQueryRepresentation[str(query_id)]
        query_repr = np.reshape(np.asarray(query_repr), -1).tolist()
        listSelecteddoc_repr = []
        for doc_id in doc_list:
            doc_repr = carpe_diem.dictDocumentRepresentation[doc_id]
            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
            listSelecteddoc_repr.append(doc_repr)
        
        policy_listTest = copy.deepcopy(carpe_diem.dictQueryPermutaion[str(query_id)]['permutation'])
        policy_c = []
        policy_c_id = []
        for can in policy_listTest:
            if can not in doc_list:
                doc_repr = carpe_diem.dictDocumentRepresentation[can]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                policy_c.append(doc_repr)
                policy_c_id.append(can)

        if len(listSelecteddoc_repr) == 0:
        	c_pred = sess.run(doc_pred_first, feed_dict={input_query: [query_repr], candidate: policy_c})
        else:
        	c_pred = sess.run(doc_pred, feed_dict={input_query: [query_repr], query_selected: listSelecteddoc_repr, candidate: policy_c})
        
        return policy_c_id, c_pred


def build_model(carpe_diem):
    V = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim*2], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    W = tf.Variable(tf.random_uniform([carpe_diem.hidden_dim*2, 1], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    
    W_b = tf.Variable(tf.random_uniform([1, 1], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    
    V_c = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim)) 
    V_h = tf.Variable(tf.random_uniform([100, carpe_diem.hidden_dim], -1./carpe_diem.hidden_dim, 1./carpe_diem.hidden_dim))
    
    q_state_c = tf.sigmoid(tf.matmul(input_query, V_c))
    q_state_h = tf.sigmoid(tf.matmul(input_query, V_h))
    q_state = tf.concat([q_state_c, q_state_h], 1)
    
    # select first doc
    logits_first = tf.reshape(tf.matmul(tf.matmul(candidate, V), tf.transpose(q_state)), [-1])
    prob_first = tf.nn.softmax(logits_first)
    prob_id_first = tf.argmax(prob_first)
    value_first = tf.sigmoid(tf.reshape(tf.matmul(q_state, W), [1, 1]) + W_b)  # [1,1]
    loss_first = tf.contrib.losses.mean_squared_error(v, value_first) - tf.matmul(p, tf.reshape(tf.log(tf.clip_by_value(prob_first, 1e-30, 1.0)), [-1, 1]))
    optimizer_first = tf.train.AdagradOptimizer(carpe_diem.learning_rate).minimize(loss_first)

    input = tf.reshape(query_selected, [1, -1, 100])
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=carpe_diem.hidden_dim, state_is_tuple=False)
    _, states = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=q_state, dtype=tf.float32)  # [1, dim]
    logits = tf.reshape(tf.matmul(tf.matmul(candidate, V), tf.transpose(states)), [-1])
    prob = tf.nn.softmax(logits)
    prob_id = tf.argmax(prob)
    value = tf.sigmoid(tf.reshape(tf.matmul(states, W), [1, 1]) + W_b)  # [1,1]

    loss = tf.contrib.losses.mean_squared_error(v, value) - tf.matmul(p, tf.reshape(tf.log(tf.clip_by_value(prob, 1e-30, 1.0)), [-1, 1]))
    optimizer = tf.train.AdagradOptimizer(carpe_diem.learning_rate).minimize(loss)
    
    return optimizer, prob, prob_id, value, prob_id_first, optimizer_first, value_first, prob_first


query_permutation_file = './data/query_permutation.json'
query_representation_file = './data/query_representation.dat'
document_representation_file = './data/doc_representation.dat'
query_document_subtopics_file = './data/query_doc.json'
folder = 'data/' + sys.argv[4]


carpe_diem = M2DIV(query_permutation_file, query_representation_file, document_representation_file, query_document_subtopics_file, folder)
opt, doc_pred, doc_pred_id, value_pred, doc_pred_id_first, opt_first, value_pred_first, doc_pred_first = build_model(carpe_diem)

saver = tf.train.Saver(max_to_keep=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ckpt = tf.train.get_checkpoint_state(folder + '/model_final_' + sys.argv[3] + '/')
if ckpt and ckpt.model_checkpoint_path:
    print 'Load model from:', ckpt.model_checkpoint_path
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

listKeys = carpe_diem.dictQueryPermutaion.keys()
iteration = 0

for e in range(carpe_diem.epoch):
    for query_id in listKeys:
        print datetime.datetime.now()
        if int(query_id) in carpe_diem.listTestSet:
            continue
        
        q = carpe_diem.dictQueryRepresentation[query_id]
        q = np.reshape(np.asarray(q), -1).tolist()

        listPermutation = copy.deepcopy(carpe_diem.dictQueryPermutaion[query_id]['permutation'])
        idealScore_without_mcts = carpe_diem.alphaDCG(0.5, query_id, listPermutation, carpe_diem.lenTrainPermutation)
        if idealScore_without_mcts == 0:
            continue
        
        listSelectedSet = []
        p_data = []
        mcts_tree = search_tree(query_id, carpe_diem.lenTrainPermutation, carpe_diem)
        start_node = 'query_' + query_id
        
        for t in xrange(carpe_diem.lenTrainPermutation):
            print '------------------'
            print t
            print len(listPermutation)
            mcts_tree.search(start_node)
            tmp_policy = mcts_tree.get_policy(start_node)
            print tmp_policy.values()
            print sum(tmp_policy.values())
            prob, select_doc_id, start_node = mcts_tree.take_action(start_node)
            p_data.append(prob)
            listSelectedSet.append(select_doc_id)

        value_with_mcts = carpe_diem.alphaDCG(0.5, query_id, listSelectedSet, carpe_diem.lenTrainPermutation)

        # sample without MCTS
        listSelectedSet_without_mcts = []
        listSelectedSet_repr_without_mcts = []
        listPermutation_without_mcts = copy.deepcopy(carpe_diem.dictQueryPermutaion[query_id]['permutation'])
        random.shuffle(listPermutation_without_mcts)

        c = []
        c_id = []
        for can in listPermutation_without_mcts:
            if can not in listSelectedSet_without_mcts:
                doc_repr = carpe_diem.dictDocumentRepresentation[can]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                c.append(doc_repr)
                c_id.append(can)
        pred_first = sess.run(doc_pred_id_first, feed_dict={input_query: [q], candidate: c})

        listSelectedSet_without_mcts.append(c_id[pred_first])
        doc_repr = carpe_diem.dictDocumentRepresentation[c_id[pred_first]]
        doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
        listSelectedSet_repr_without_mcts.append(doc_repr)

        while len(listSelectedSet_without_mcts) < carpe_diem.lenTrainPermutation:
            c = []
            c_id = []
            for can in listPermutation_without_mcts:
                if can not in listSelectedSet_without_mcts:
                    doc_repr = carpe_diem.dictDocumentRepresentation[can]
                    doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                    c.append(doc_repr)
                    c_id.append(can)
            pred = sess.run(doc_pred_id, feed_dict={input_query: [q], query_selected: listSelectedSet_repr_without_mcts, candidate: c})

            listSelectedSet_without_mcts.append(c_id[pred])
            doc_repr = carpe_diem.dictDocumentRepresentation[c_id[pred]]
            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
            listSelectedSet_repr_without_mcts.append(doc_repr)
        value_without_mcts = carpe_diem.alphaDCG(0.5, query_id, listSelectedSet_without_mcts, carpe_diem.lenTrainPermutation)

        value_with_mcts = value_with_mcts / idealScore_without_mcts
        value_without_mcts = value_without_mcts / idealScore_without_mcts
        
        carpe_diem.fileReward.write(str(e) + ' ' + str(iteration) + ' ' + query_id + ' ' + str(value_with_mcts) + ' ' + str(value_without_mcts) + '\n')
        carpe_diem.fileReward.flush()

        s = []
        for doc in listSelectedSet:
            doc_repr = carpe_diem.dictDocumentRepresentation[doc]
            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
            s.append(doc_repr)

        for prob_id, prob_data in enumerate(p_data):
            c = []
            policy = []
            for prob_key, prob_value in prob_data.items():
                doc_repr = carpe_diem.dictDocumentRepresentation[prob_key]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                c.append(doc_repr)
                policy.append(prob_value)
            if prob_id == 0:
            	sess.run(opt_first, feed_dict={input_query: [q], candidate: c, p: [policy], v: [[value_with_mcts]]})
            else:
            	sess.run(opt, feed_dict={input_query: [q], query_selected: s[:prob_id], candidate: c, p: [policy], v: [[value_with_mcts]]})
            
        print datetime.datetime.now()

        ## test
        if iteration % 50 == 0:
            print 'test'
            floatSumResultScore_ndcg_5 = 0.0
            floatSumResultScore_ndcg_10 = 0.0
            floatSumResultScore_srecall_5 = 0.0
            floatSumResultScore_srecall_10 = 0.0
            floatSumResultScore_err_5 = 0.0
            floatSumResultScore_err_10 = 0.0

            floatSumResultScore_ndcg_5_q = 0.0
            floatSumResultScore_ndcg_10_q = 0.0
            floatSumResultScore_srecall_5_q = 0.0
            floatSumResultScore_srecall_10_q = 0.0
            floatSumResultScore_err_5_q = 0.0
            floatSumResultScore_err_10_q = 0.0

            dictResult = {}

            fileTmpResult_policy = open(carpe_diem.folder + '/tmp_result_policy_' + sys.argv[3] + '.txt', 'w')
            fileTmpResult_value = open(carpe_diem.folder + '/tmp_result_value_' + sys.argv[3] + '.txt', 'w')

            for query_test in carpe_diem.listTestSet:
                listSelectedSet = []
                listSelectedSet_repr = []
                listSelectedSet_q = []
                listSelectedSet_repr_q = []
                listTest = copy.deepcopy(carpe_diem.dictQueryPermutaion[str(query_test)]['permutation'])
                idealScore_ndcg_10 = carpe_diem.alphaDCG(0.5, str(query_test), listTest, 10)
                idealScore_ndcg_5 = carpe_diem.alphaDCG(0.5, str(query_test), listTest, 5)
                if idealScore_ndcg_5 == 0 or idealScore_ndcg_10 == 0:
                    continue
                random.shuffle(listTest)
                q_test = carpe_diem.dictQueryRepresentation[str(query_test)]
                q_test = np.reshape(np.asarray(q_test), -1).tolist()
                
                # policy
                c = []
                c_id = []
                for can in listTest:
                    if can not in listSelectedSet:
                        doc_repr = carpe_diem.dictDocumentRepresentation[can]
                        doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                        c.append(doc_repr)
                        c_id.append(can)
                pred_first = sess.run(doc_pred_id_first, feed_dict={input_query: [q_test], candidate: c})

                listSelectedSet.append(c_id[pred_first])
                doc_repr = carpe_diem.dictDocumentRepresentation[c_id[pred_first]]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                listSelectedSet_repr.append(doc_repr)

                while len(listSelectedSet) < 10:
                    c = []
                    c_id = []
                    for can in listTest:
                        if can not in listSelectedSet:
                            doc_repr = carpe_diem.dictDocumentRepresentation[can]
                            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                            c.append(doc_repr)
                            c_id.append(can)
                    pred = sess.run(doc_pred_id, feed_dict={input_query: [q_test], query_selected: listSelectedSet_repr, candidate: c})
    
                    listSelectedSet.append(c_id[pred])
                    doc_repr = carpe_diem.dictDocumentRepresentation[c_id[pred]]
                    doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                    listSelectedSet_repr.append(doc_repr)
                    
                # save result
                for id_num, doc_id_selected in enumerate(listSelectedSet):
                    fileTmpResult_policy.write(str(query_test) + ' Q0 ' + doc_id_selected + ' ' +str(id_num+1) + ' ' + str(len(listSelectedSet)-id_num) + ' ' + sys.argv[4] + '_' + sys.argv[3] + '\n')
                    fileTmpResult_policy.flush()

                # value function
                c = listSelectedSet_repr_q
                max_one_value_pred_test = float("-inf")
                one_doc_pred_test = ''
                for can in listTest:
                    if can not in listSelectedSet_q:
                        doc_repr = carpe_diem.dictDocumentRepresentation[can]
                        doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                        c_tmp = c + [doc_repr]
                        one_doc_value_pred_test = sess.run(value_pred_first, feed_dict={input_query: [q_test]})
                        if one_doc_value_pred_test > max_one_value_pred_test:
                            one_doc_pred_test = can
                            max_one_value_pred_test = one_doc_value_pred_test
    
                listSelectedSet_q.append(one_doc_pred_test)
                doc_repr = carpe_diem.dictDocumentRepresentation[one_doc_pred_test]
                doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                listSelectedSet_repr_q.append(doc_repr)

                while len(listSelectedSet_q) < 10:
                    c = listSelectedSet_repr_q
                    max_one_value_pred_test = float("-inf")
                    one_doc_pred_test = ''
                    for can in listTest:
                        if can not in listSelectedSet_q:
                            doc_repr = carpe_diem.dictDocumentRepresentation[can]
                            doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                            c_tmp = c + [doc_repr]
                            one_doc_value_pred_test = sess.run(value_pred, feed_dict={input_query: [q_test], query_selected: c_tmp})
                            if one_doc_value_pred_test > max_one_value_pred_test:
                                one_doc_pred_test = can
                                max_one_value_pred_test = one_doc_value_pred_test
        
                    listSelectedSet_q.append(one_doc_pred_test)
                    doc_repr = carpe_diem.dictDocumentRepresentation[one_doc_pred_test]
                    doc_repr = np.reshape(np.asarray(doc_repr), -1).tolist()
                    listSelectedSet_repr_q.append(doc_repr)
                    
                # save result
                for id_num, doc_id_selected in enumerate(listSelectedSet_q):
                    fileTmpResult_value.write(str(query_test) + ' Q0 ' + doc_id_selected + ' ' +str(id_num+1) + ' ' + str(len(listSelectedSet_q)-id_num) + ' ' + sys.argv[4] + '_' + sys.argv[3] + '\n')
                    fileTmpResult_value.flush()
    
                resultScore_ndcg_10 = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet, 10)
                resultScore_ndcg_5 = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet, 5)
                resultScore_srecall_10 = carpe_diem.subtopic_recall(str(query_test), listSelectedSet, 10)
                resultScore_srecall_5 = carpe_diem.subtopic_recall(str(query_test), listSelectedSet, 5)
                resultScore_err_10 = carpe_diem.expected_reciprocal_rank(str(query_test), listSelectedSet, 10)
                resultScore_err_5 = carpe_diem.expected_reciprocal_rank(str(query_test), listSelectedSet, 5)
                floatSumResultScore_ndcg_5 += resultScore_ndcg_5 / idealScore_ndcg_5
                floatSumResultScore_ndcg_10 += resultScore_ndcg_10 / idealScore_ndcg_10
                floatSumResultScore_srecall_5 += resultScore_srecall_5
                floatSumResultScore_srecall_10 += resultScore_srecall_10
                floatSumResultScore_err_5 += resultScore_err_5
                floatSumResultScore_err_10 += resultScore_err_10

                resultScore_ndcg_10_q = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet_q, 10)
                resultScore_ndcg_5_q = carpe_diem.alphaDCG(0.5, str(query_test), listSelectedSet_q, 5)
                resultScore_srecall_10_q = carpe_diem.subtopic_recall(str(query_test), listSelectedSet_q, 10)
                resultScore_srecall_5_q = carpe_diem.subtopic_recall(str(query_test), listSelectedSet_q, 5)
                resultScore_err_10_q = carpe_diem.expected_reciprocal_rank(str(query_test), listSelectedSet_q, 10)
                resultScore_err_5_q = carpe_diem.expected_reciprocal_rank(str(query_test), listSelectedSet_q, 5)
                floatSumResultScore_ndcg_5_q += resultScore_ndcg_5_q / idealScore_ndcg_5
                floatSumResultScore_ndcg_10_q += resultScore_ndcg_10_q / idealScore_ndcg_10
                floatSumResultScore_srecall_5_q += resultScore_srecall_5_q
                floatSumResultScore_srecall_10_q += resultScore_srecall_10_q
                floatSumResultScore_err_5_q += resultScore_err_5_q
                floatSumResultScore_err_10_q += resultScore_err_10_q
                
                dictResult[query_test] = [resultScore_ndcg_5 / idealScore_ndcg_5, resultScore_ndcg_10 / idealScore_ndcg_10, resultScore_srecall_5, resultScore_srecall_10, resultScore_err_5, resultScore_err_10, resultScore_ndcg_5_q / idealScore_ndcg_5, resultScore_ndcg_10_q / idealScore_ndcg_10, resultScore_srecall_5_q, resultScore_srecall_10_q, resultScore_err_5_q, resultScore_err_10_q]
                
                
            result_ndcg_5 = floatSumResultScore_ndcg_5 / len(dictResult.keys())
            result_ndcg_10 = floatSumResultScore_ndcg_10 / len(dictResult.keys())
            result_srecall_5 = floatSumResultScore_srecall_5 / len(dictResult.keys())
            result_srecall_10 = floatSumResultScore_srecall_10 / len(dictResult.keys())
            result_err_5 = floatSumResultScore_err_5 / len(dictResult.keys())
            result_err_10 = floatSumResultScore_err_10 / len(dictResult.keys())

            result_ndcg_5_q = floatSumResultScore_ndcg_5_q / len(dictResult.keys())
            result_ndcg_10_q = floatSumResultScore_ndcg_10_q / len(dictResult.keys())
            result_srecall_5_q = floatSumResultScore_srecall_5_q / len(dictResult.keys())
            result_srecall_10_q = floatSumResultScore_srecall_10_q / len(dictResult.keys())
            result_err_5_q = floatSumResultScore_err_5_q / len(dictResult.keys())
            result_err_10_q = floatSumResultScore_err_10_q / len(dictResult.keys())

            # metrics
            p_can = subprocess.Popen(['./ndeval', 'metrics/my_qrels.txt', carpe_diem.folder + '/tmp_result_policy_' + sys.argv[3] + '.txt'], shell=False, stdout=subprocess.PIPE, bufsize=-1)
            output_eval = p_can.communicate()
            output_eval = output_eval[-2].split('\n')[-2]
            output_eval = output_eval.split(',')
            metrics_err_5 = output_eval[2]
            metrics_err_10 = output_eval[3]
            metrics_ndcg_5 = output_eval[11]
            metrics_ndcg_10 = output_eval[12]
            metrics_srecall_5 = output_eval[20]
            metrics_srecall_10 = output_eval[21]

            p_can_q = subprocess.Popen(['./ndeval', 'metrics/my_qrels.txt', carpe_diem.folder + '/tmp_result_value_' + sys.argv[3] + '.txt'], shell=False, stdout=subprocess.PIPE, bufsize=-1)
            output_eval_q = p_can_q.communicate()
            output_eval_q = output_eval_q[-2].split('\n')[-2]
            output_eval_q = output_eval_q.split(',')
            metrics_err_5_q = output_eval_q[2]
            metrics_err_10_q = output_eval_q[3]
            metrics_ndcg_5_q = output_eval_q[11]
            metrics_ndcg_10_q = output_eval_q[12]
            metrics_srecall_5_q = output_eval_q[20]
            metrics_srecall_10_q = output_eval_q[21]

            carpe_diem.fileResult.write(str(e) + ' ' + str(iteration) + ' ' + str(result_ndcg_5) + ' ' + str(result_ndcg_10) + ' ' + str(result_srecall_5) + ' ' + str(result_srecall_10) + ' ' + str(result_err_5) + ' ' + str(result_err_10) + '\n')
            carpe_diem.fileResult.write(str(e) + ' ' + metrics_ndcg_5 + ' ' + metrics_ndcg_10 + ' ' + metrics_srecall_5 + ' ' + metrics_srecall_10 + ' ' + metrics_err_5 + ' ' + metrics_err_10 + '\n')
            carpe_diem.fileResult.write(str(e) + ' ' + str(iteration) + ' ' + str(result_ndcg_5_q) + ' ' + str(result_ndcg_10_q) + ' ' + str(result_srecall_5_q) + ' ' + str(result_srecall_10_q) + ' ' + str(result_err_5_q) + ' ' + str(result_err_10_q) + '\n')
            carpe_diem.fileResult.write(str(e) + ' ' + metrics_ndcg_5_q + ' ' + metrics_ndcg_10_q + ' ' + metrics_srecall_5_q + ' ' + metrics_srecall_10_q + ' ' + metrics_err_5_q + ' ' + metrics_err_10_q + '\n')
            carpe_diem.fileResult.write('\n')
            carpe_diem.fileResult.flush()

            saver.save(sess, folder + '/model_final_' + sys.argv[3] + '/' + 'model.ckpt', global_step=iteration)
            print 'Save model @ EPOCH %d' % iteration

        iteration += 1
        print datetime.datetime.now()
        print iteration

print "Game over!"
