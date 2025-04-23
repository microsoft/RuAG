import json
import os
from collections import defaultdict


class RuleEvaluator:
    def __init__(self, file_path, batch_size):
        self.file_path = file_path


        self.batch_size = batch_size
        self.batch_index = 0

        self.relation_to_triples_batches = []
        self.relation_to_triples_all =  defaultdict(list)
        self._load_data_and_create_batches()

    def _load_data_and_create_batches(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        json_data_list = list(json_data.keys())

        # 将数据划分为指定大小的批次
        for i in range(0, len(json_data_list), self.batch_size):
            batch_keys = json_data_list[i:i + self.batch_size]
            batch_triples = []
            # 从每篇文章的响应中提取三元组
            for key in batch_keys:
                triples = json_data[key]['relations']
                batch_triples.extend(triples)
            # 转换为集合，避免重复，确保三元组是有效的
            triples_set = set(tuple(triple) for triple in batch_triples if triple[0] != triple[2])
            # 创建一个关系到三元组的映射
            relation_to_triples = defaultdict(list)
            for triple in triples_set:
                relation_to_triples[triple[1]].append(triple)
            self.relation_to_triples_batches.append(relation_to_triples)

        ## 生成全部all_triples
        all_triples = []
        json_data_list = list(json_data.keys())

        for key in json_data_list:
            # 从每篇文章的响应中提取三元组
            triples = json_data[key]['relations']
            all_triples.extend(triples)
        all_triples_set = set(tuple(triple) for triple in all_triples if triple[0] != triple[2])
        for triple in all_triples_set:
            self.relation_to_triples_all[triple[1]].append(triple)

    def evaluate_precision_in_batches(self, rule, target_relation):
        # 获取当前批次
        if not self.relation_to_triples_batches:
            return "No batches available"

        relation_to_triples = self.relation_to_triples_batches[self.batch_index]
        precision,total_predictions = self._evaluate_rule_precision(rule, target_relation, relation_to_triples)
        print("Precision for this batch:", precision)

        # 更新批次计数器
        self.batch_index = (self.batch_index + 1) % len(self.relation_to_triples_batches)
        return precision,total_predictions

    def evaluate_precision_all_batches(self, rule, target_relation):
        relation_to_triples = self.relation_to_triples_all
        precision,total_predictions = self._evaluate_rule_precision(rule, target_relation, relation_to_triples)
        print("Precision for all batch:", precision)
        return precision

    def evaluate_precision_and_occurrence_all_batches(self, rule, target_relation):
        relation_to_triples = self.relation_to_triples_all
        precision,total_predictions = self._evaluate_rule_precision(rule, target_relation, relation_to_triples)
        # print("Precision for all batch:", precision)
        return precision,total_predictions

    def _evaluate_rule_precision(self, rule, target_relation, relation_to_triples):
        correct_predictions = 0
        total_predictions = 0  # 防止除以零
        if len(rule) == 1:
            for start_triple in relation_to_triples[rule[0]]:
                X, r1, Y = start_triple
                total_predictions += 1
                for triple in relation_to_triples[target_relation]:
                    if triple[0] == X and triple[2] == Y and triple[1] != r1:
                        correct_predictions += 1
                        break

        if len(rule) == 2:
            for start_triple in relation_to_triples[rule[0]]:
                X, r1, Y = start_triple
                for end_triple in relation_to_triples[rule[1]]:
                    _, r2, Z = end_triple
                    if Y == end_triple[0]:  # 确保链的连贯性
                        total_predictions += 1
                        for triple in relation_to_triples[target_relation]:
                            if triple[0] == X and triple[2] == Z:
                                correct_predictions += 1
                                break

        precision = 0
        if total_predictions != 0:
            precision = correct_predictions / total_predictions

        return precision,total_predictions
