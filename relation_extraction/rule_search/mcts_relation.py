import os

from utils.monte_carlo_tree_search_model import MCTS, Node
import random
import json

import RuleEvaluator


class RelationRuleNode(Node):
    def __init__(self, rule, relations, target_relation, ruleEvaluator, potential_relations):
        self.rule = rule  # current rule ['r1', 'r2']
        self.relations = relations  # all optional body predicate
        self.target_relation = target_relation  # target predicate

        self.ruleEvaluator = ruleEvaluator
        self.potential_relations = potential_relations  #  current optional body predicate

        self.is_terminal_var = None

    def find_children(self):

        if self.is_terminal():  # leaf node, don`t expand了
            return set()
        children = set()
        for relation in self.relations:
            if relation in self.potential_relations and relation not in self.rule:
                new_rule = self.rule + [relation]
                child = RelationRuleNode(new_rule, self.relations, self.target_relation, self.ruleEvaluator,
                                         self.potential_relations)
                children.add(child)
        return children

    def find_random_child(self):
        if self.is_terminal():
            return None
        relation = random.choice(list(set(self.potential_relations) - set(self.rule)))
        new_rule = self.rule + [relation]
        return RelationRuleNode(new_rule, self.relations, self.target_relation, self.ruleEvaluator,
                                self.potential_relations)

    def is_terminal(self):

        if self.is_terminal_var == None:
            if len(self.rule) >= 2:
                self.is_terminal_var = True
            elif len(self.rule) < 2:
                precision = self.ruleEvaluator.evaluate_precision_all_batches(self.rule, self.target_relation)
                # while True:
                #     precision, total_prediction = self.ruleEvaluator.evaluate_precision_in_batches(self.rule,
                #                                                                                    self.target_relation)
                #     if total_prediction > 0:
                #         break
                # self.ruleEvaluator.evaluate_precision_in_batches(self.rule, self.target_relation) > 0.9
                if precision >= 0.9:
                    self.is_terminal_var = True
            elif len(list(set(self.potential_relations) - set(self.rule))) <= 0:
                self.is_terminal_var = True
            else:
                self.is_terminal_var = False
        else:
            return self.is_terminal_var
        return self.is_terminal_var  # or self.reward() > 1

    def reward(self):
        precision = self.ruleEvaluator.evaluate_precision_all_batches(self.rule, self.target_relation)

        return precision

    def __hash__(self):
        return hash(tuple(self.rule))

    def __eq__(self, other):
        return tuple(self.rule) == tuple(other.rule)


def extract_logic_rule(target_relation, relations, ruleEvaluator):
    root = RelationRuleNode([], relations, target_relation, ruleEvaluator, potential_relations=potential_relations)

    tree = MCTS(exploration_weight=0.7)
    board = root
    while True:
        if board.is_terminal():
            break

        for _ in range(EPOCH):
            tree.do_rollout(board)
            print('-----------------' + str(_) + '-------------------')
            # if _ > 10:
            #     exit()
        board = tree.choose(board)
        print('choose_board rule:', board.rule)
        # print(board.to_pretty_string())
        if board.is_terminal():
            break
    print("Extracted rule:", " , ".join(board.rule), "->", target_relation)

    # Get the sorted rules and their last rewards.
    sorted_rules = tree.get_all_rules_last_reward()
    return sorted_rules


def preprocess_relations(all_triples, target_relation):
    filtered_relations = set()
    target_triples = []
    for triple in all_triples:
        if triple[1] == target_relation:
            target_triples.append(triple)

    for triple in all_triples:
        X, relation, Y = triple
        #Either X or Y appears in r3. Given X r1 Y, Y r2 Z, and X r3 Z.
        for target_triple in target_triples:
            if X == target_triple[0] or Y == target_triple[2]:
                filtered_relations.add(relation)
    filtered_relations.discard(target_relation)
    return filtered_relations


def process_data(folder_path):
    with open("../dataset/relations_dict.json",'r') as f:
        relation_dict = json.load(f)
    top20_relations = set(relation_dict.keys())
    print(top20_relations)

    triples_set = set()
    train_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                    doc_relation = []
                    for r in data['relations']:
                        if r[2] in top20_relations:
                            doc_relation.append([r[0],r[2],r[1]])
                    train_data[os.path.splitext(file)[0].replace("_relations","")] = {
                        "document": data["content"],
                        "relations": doc_relation
                    }
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {file_path}")

    for file_name,doc in train_data.items():
        for relation in doc['relations']:
            if relation[0] != relation[2]:
                triples_set.add(tuple((relation[0], relation[1], relation[2])))

    return triples_set, train_data


if __name__ == "__main__":

    triples_set, train_data = process_data(folder_path='../dataset/entity_relations_pairs/train')
    removal_predicates = ["vs", "appears_in", "player_of"] # "vs", "appears_in", "player_of"
    if removal_predicates and len(removal_predicates) > 0:
        triples_set = [i for i in triples_set if i[1] not in removal_predicates]

    train_data_path = '../results/mcts_train_data.json'
    with open(train_data_path, 'w') as json_file:
        json.dump(train_data,json_file)

    relations_list = set([triple[1] for triple in triples_set])

    TOTAL_ = 0  # When calculating the reward, add it to the denominator to avoid the rule of sampling with a small sample size.

    all_logic_rules = {}

    EPOCH = 100

    BATCH_SZIE = 50
    ruleEvaluator = RuleEvaluator.RuleEvaluator(train_data_path, batch_size=BATCH_SZIE)

    # Take each relation as the target predicate
    for target_relation in relations_list:  #
        target_logic_rules = {}

        potential_relations = preprocess_relations(triples_set, target_relation)
        path_reward = extract_logic_rule(target_relation, relations_list, ruleEvaluator)
        for rule, reward in path_reward:
            path_reward_key = tuple([rule, target_relation])
            all_logic_rules[path_reward_key] = reward
            target_logic_rules[path_reward_key] = reward

        print("*****************************")
        print("Target Predicate Sorted Rules and Rewards:")
        sorted_rules = sorted(target_logic_rules.items(), key=lambda x: x[1], reverse=True)
        file_path = f'../results/target_predicate/rules_with_rewards_{target_relation}.txt'

        if not os.path.exists('../results/target_predicate/'):
            os.makedirs('../results/target_predicate/')
        with open(file_path, 'w') as file:
            for rule, reward in sorted_rules:
                if reward > 0:
                    print(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; Reward: {reward}")
                    file.write(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; Reward: {reward}\n")

    print("Final Sorted Rules and Rewards:")
    sorted_rules = sorted(all_logic_rules.items(), key=lambda x: x[1], reverse=True)
    file_path = f'../results/all_rules_with_rewards_final_{EPOCH}.txt'

    selected_rules = []
    with open(file_path, 'w') as file:
        for rule, reward in sorted_rules:
            if reward > 0:
                print(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; Reward: {reward}")
                file.write(f"Rule: {' , '.join(rule[0])} -> {rule[1]}; Reward: {reward}\n")
            if reward > 0.5:
                selected_rules.append({
                    "Rule":rule,
                    "Confidence":reward
                })

    with open(f"../results/selected_rules_with_reward.json",'w') as f:
        json.dump(selected_rules,f)