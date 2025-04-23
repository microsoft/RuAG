import copy
import json
import numpy as np
import os
import random

from tqdm import trange
from utils.monte_carlo_tree_search_model import MCTS, Node
from decision_making.llm_gen.utils import rule2desc, LLM_TARGET_PREDICTS
from decision_making.data.utils import get_data, get_chosen_data


class TrajectoryNode(Node):
    def __init__(self, rule=None, remaining_features=None, all_feature=None, action_space = None, select_goal=None, max_length=5, parent_metric_list=[]):
        self.rule = rule if rule is not None else []
        self.max_length = max_length
        self.remaining_features = remaining_features
        self.all_feature = all_feature
        self.action_space = action_space
        self.select_goal = select_goal
        self.parent_metric_list = parent_metric_list

            
        defined_index = [r[0] for r in self.rule]
        self.candidates_node = [r for r in self.action_space if r[0] not in defined_index]
        self.minimal_traj_num = 1
    
    def init_children(self, ):
        if len(self.remaining_features) < self.minimal_traj_num:
            return set()
        children = set()
        for new_rule_node in self.candidates_node:
            new_rule = self.rule + [new_rule_node]
            new_remaining_features = [f for f in self.remaining_features if f[new_rule_node[0]] == new_rule_node[1]]
            if len(new_remaining_features) < self.minimal_traj_num:
                continue
            if not len(self.parent_metric_list) == len(self.rule)+1:
                self.node_reward()
            children.add(TrajectoryNode(new_rule, new_remaining_features, self.all_feature, self.action_space, self.select_goal, self.max_length, self.parent_metric_list))
        self.children_set = children
        return children
    def find_children(self):
        if hasattr(self, 'children_set'):
            return self.children_set
        return self.init_children()

    def find_random_child(self):
        if not hasattr(self, 'children_set'):
            self.init_children()
        return np.random.choice(list(self.children_set))
    def is_terminal(self):
        if not hasattr(self, 'reward_value'):
            self.node_reward()
        if not hasattr(self, 'children_set'):
            self.init_children()
        if len(self.candidates_node) == 0 or len(self.children_set) == 0:
            return True
            
        if self.precision == 1 or self.recall_negative == 1:
            return True
        if len(self.remaining_features) < self.minimal_traj_num:
            return True
        return len(self.rule) > self.max_length

    def node_reward(self):
        self.reward_value, self.feature_count, self.accuracy, precision, recall, f1 = evaluate_rule(self.rule, self.all_feature)
        precision, precision_negative = precision
        recall, recall_negative = recall
        f1, f1_negative = f1
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.precision_negative = precision_negative
        self.recall_negative = recall_negative
        self.f1_negative = f1_negative

        self.parent_metric_list = self.parent_metric_list + \
            [copy.deepcopy(([[self.precision, self.precision_negative], [self.recall, self.recall_negative], [self.f1, self.f1_negative]]))]
        assert len(self.parent_metric_list) == len(self.rule)+1
        return self.reward_value
    def reward(self):
        if not hasattr(self, 'reward_value'):
            self.node_reward()
        return self.reward_value

    def __hash__(self):
        return hash(frozenset(self.rule))

    def __eq__(self, other):
        return frozenset(self.rule) == frozenset(other.rule)


def evaluate_rule(rule, data):
    rule = sorted(rule, key=lambda x: x[0])
    data_array = np.concatenate([data['matched'], data['unmatched']])
    goal_matched = np.array([True for i in range(len(data['matched']))] + [False for i in range(len(data['unmatched']))] )
    feature_matched = np.array([data_array[:, r[0]] == r[1] for r in rule])
    if not len(feature_matched):
        accuracy = precision = (goal_matched).mean()
        precision_negative = 0
        recall = 1
        recall_negative = 0
        f1 = (1 + 0.5*0.5) * precision * recall / (0.5*0.5*precision + recall)
        
        f1_negative = 0
        feature_count = goal_matched.sum()
    else:
        all_feature_matched = np.vstack(feature_matched).all(axis=0)
        feature_not_matched = 1 - np.vstack(feature_matched).all(axis=0)
        feature_count = (all_feature_matched[goal_matched]).sum()

        # accuracy
        accuracy = (all_feature_matched == goal_matched).mean()
        
        # precision
        precision = (all_feature_matched * goal_matched).sum() /all_feature_matched.sum()
        precision_negative = (feature_not_matched * (1 - goal_matched)).sum() / feature_not_matched.sum()
    
        # recall 
        recall = (goal_matched * all_feature_matched).sum() / goal_matched.sum()
        recall_negative = ((1 - goal_matched) * feature_not_matched).sum() / (1 - goal_matched).sum()
        
        f1 = (1 + 0.5*0.5) * precision * recall / (0.5*0.5*precision + recall)
        f1_negative = (1 + 0.5*0.5) * precision_negative * recall_negative / (0.5*0.5*precision_negative + recall_negative)
        # return precision

    reward = precision * 2 + feature_count / len(data['matched']) * (precision > 0.99)

    return reward, feature_count, accuracy, [precision, precision_negative], [recall, recall_negative], [f1, f1_negative]


def search_rule(select_goal, feature_dict, max_length, search_epoch_num=500):
        # get value space of each feature
        MCTS_ACTIONS = []
        for k in feature_dict.keys():
            feature_dict[k] = np.array(feature_dict[k])
        print("matched:", feature_dict['matched'].shape)
        print("unmatched:", feature_dict['unmatched'].shape)
        if feature_dict['matched'].shape[0] < 3:
            return []
        all_feat_tmp = np.concatenate([feature_dict['matched'], feature_dict['unmatched']])
        unique_values_log = []
        for i in range(all_feat_tmp.shape[1]):
            unique_values = np.unique(all_feat_tmp[:, i])
            for unique_action in unique_values:
                if len(unique_values) < 2:
                    print('Skip', i, f"[{rule2desc([[i, unique_values[0]]])}]", unique_values, 'because of single value of unique action.')
                    unique_values_log.append((i, unique_values))
        for i in range(feature_dict['matched'].shape[1]):
            # print(i, unique_action)
            unique_values = np.unique(feature_dict['matched'][:, i])
            
            for unique_action in unique_values:
                if [i, unique_action] in unique_values_log:
                    continue
                if isinstance(select_goal, tuple):
                
                    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8]: # remove teammate's position and relative position of treasure
                        continue
                    if i in [13, 14, 15, 16, 17, 18, 19, 20, 21]: # remove teammate's position relative position of treasure
                        continue
                    
                if i in [9, 10]: # remove teammate's position and relative position of treasure
                    continue
                if i in [22, 23]: # remove teammate's position relative position of treasure
                    continue
                
                if i in [4, 17]:
                    continue
                
                if i in [50, ]:
                    continue
                
                MCTS_ACTIONS.append((i, unique_action))
            
        if isinstance(select_goal, tuple):
             MCTS_ACTIONS = [rr for rr in MCTS_ACTIONS if rr[0] != select_goal[0]]

        sub_actions_dict = {k: [] for k in MCTS_ACTIONS}
        for aa in MCTS_ACTIONS:
            sub_actions_dict[aa] = [aa_sub for aa_sub in MCTS_ACTIONS if aa_sub[0] != aa[0]]
        
        print(f"Number of actions: {len(MCTS_ACTIONS)} -> {np.mean([len(sub_actions_dict[aa]) for aa in MCTS_ACTIONS])}")

        
        board = TrajectoryNode(rule=[], remaining_features=feature_dict['matched'], all_feature=feature_dict, action_space=MCTS_ACTIONS, select_goal=select_goal, max_length=max_length)

        mcts = MCTS()

        while True:
        
            for search_epoch in trange(search_epoch_num):
                mcts.do_rollout(board)
            board = mcts.choose(board)
            print('Chosen rule:', rule2desc(board.rule))
            if board.is_terminal():
                break
        return mcts.get_all_rules_all_simulated_path()


def main(coef=0.5):

    seed = 345
    random.seed(seed)
    np.random.seed(seed)


    sas_pairs = get_data(f"../results/golden_history-stoch{coef}.json")

    
    goals_for_searching = LLM_TARGET_PREDICTS

    max_length = 5
    all_rules = {str(k): [] for k in goals_for_searching}

    for select_goal in goals_for_searching:
        
        save_name = f"../results/rules-stoch{coef}/raw/{select_goal}.json"
        
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        print("=====================================")
        print(f"Search rules for obtaining goal {select_goal} {(':' + rule2desc([select_goal])) if isinstance(select_goal, tuple) else ''}")

        feature_dict = get_chosen_data(sas_pairs, select_goal)

        sorted_rules_org = search_rule(select_goal, feature_dict, max_length, search_epoch_num=500)
        
        sorted_rules = [r for r in sorted_rules_org if r[1] > 2]
        sorted_rules = [(sorted(list(r[0]), key=lambda x: x[0]), r[1]) for r in sorted_rules]
        

        for rule, reward in sorted_rules:
            rule = sorted(rule, key=lambda x: x[0])
            rule_text = rule2desc(rule)
            eval_r = evaluate_rule(rule, feature_dict)
            all_rules[str(select_goal)].append({
                'rule_text': rule_text, 
                'rule': [[int(v) for v in r] for r in rule],
                'MCTS_reward': reward,
                'evaluation': [float(qq) if isinstance(qq, np.int64) else qq for qq in eval_r],
                })

        with open(save_name, "w") as f:
            json.dump(all_rules[str(select_goal)], f, indent=4)
            

                
    return all_rules

if __name__ == "__main__":
    coef=0.5
    main(coef)