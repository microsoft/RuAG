import json
import numpy as np
import copy
from tqdm import tqdm
import os

from decision_making.data.utils import get_data, get_chosen_data
from decision_making.llm_gen.utils import rule2desc

from decision_making.rule_search.MCTS_search import evaluate_rule, LLM_TARGET_PREDICTS


def remove_duplicate_rule(all_rules):
    all_rules_removed = copy.deepcopy(all_rules)

    while True:
        need_to_remove = []
        for rule in all_rules_removed:
            if rule['rule_text'] in need_to_remove:
                continue
            rule_text = rule['rule_text']
            rule_org = [tuple(r) for r in rule['rule']]
            reward = rule['MCTS_reward']
            for other_rule in all_rules_removed:
                if rule['rule_text'] in need_to_remove:
                    continue
                other_rule_text = other_rule['rule_text']
                other_rule_org = other_rule['rule']
                other_rule_org = [tuple(r) for r in other_rule['rule']]
                other_reward = other_rule['MCTS_reward']
                if rule_text == other_rule_text:
                    continue
                if set(tuple(rule_org)).issubset(set(tuple(other_rule_org))): # rule_org < other_rule_org
                    gap = set(other_rule_org) - set(rule_org)
                    for gap_i in gap:
                        if gap_i[0] >=36:
                            continue
                    if reward >= other_reward:
                        if other_rule['rule_text'] not in need_to_remove:
                            need_to_remove.append(other_rule['rule_text'])
        if len(need_to_remove) == 0:
            break 
        all_rules_removed = [rule_tmp for rule_tmp in all_rules_removed if rule_tmp['rule_text'] not in need_to_remove]

    return all_rules_removed


if __name__ == "__main__":
    coef = 0.5
    select_goals = LLM_TARGET_PREDICTS
    offline_data_path = f"../results/golden_history-stoch{coef}.json"
    
    # rule root dir
    root_dir = f"../results/rules-stoch{coef}"


    sas_pairs = get_data(offline_data_path)

    for goal in select_goals:
        print('\n')

        if isinstance(goal, tuple):
            print(goal, rule2desc([goal]))
            rule_print_info = rule2desc([goal])
        else:
            print('reward:', goal)
            rule_print_info = f"reward: {goal}"
            
        feature_dict = get_chosen_data(sas_pairs, goal)
        
        if len(feature_dict['matched']) == 0 or len(feature_dict['unmatched']) == 0:
            print(f"Goal {goal} not matched: {rule_print_info}")
            continue
        data_array = np.concatenate([feature_dict['matched'], feature_dict['unmatched']])
        goal_matched = np.array([True for i in range(len(feature_dict['matched']))] + [False for i in range(len(feature_dict['unmatched']))])
        data_dir = f"{root_dir}/raw/{goal}.json"
        if not os.path.exists(data_dir):
            print(f"File {data_dir} not exists: {rule_print_info}")
            continue
        with open(data_dir) as f:
            rules = json.load(f)
        
        print(len(rules))
        rule_processed = []

        direct_load = False
        if not direct_load:
            for rule in tqdm(rules):
                rule_org = rule['rule']
                rule_temp = copy.deepcopy(rule_org)
                while True:
                    flag = False
                    feature_matched = np.array([data_array[:, r[0]] == r[1] for r in rule_temp])
                    feature_matched = np.vstack(feature_matched).all(axis=0)
                    for con in rule_temp:
                        rule_new = [r for r in rule_temp if r != con]
                        
                        feature_matched_new = np.array([data_array[:, r[0]] == r[1] for r in rule_new])
                        feature_matched_new = np.vstack(feature_matched_new).all(axis=0)
                        eval_r = evaluate_rule(rule_org, feature_dict)
                        eval_r_new = evaluate_rule(rule_new, feature_dict)

                        if not np.isnan(eval_r_new[0]) and not np.isnan(eval_r[0]) and (eval_r[0] <= eval_r_new[0]):
                            flag = True
                            break
                    if not flag:
                        break
                    rule_temp = copy.deepcopy(rule_new)
                if rule_temp not in rule_processed:
                    rule_processed.append(rule_temp)



            print('After self-removing', len(rule_processed))
            rule_save = []
            for rule in rule_processed:
                rule_text = rule2desc(rule)
                eval_r = evaluate_rule(rule, feature_dict)
                rule_save.append({
                    'rule_text': rule_text, 
                    'rule': [[int(v) for v in r] for r in rule],
                    'MCTS_reward': eval_r[0],
                    'evaluation': [float(qq) if isinstance(qq, np.int64) else qq for qq in eval_r],
                    })

            os.makedirs(f'{root_dir}/self-removed/', exist_ok=True)
            with open(f'{root_dir}/self-removed/{goal}.json', 'w') as f:
                json.dump(rule_save, f, indent=4)
        else:
            with open(f'{root_dir}/self-removed/{goal}.json', 'r') as f:
                rule_save = json.load(f)
        if len(rule_save) != 0:
            if max([r['MCTS_reward'] for r in rule_save]) > 2.1:
                th = 2.1
            else:
                th = np.percentile([r['MCTS_reward'] for r in rule_save], 10)
            print('max reward', max([r['MCTS_reward'] for r in rule_save]))
            print('threshold', th)
            clean_rules = [r for r in rule_save if r['MCTS_reward'] >= th]
            os.makedirs(f'./{root_dir}/reward-removed', exist_ok=True)
            with open(f'./{root_dir}/reward-removed/{goal}.json', 'w') as f:
                json.dump(clean_rules, f, indent=4)
            print('After reward-removing', len(clean_rules))


            clean_rules = remove_duplicate_rule(clean_rules)
            
            print('After cross-removing', len(clean_rules))
            

            save_name = f"./{root_dir}/cleaned_rules-after-self/{goal}.json"

            os.makedirs(f'./{root_dir}/cleaned_rules-after-self', exist_ok=True)
                
            with open(save_name, "w") as f:
                json.dump(clean_rules, f, indent=4)
                