import os
import re
import json
import numpy as np

from decision_making.llm_gen.utils import rule2desc, get_other_consistent_values
from decision_making.data.utils import get_chosen_data, get_data

coef=0.5
data_path = f"../results/golden_history-stoch{coef}.json" # data path
save_dir = f"../results/rules-stoch{coef}" # save dir

# rule root dir

sas_pairs = get_data(data_path)
root_dir = f"{save_dir}/cleaned_rules-after-self"
os.makedirs(root_dir, exist_ok=True)
SUMMARIZED_RULE_PROMPTS = ''
index=0
for select_goal in os.listdir(root_dir):
    select_goal = select_goal.replace(".json", "")
    try:
        select_goal = float(select_goal)
        if select_goal < 0:
            goal_desc_will = f"the team will receive a Penalty of {select_goal} reward"
            goal_desc = f"the team receive a Penalty of {select_goal} reward"
        else:
            goal_desc_will = f"the team will receive a Reward = {select_goal}"
            goal_desc = f"the team receives a Reward = {select_goal} (Game Win)"
    except:
        select_goal = tuple(map(int, re.findall(r'\d+', select_goal)))
        goal_desc = rule2desc([select_goal], True)
        goal_desc_will = goal_desc.replace("stands on", "will stand on")
        
    feature_dict = get_chosen_data(sas_pairs, select_goal, print_=False)
    with open(os.path.join(root_dir, f"{select_goal}.json"), "r") as f:
        rules = json.load(f)

    index += 1

    rule_prompt = f'{index}) Experiences related to **{goal_desc}**\n'
    rule_prompt_list = []
    
    
    consistent_values_list = []
    condition_list = []

    for i, rule in enumerate(rules):
        tmp = get_other_consistent_values(feature_dict, rule['rule'], goal_desc_will)
        if tmp is not None:
            p, tmp, consistent_values = tmp

            # tmp = tmp.replace(goal_desc, '')
            # tmp = tmp.replace(', .', '.')
            # tmp = tmp.replace(', ,', ',')
            # rule_prompt_list += [(p, tmp)]
            consistent_values_list.append(consistent_values)
    
    # keep the common item in consistent_values_list
    # consistent_values_list = [[, ,], [, , ,], ...]
    sets = [set((int(a), int(b)) for [a, b] in sublist) for sublist in consistent_values_list]

    common_items_set = set.intersection(*sets)

    common_items = [[a, np.int64(b)] for (a, b) in sorted(common_items_set)]

    for i, rule in enumerate(rules):
        tmp = get_other_consistent_values(feature_dict, rule['rule'], goal_desc_will, common_items)
        if tmp is not None:
            p, tmp, consistent_values = tmp

            tmp = tmp.replace(goal_desc, '')
            tmp = tmp.replace(', .', '.')
            tmp = tmp.replace(', ,', ',')
            rule_prompt_list += [(p, tmp)]
            # consistent_values_list.append(consistent_values)
    

    if len(rule_prompt_list) == 0:
        continue
    rule_prompt_list = sorted(rule_prompt_list, key=lambda x: x[0], reverse=True)
    rule_prompt_list = [r[1] for r in rule_prompt_list]
    rule_prompt_list = [f"\t- " + tmp for i, tmp in enumerate(rule_prompt_list)]
    if len(rule_prompt_list) > 0:
        rule_prompt_list = [f"\t- " + tmp for i, tmp in enumerate(rule_prompt_list)]
    rule_prompt += "\tConditions: " + rule2desc(common_items, True) + "\n"
    rule_prompt += "\n".join(rule_prompt_list)
    SUMMARIZED_RULE_PROMPTS += rule_prompt + "\n"

# save to txt file
with open(f"{save_dir}/SUMMARIZED_RULE_PROMPTS.txt", "w") as f:
    f.write(SUMMARIZED_RULE_PROMPTS)
       