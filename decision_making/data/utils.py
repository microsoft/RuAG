import numpy as np
import json

from decision_making.llm_gen.utils import rule2desc

def get_data(data_dir):
    with open(data_dir, "r") as f:
        history = json.load(f)

    sas_pairs = []
    for his in history:
        visited = [[], []]
        last_stand = [[0 for i in range(3)], [0 for i in range(3)]]
        for h in his:
            visited[0].append(h['info'][0])
            visited[1].append(h['info'][1])
            visited_onehot = [[0 for i in range(3)], [0 for i in range(3)]]
            if 2 in visited[0][:-1]:
                visited_onehot[0][0] = 1
            if 3 in visited[0][:-1]:
                visited_onehot[0][1] = 1
            if 4 in visited[0][:-1]:
                visited_onehot[0][2] = 1
            if 2 in visited[1][:-1]:
                visited_onehot[1][0] = 1
            if 3 in visited[1][:-1]:
                visited_onehot[1][1] = 1
            if 4 in visited[1][:-1]:
                visited_onehot[1][2] = 1
            current_stand = [[0 for i in range(4)], [0 for i in range(4)]]
            if h['info'][0] == 2:
                current_stand[0][0] = 1
            elif h['info'][0] == 3:
                current_stand[0][1] = 1
            elif h['info'][0] == 4:
                current_stand[0][2] = 1
            elif h['info'][0] == 7:
                current_stand[0][3] = 1
                
            if h['info'][1] == 2:
                current_stand[1][0] = 1
            elif h['info'][1] == 3:
                current_stand[1][1] = 1
            elif h['info'][1] == 4:
                current_stand[1][2] = 1
            elif h['info'][1] == 7:
                current_stand[1][3] = 1

            sas_pairs.append({'obs': h['obs'], 'act': h['act'], 'rew': h['rew'], 
                              'info': h['info'],
                              'visited': visited_onehot,
                              'last_stand': last_stand,
                              'stand': current_stand,
                             'game_win': int(h['rew'] > 95)}
                             )
            last_stand = current_stand
    return sas_pairs

def get_chosen_data(sas_pairs, select_goal, print_=True):
    if not isinstance(select_goal, tuple):
        feature_dict = {
            'matched': [],
            'unmatched': [],
            }
        for sa in sas_pairs:
            features = sa['obs'][0] + sa['obs'][1]
            act = [[0 for i in range(5)] for j in range(2)]
            act[0][sa['act'][0]] = 1
            act[1][sa['act'][1]] = 1
            features += (act[0] + act[1])

            
            if sa['rew'] == select_goal:
                feature_dict['matched'].append(features)
            else:
                feature_dict['unmatched'].append(features)
    
    else:
        feature_dict = {
            'matched': [],
            'unmatched': [],
            'matched_r': [], 
            'unmatched_r': [], 
        }
        for sa in sas_pairs:
            features = sa['obs'][0] + sa['obs'][1] 
            act = [[0 for i in range(5)] for j in range(2)]
            act[0][sa['act'][0]] = 1
            act[1][sa['act'][1]] = 1
            features += (act[0] + act[1])
            features += sa['visited'][0] + sa['visited'][1] + sa['stand'][0] + sa['stand'][1] +[ sa['game_win']]
            if features[select_goal[0]] == select_goal[1]:
                feature_dict['matched'].append(features)
                feature_dict['matched_r'].append(sa['rew'])
            else:
                feature_dict['unmatched'].append(features) 
                feature_dict['unmatched_r'].append(sa['rew'])
    
    # unique
    pre_matched = len(feature_dict['matched'])
    pre_unmatched = len(feature_dict['unmatched'])
    feature_dict['matched'] = np.array(list(set([tuple(d) for d in feature_dict['matched']])))
    feature_dict['unmatched'] = np.array(list(set([tuple(d) for d in feature_dict['unmatched']])))
    if print_: print(f"After removing duplicates [{len(feature_dict['matched']) + len(feature_dict['unmatched'])}]: {pre_matched} -> {len(feature_dict['matched'])} (matched), {pre_unmatched} -> {len(feature_dict['unmatched'])} (unmatched)")
    return feature_dict

