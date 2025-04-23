import random
import tqdm

import numpy as np
import json
import argparse


from decision_making.env import Alice_and_Bob


class CentralizedGoldenAgent:
    def __init__(self, env_info=None, coef=0.5):
        assert env_info is not None
        self.env_info = env_info
        print(self.env_info)
        self.stage = 0
        self.explore_ratio = coef
    def get_action(self, key_pos, avail_actions):
        if key_pos == 1: 
            action = 2
        elif key_pos == 7: 
            action = 3 
        elif key_pos == 3: 
            action = 1
        elif key_pos == 5: 
            action = 0
        elif key_pos == 6: 
            action = [a for a in avail_actions if a in [1, 3]] 
            action = random.choice(action)
        elif key_pos == 2: 
            action = [a for a in avail_actions if a in [0, 2]]
            action = random.choice(action)
        elif key_pos == 0: 
            action = [a for a in avail_actions if a in [1, 2]]
            action = random.choice(action)
        elif key_pos == 8: 
            action = [a for a in avail_actions if a in [0, 3]]
            action = random.choice(action)
        elif key_pos == 4:
            action = 4
        try:
            return action
        except:
            import ipdb; ipdb.set_trace()
            
    def reset(self):
        self.stage = 0
        self.alice_need_to_stay = False
        self.bob_need_to_stay = False
        
    def step(self, obs):
        obs_alice, obs_bob = obs
        avail_actions = []
        for i in range(2):
            avail_actions_temp = [4] # stop
            if obs[i][5] != self.env_info['obs2block']['wall']: avail_actions_temp.append(0) # up
            if obs[i][3] != self.env_info['obs2block']['wall']: avail_actions_temp.append(1) # down
            if obs[i][1] != self.env_info['obs2block']['wall']: avail_actions_temp.append(2) # left
            if obs[i][7] != self.env_info['obs2block']['wall']: avail_actions_temp.append(3) # right
            avail_actions.append(avail_actions_temp)
            
        if self.stage == 0:
            # if bob see the key
            key_pos = np.where(obs_bob[:9]==self.env_info['obs2block']['key'][0])[0]
            if len(key_pos) and key_pos[0] != 4:
                action_bob = self.get_action(key_pos, avail_actions[1])
            else:

                action_bob = random.choice(avail_actions[1])
            if random.random() < 0.6:
                action_alice = 3
            else:
                action_alice = random.choice(avail_actions[0])
            if (obs_alice[9:11] == (np.array(self.env_info['key_pos'][0]) - np.array(self.env_info['treasure_pos']))).all():
                self.stage = 1
        if self.stage == 1:
            # if see the key
            key_pos = np.where(obs_alice[:9]==self.env_info['obs2block']['key'][1])[0]

            if len(key_pos) and key_pos[0] != 4:
                action_alice = self.get_action(key_pos, avail_actions[0])
            else:
                if random.random() < 0.6:
                    action_alice = 3
                else:
                    action_alice = random.choice(avail_actions[0])
            action_bob = random.choice(avail_actions[1])       
            if (obs_bob[9:11] == (np.array(self.env_info['key_pos'][1]) - np.array(self.env_info['treasure_pos']))).all():
                self.stage = 2
        if self.stage == 2:
            # check who stand on lever
            lever_pos_alice = np.where(obs_alice[:9]==self.env_info['obs2block']['lever'])[0]
            lever_pos_bob = np.where(obs_bob[:9]==self.env_info['obs2block']['lever'])[0]
            if len(lever_pos_alice)  and lever_pos_alice[0] != 4:
                action_alice = self.get_action(lever_pos_alice, avail_actions[0])
                action_bob = random.choice(avail_actions[1])
            elif len(lever_pos_bob)  and lever_pos_bob[0] != 4:
                action_bob = self.get_action(lever_pos_bob, avail_actions[1])
                action_alice = random.choice(avail_actions[0])
            else:
                action_alice = random.choice(avail_actions[0])
                action_bob = random.choice(avail_actions[1])
                if (obs_bob[9:11] == (np.array(self.env_info['lever_pos']) - np.array(self.env_info['treasure_pos']))).all():
                    if random.random() < 0.5:
                        action_alice = 4 # stay
                    else:
                        action_alice = random.choice(avail_actions[0])
                    treasure_pos = obs_bob[11:13]
                    # step towards treasure pos
                    a_temp = []
                    if treasure_pos[0] > 0:
                        a_temp.append(2)
                    elif treasure_pos[0] < 0:
                        a_temp.append(3)
                    if treasure_pos[1] > 0:
                        a_temp.append(1)
                    elif treasure_pos[1] < 0:
                        a_temp.append(0)
                    if list(treasure_pos) == [0, -2]:
                        avail_actions.append(0)
                    elif list(treasure_pos) == [-2, 0]:
                        avail_actions.append(3)
                    a_temp = [a for a in a_temp if a in set(avail_actions[1])]
                    if len(a_temp):
                        action_bob = random.choice(a_temp)
                    else:
                        action_bob = random.choice(avail_actions[1])
                elif (obs_alice[9:11] == (np.array(self.env_info['lever_pos']) - np.array(self.env_info['treasure_pos']))).all():
                    if random.random() < 0.5:
                        action_bob = 4 # stay
                    else:
                        action_bob = random.choice(avail_actions[1])
                        
                    treasure_pos = obs_alice[11:13]
                    a_temp = []
                    if treasure_pos[0] > 0:
                        a_temp.append(2)
                    elif treasure_pos[0] < 0:
                        a_temp.append(3)
                    if treasure_pos[1] > 0:
                        a_temp.append(1)
                    elif treasure_pos[1] < 0:
                        a_temp.append(0)
                    if list(treasure_pos) == [0, -2]:
                        avail_actions.append(0)
                    elif list(treasure_pos) == [-2, 0]:
                        avail_actions.append(3)
                    a_temp = [a for a in a_temp if a in set(avail_actions[0])]
                    if len(a_temp):
                        action_alice = random.choice(a_temp)
                    else:
                        action_alice = random.choice(avail_actions[0])        
        if random.random() < self.explore_ratio:
            action_alice = random.choice(range(5))

        if random.random() < self.explore_ratio:
            action_bob = random.choice(range(5))

        return [action_alice, action_bob]
   

class Runner:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
    def play_game(self):
        history = []
        obs = self.env.reset()
        self.agents.reset()
        # self.env.render(0)
        rew_ep = 0
        for t in range(100):
            actions = self.agents.step(obs)
            next_obs, rewards, done, info = self.env.step(actions)
            rew_ep += rewards
            history.append({'obs': [o.astype(np.int32).tolist() for o in obs], 
                            'act': actions,
                            'rew': rewards, 
                            'done': done, 
                            'info': info})
            obs = next_obs
            if done:
                break
        return history, rew_ep



def main(coef):
    env = Alice_and_Bob()
    env_info = env.get_info()
    print(env_info)

    agents = CentralizedGoldenAgent(env_info, coef=coef)
    runner = Runner(env, agents)

    history = []
    rew_all = []
    win_list = []

    for n in tqdm.trange(1000):
        his_ep, rew_ep = runner.play_game()
        history.append(his_ep)
        rew_all.append(rew_ep)
        win_list.append(his_ep[-1]['rew'] > 95)

        tqdm.tqdm.write(
            f"Game {n}, win_num: {np.sum(win_list)} avg_rew: {np.mean(rew_all)}, "
            f"std_rew: {np.std(rew_all)}, avg_len: {len(his_ep)}, win_rate: {np.mean(win_list)}"
        )

        with open(f'../results/golden_history-stoch{coef}.json', 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coef', type=float, default=0.5, help='Exploration coefficient')
    args = parser.parse_args()

    main(args.coef)
