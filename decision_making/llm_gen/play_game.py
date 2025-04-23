import json
import numpy as np
import os
from tqdm import trange

from decision_making.env import Alice_and_Bob
from decision_making.llm_gen.utils import LLM_Agent



class Runner:
    def __init__(self, env, agents, name='none'):
        self.env = env
        self.agents = agents
        self.runner_name = name
    def test_rest(self, ):
        for i in range(2):
            self.agents[i].reset()
        self.env.render(0)
        
    def play_game(self, max_iter=50):
        history = []
        obs = self.env.reset()
        info=None
        for i in range(2):
            self.agents[i].reset()
            info=None
        self.env.render(0)
        for t in trange(max_iter):
            actions_reasons = [self.agents[i].chat_act(obs[i], info, t) for i in range(2)]
            
            actions, action_texts, obs_texts, thoughts, plans, references, prompts, responses = [], [], [], [], [], [], [], []
            for i in range(2):
                actions.append(actions_reasons[i][0])
                action_texts.append(actions_reasons[i][1])
                obs_texts.append(actions_reasons[i][2])
                thoughts.append(actions_reasons[i][3])
                plans.append(actions_reasons[i][4])
                references.append(actions_reasons[i][5])
                prompts.append(actions_reasons[i][6])
                responses.append(actions_reasons[i][7])
            
            next_obs, rewards, done, info = self.env.step(actions)
            print(rewards)
            history.append({'obs': [o.astype(np.int32).tolist() for o in obs], 
                            'obs_text': obs_texts,
                            'act': actions,
                            'action_text': action_texts,
                            'thoughts': thoughts,
                            'plans': plans,
                            'reference': references,
                            'prompt': prompts,
                            'llm_response': responses,
                            'rew': rewards, 
                            'done': done, 
                            'info': info})
            
            self.env.render(t+1)
            obs = next_obs
            if done:
                break
        return history


def evaluate(tips_choice, save_dir, coef=0.5):

    if tips_choice == "RuAG":

        tips = "## Helpful summarized experiences for infering the key step \n"
        tips += "Since it is not applicable to navigate to the treasure directly, please analysis the summarized experiences to infer:"
        tips += "\n\t- the key steps before the game win\n\t- the agents' task assignment for game win, especially the final timestep: \n"
        
        with open(f'../results/rules-stoch{coef}/SUMMARIZED_RULE_PROMPTS.txt', 'r') as f:
            tips += f.read()
        
        tips += "\n"

    else:
        assert False, "Invalid tips choice"

    env = Alice_and_Bob()

    alice = LLM_Agent("Alice", "Bob", tips)
    bob = LLM_Agent("Bob", "Alice", tips)
    
    agents = [alice, bob]
    
    runner = Runner(env, agents, name=tips_choice)

    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, tips_choice + '.json')
    history = []
    
    for _ in trange(30):
        his_ep = runner.play_game()

        history += [his_ep]
        with open(save_dir, 'w') as f:
            json.dump(history, f, indent=4)

    for agent in agents:
        agent.kill()

if __name__ == "__main__":
    save_dir = "../results/final" # save path
    tips_choice_list = ["RuAG"]

    for tips_choice in tips_choice_list:
        print(save_dir)
        evaluate(tips_choice, save_dir, coef=0.5)
        