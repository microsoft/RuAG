import re
import time
import openai
import numpy as np
import time
from typing import List, Optional
import openai
import random

# from utils.cloudgpt_aoai import get_openai_token, auto_refresh_token

LLM_TARGET_PREDICTS = \
    [-10.0, 100.0] + [(42, 1), (43, 1), (44, 1), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1)]

NUM2COLOR = {0: 'white', 
             1: 'black', 
             2: 'purple', 
             3: 'yellow', 
             4: 'skyblue', 
             5: 'red', 
             6: 'blue', 
             7: 'green'}
NUM2LOCATION = {
    0: 'lower left',
    1: 'center left', #'above',
    2: 'upper left',
    3: 'lower center', # left
    4: 'self',
    5: 'upper center', #'right',
    6: 'lower right',
    7: 'center right',
    8: 'upper right'
}
NUM2ACTION={
    0: 'up', # ->
    1: 'down',
    2: 'left',
    3: 'right',
    4: 'stand'
}



def history2text(traj):
    text = ''
    for tt, timestep in enumerate(traj):
        visited_subgoals = [[], []]
        text += f"Observation at timestep {tt}:\n"
        for agent_id, (vec, act) in enumerate(zip(timestep['obs'], timestep['act'])):
            name = ['Alice', 'Bob'][agent_id]
            colored_vec = [f'{NUM2LOCATION[i]} block' + f': {NUM2COLOR[x] + (" (reachable)" if NUM2COLOR[x] != "black" else " (unreachable)")}' for i, x in enumerate(vec[:9])]
            colored_vec_text = ', '.join(colored_vec[:4] + colored_vec[5:])
            
            treasure_relative_pos = f"{int(abs(vec[-1]))} blocks {'down' if vec[-1] < 0 else 'up'} and {int(abs(vec[-2]))} blocks to the {'left' if vec[-2] < 0 else 'right'} of treasure"
            
            if vec[-2] == 0 and vec[-1] == 0:
                visited_subgoals[agent_id].append('green block')
            if vec[-2] == -6 and vec[-1] == 0:
                visited_subgoals[agent_id].append('skyblue block')
            if vec[-2] == 0 and vec[-1] == -6:
                visited_subgoals[agent_id].append('yellow block')
            if vec[-2] == -8 and vec[-1] == -2:
                visited_subgoals[agent_id].append('purple block')      
            
            
            text += f"{name}'s surrounding blocks and their colors are: {colored_vec_text}. " + \
                f"{name}'s is currently located {treasure_relative_pos}." + \
                f"{name}'s current action is {NUM2ACTION[act]}. " + \
                (f"The blocks {name} have already visited: {', '.join(set(visited_subgoals[agent_id]))}. " if visited_subgoals[agent_id] else "" )
        text += "The team reward is {}.\n".format(timestep['rew'])
    return text


def parse_response(response_text):
    thoughts_pattern = r'"Thoughts": "(.*?)"'
    plan_pattern = r'"Plan": "(.*?)"'
    action_pattern = r'"Chosen Action": "(.*?)"'
    reference_pattern = r'"Candidate Action Analysis": "(.*?)"'

    thoughts_match = re.search(thoughts_pattern, response_text, re.DOTALL)
    plan_match = re.search(plan_pattern, response_text, re.DOTALL)
    action_match = re.search(action_pattern, response_text, re.DOTALL)
    reference_match = re.search(reference_pattern, response_text, re.DOTALL)
    
    thoughts = thoughts_match.group(1).strip() if thoughts_match else None
    plan = plan_match.group(1).strip() if plan_match else None
    chosen_action = action_match.group(1).strip() if action_match else None
    reference = reference_match.group(1).strip() if reference_match else None

    return {
        "Thoughts": thoughts,
        "Plan": plan,
        "Chosen Action": chosen_action,
        "Candidate Action Analysis": reference
    }
 

def rule2desc(rule, no_aux=False):
    if isinstance(rule[0], int):
        rule = [rule]
    
    if 11 in [r[0] for r in rule] and 12 in [r[0] for r in rule]:
        r11 = [r for r in rule if r[0] == 11][0][1]
        r12 = [r for r in rule if r[0] == 12][0][1]
        obs_desc = f"Alice locates at {int(abs(r12))} blocks {'down' if r12 < 0 else 'up'} and {int(abs(r11))} blocks to the {'left' if r11 < 0 else 'right'} of treasure"
    elif 11 in [r[0] for r in rule]:
        r11 = [r for r in rule if r[0] == 11][0][1]
        obs_desc = f"Alice locates at {int(abs(r11))} blocks to the {'left' if r11 < 0 else 'right'} of treasure"
    elif 12 in [r[0] for r in rule]:
        r12 = [r for r in rule if r[0] == 12][0][1]
        obs_desc = f"Alice locates at {int(abs(r12))} blocks {'down' if r12 < 0 else 'up'} of treasure"
    elif 24 in [r[0] for r in rule] and 25 in [r[0] for r in rule]:
        r24 = [r for r in rule if r[0] == 24][0][1]
        r25 = [r for r in rule if r[0] == 25][0][1]
        obs_desc = f"Bob locates at {int(abs(r25))} blocks {'down' if r25 < 0 else 'up'} and {int(abs(r24))} blocks to the {'left' if r24 < 0 else 'right'} of treasure"
    elif 24 in [r[0] for r in rule]:
        r24 = [r for r in rule if r[0] == 24][0][1]
        obs_desc = f"Bob locates at {int(abs(r24))} blocks to the {'left' if r24 < 0 else 'right'} of treasure"
    elif 25 in [r[0] for r in rule]:
        r25 = [r for r in rule if r[0] == 25][0][1]
        obs_desc = f"Bob locates at {int(abs(r25))} blocks {'down' if r25 < 0 else 'up'} of treasure"
    else:
        obs_desc = ""
    
    act_desc = []
    for r in rule:
        if r[0] >= 26 and r[0] < 31:
            if r[1]:
                if NUM2ACTION[r[0] - 26] != 'stand':
                    act_desc += [f"Alice moves {NUM2ACTION[r[0] - 26]}"]
                else:
                    act_desc += [f"Alice keep standing on current block"]
            else:
                act_desc += [f"Alice's action is not {NUM2ACTION[r[0] - 26]}"]        
        elif r[0] >= 31 and r[0] < 36:
            if r[1]:
                if NUM2ACTION[r[0] - 31] != 'stand':
                    act_desc += [f"Bob moves {NUM2ACTION[r[0] - 31]}"]
                else:
                    act_desc += [f"Bob keep standing on current block"] 
            else:
                act_desc += [f"Bob's action is not {NUM2ACTION[r[0] - 31]}"]
    if act_desc:
        act_desc = ", ".join(act_desc)
    else:
        act_desc = ""
    rule_text = []
    for r in rule:
        if r[0] < 9:
            if r[0] == 4 and NUM2COLOR[r[1]] not in ['white', 'blue', 'red']:
                rule_text += [f"Alice stands on {NUM2COLOR[r[1]]} block"]
            else:
                rule_text += [f"Alice's {NUM2LOCATION[r[0]]} block is {NUM2COLOR[r[1]]}"]
        elif r[0] == 9:
            rule_text += [f"Alice's teammate's x-location is {r[1]}"]
            
        elif r[0] == 10:
            rule_text += [f"Alice's teammate's y-location is {r[1]}"]
        elif 13 <= r[0] < 22:
            if r[0] == 17 and NUM2COLOR[r[1]] not in ['white', 'blue', 'red']:
                rule_text += [f"Bob stands on {NUM2COLOR[r[1]]} block"]
            else:
                rule_text += [f"Bob's {NUM2LOCATION[r[0]-13]} block is {NUM2COLOR[r[1]]}"]
        elif r[0] == 22:
            rule_text += [f"Bob's teammate's x-location is {r[1]}"]
        elif r[0] == 23:
            rule_text += [f"Bob's teammate's y-location is {r[1]}"]

        elif r[0] == 36:
            if r[1]: 
                rule_text += [f"Alice visited purple block"]
            else:
                rule_text += [f"Alice did not visit purple block"]
        elif r[0] == 37:
            if r[1]: 
                rule_text += [f"Alice visited yellow block"]
            else:
                rule_text += [f"Alice did not visit yellow block"]
        elif r[0] == 38:
            if r[1]: 
                rule_text += [f"Alice visited skyblue block"]
            else:
                rule_text += [f"Alice did not visit skyblue block"]
        elif r[0] == 39:
            if r[1]: 
                rule_text += [f"Bob visited purple block"]
            else:
                rule_text += [f"Bob did not visit purple block"]
                
        elif r[0] == 40:
            if r[1]: 
                rule_text += [f"Bob visited yellow block"]
            else:
                rule_text += [f"Bob did not visit yellow block"]
        elif r[0] == 41:
            if r[1]: 
                rule_text += [f"Bob visited skyblue block"]
            else:
                rule_text += [f"Bob did not visit skyblue block"]
        elif r[0] == 42:
            if r[1]: 
                rule_text += [f"Alice stands on purple block"]
            else:
                rule_text += [f"Alice does not stand on purple block"]
        elif r[0] == 43:
            if r[1]: 
                rule_text += [f"Alice stands on yellow block"]
            else:
                rule_text += [f"Alice does not stand on yellow block"]
        elif r[0] == 44:
            if r[1]: 
                rule_text += [f"Alice stands on skyblue block"]
            else:
                rule_text += [f"Alice does not stand on skyblue block"]
        elif r[0] == 45:
            if r[1]: 
                rule_text += [f"Alice stands on green block"]
            else:
                rule_text += [f"Alice does not stand on green block"]
        elif r[0] == 46:
            if r[1]: 
                rule_text += [f"Bob stands on purple block"]
            else:
                rule_text += [f"Bob does not stand on purple block"]
        elif r[0] == 47:
            if r[1]: 
                rule_text += [f"Bob stands on yellow block"]
            else:
                rule_text += [f"Bob does not stand on yellow block"]
        elif r[0] == 48:
            if r[1]: 
                rule_text += [f"Bob stands on skyblue block"]
            else:
                rule_text += [f"Bob does not stand on skyblue block"]
        elif r[0] == 49:
            if r[1]: 
                rule_text += [f"Bob stand on green block"]
            else:
                rule_text += [f"Bob does not stand on green block"]
        elif r[0] == 50:
            if r[1]: 
                rule_text += [f"Game Win"]
        elif r[0] in [11, 12, 24, 25]:
            continue
        elif 26 <= r[0] < 36:
            continue
        else:
            raise ValueError(f"Unknown rule {r}")
    rule_text = ", ".join(rule_text)
    if no_aux:
        if obs_desc:
            return obs_desc + ', ' + act_desc  + rule_text
        else:
            return act_desc + rule_text
    if obs_desc and act_desc:
        return 'When ' + obs_desc +', ' + rule_text + 'if ' + act_desc 
    if obs_desc:
        return 'When ' + obs_desc +', ' + rule_text
    if act_desc:
        return 'When ' + rule_text +', ' + 'if ' + act_desc
    return 'When' + rule_text + ', '




def get_other_consistent_values(feature_dict, rule, goal_desc=None, common_items=None):
    data_array = np.concatenate([feature_dict['matched'], feature_dict['unmatched']])
    goal_matched = np.array([True for i in range(len(feature_dict['matched']))] + [False for i in range(len(feature_dict['unmatched']))])


    feature_matched = np.array([data_array[:, r[0]] == r[1] for r in rule])
    feature_matched = np.vstack(feature_matched).all(axis=0)
    both_matched = goal_matched & feature_matched
    
    matched_features = data_array[both_matched]
    consistent_values = []
    for index in range(matched_features.shape[1]):
        if index in [r[0] for r in rule]:
            continue
        if index in [0, 1, 2, 3, 5, 6, 7, 8]: # remove teammate's position and relative position of treasure
            continue
        if index in [13, 14, 15, 17, 18, 19, 20, 21]: # remove teammate's position relative position of treasure
            continue
        if index in [50]:
            continue
        if index in [9, 10]: # remove teammate's position and relative position of treasure
            continue
        if index in [22, 23]: # remove teammate's position relative position of treasure
            continue

        values = np.unique(matched_features[:, index])
        if len(values) == 1:
            if 'not' not in rule2desc([index, values[0]]) and 'self' not in rule2desc([index, values[0]]):
                if common_items is not None:
                    for common_item in common_items:
                        if values[0] not in common_item:
                            consistent_values.append([index, values[0]])
                else:   
                    consistent_values.append([index, values[0]])

    rule_prompt = f"{rule2desc(rule)}, then {goal_desc}. " 

    percentage = int(both_matched.sum()/goal_matched.sum()*100)
    if percentage <= 1:
        return None

    if 'matched_r' in feature_dict:
        rew = np.unique(feature_dict['matched_r'])
    else:
        rew = []
    if len(rew) == 1:
        if consistent_values:
            rule_prompt += f"In all these cases, {rule2desc(consistent_values, True)}, "
        rule_prompt += f"and the reward is {rew[0]}. "
    else:
        if consistent_values:
            rule_prompt += f"In all these cases, {rule2desc(consistent_values, True)}. "
    return percentage, rule_prompt, consistent_values

    
    
class LLM_Azure_Agent:
    def __init__(self, name="Alice", teammate="Bob", rules=None, client_id=None, client_secret=None):
        self.name = name
        self.teammate = teammate
        self.rules = rules
        
        self.client = openai.AzureOpenAI(
        azure_endpoint="https://cloudgpt-openai.azure-api.net/",
        azure_ad_token=get_openai_token(client_id=client_id, client_secret=client_secret),
        )
    def chat(self, test_chat_message, model="gpt-4o-20240513", max_retries =5, stream=False, args=[], **kwargs):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=model,
                        messages=test_chat_message,
                        temperature=0, #0.7,
                        max_tokens=2048,
                        top_p=1, #0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stream=stream  # 启用流式传输
                )
                if not stream: return response_stream.choices[0].message.content

                content = ''
                for response in response_stream:

                    if not response.choices:
                        continue
                    if response.choices[0].delta.role == 'system':
                        continue  # 跳过系统消息
                    if response.choices[0].delta.content is None:
                        continue
                    content += response.choices[0].delta.content
                return content # 请求成功，返回响应
            except openai.RateLimitError:
                time.sleep(1)  
                retry_count += 1  
                print(f"retry: {retry_count} | RateLimitError")
            except openai.AuthenticationError:
                self.client = openai.AzureOpenAI(
                    api_version="2024-04-01-preview",
                    azure_endpoint="https://cloudgpt-openai.azure-api.net/",
                    azure_ad_token=get_openai_token(client_id=None, client_secret=None),
                )
                time.sleep(1) 
                retry_count += 1
                print(f"retry: {retry_count} | AuthenticationError")


    def kill(self):
        stop_token_refresh = auto_refresh_token()
        stop_token_refresh()
        self.client = None
        print("Token refresh stopped and client set to None.")
        


class LLM_OpenAI_Agent:
    def __init__(self, name="Alice", teammate="Bob", rules=None, client_id=None, client_secret=None):
        self.name = name
        self.teammate = teammate
        self.rules = rules
        
        self.client = openai.OpenAI()
        
    def chat(
        self,
        test_chat_message: List[dict],
        model: str = "gpt-4o",
        max_retries: int = 5,
        stream: bool = False,
        args=[],
        **kwargs
    ) -> str:
        retry_count = 0
        while retry_count < max_retries:
            try:
                response_stream = self.client.chat.completions.create(
                    model=model,
                    messages=test_chat_message,
                    temperature=0,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stream=stream,
                    **kwargs
                )

                if not stream:
                    return response_stream.choices[0].message.content

                # 处理 stream 内容
                content = ''
                for response in response_stream:
                    if not response.choices:
                        continue
                    delta = response.choices[0].delta
                    if hasattr(delta, "role") and delta.role == "system":
                        continue
                    if delta.content is None:
                        continue
                    content += delta.content
                return content

            except openai.RateLimitError:
                retry_count += 1
                print(f"Retry {retry_count} due to RateLimitError")
                time.sleep(2)
            except openai.AuthenticationError:
                retry_count += 1
                print(f"Retry {retry_count} due to AuthenticationError")
                time.sleep(2)
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count} due to unknown error: {e}")
                time.sleep(2)

        raise RuntimeError("Max retries reached.")

    def kill(self):
        self.client = None
        print("Client set to None.")
class LLM_Agent(LLM_OpenAI_Agent):
    def vec2text(self, vec, info, t):
        colored_vec = [f'\t\t{NUM2LOCATION[i]} block' + f': {NUM2COLOR[x] + (" (reachable)" if NUM2COLOR[x] != "black" else " (unreachable)")}' for i, x in enumerate(vec[:9])]
        colored_vec_text = '\n'.join(colored_vec[:4] + colored_vec[5:])
        teammate_pos = f"{int(abs(vec[-3]))} blocks {'down' if vec[-3] < 0 else 'up'} and {int(abs(vec[-4]))} blocks to the {'left' if vec[-4] < 0 else 'right'} of treasure"
        treasure_relative_pos = f"{int(abs(vec[-1]))} blocks {'down' if vec[-1] < 0 else 'up'} and {int(abs(vec[-2]))} blocks to the {'left' if vec[-2] < 0 else 'right'} of treasure"
        current_stand = 'white block'
        if vec[-2] == 0 and vec[-1] == 0:
            current_stand = 'green block'
            self.visited_subgoals.append('green block')
        if vec[-2] == -6 and vec[-1] == 0:
            current_stand = 'skyblue block'
            self.visited_subgoals.append('skyblue block')
        if vec[-2] == 0 and vec[-1] == -6:
            current_stand = 'yellow block'
            self.visited_subgoals.append('yellow block')
        if vec[-2] == -8 and vec[-1] == -2:
            current_stand = 'purple block'
            self.visited_subgoals.append('purple block')      
          
        return (
            "\t- The blocks surrounding you and their colors are:\n {}. \n".format(colored_vec_text) +
            ("\t- The block you are currently on is {}. You can choose 'stand' to remain on this block in the next timestep. \n".format(current_stand) if current_stand is not None else '') +
            f"\t- Your teammate, {self.teammate}, is currently located at {teammate_pos}. \n" +
            f"\t- You are currently located at {treasure_relative_pos}. \n" + 
            (f"\t- The blocks you have already visited: {', '.join(set(self.visited_subgoals))}. \n" if self.visited_subgoals else "" )
        )

            
    def chat_act(self, obs, info, t):
        obs_description = self.vec2text(obs, info, t)
        prompt = f"You are {self.name}, currently collaborating with your teammate, {self.teammate}, in a grid world to obtain the treasure (green block). "
        prompt += "Due to the impossible communication with your teammate, please monitor the state of your teammate and adjust your plan in time. "
        prompt += "\n\n"
        prompt += f"## Game Win: \n\tYou or {self.teammate} reaches the treasure. Please actively collaborate with your teammate to achieve the goal. \n"
        prompt += "\n\n"
        
        prompt += f"## Candidate actions: \n"
        prompt += "\t'up': move to stand on your **upper center** block if not black;\n"
        prompt += "\t'down': move to stand on your **lower center** block if not black;\n"
        prompt += "\t'left': move to stand on your **center left** block if not black;\n"
        prompt += "\t'right': move to stand on your **center right** block if not black;\n"
        prompt += "\t'stand': keep standing on the current block. Be careful to stand on the same block for a long time.\n" 
        prompt += "\n"

       
        prompt += "## Explanation about your surrounding blocks: "
        prompt += "\n\t- Center left, center right, upper center, lower center blocks: you can only move to any of them as long as they are non-black blocks; otherwise, you will receive a penalty and stay on original block. " 
        prompt += "\n\t- Upper left, Upper right, Lower left, lower right: You need move twice to reach those blocks. So if you want to move to those blocks, please be careful to plan the path and make sure all the blocks in the path are movable. As an example: if you want to move up then right, please make sure both center right and upper center blocks are reachable. "
        prompt += "\n\n"

        prompt += "## Some examples to avoid obstacles:" 
        prompt += "\n\t- If you want to move to the lower right block and your center right block is black, you can move down first then right if your lower center blobk is white."
        prompt += "\n\t- If moving right would bring you closer to your destination but the 'center right block' is unmovable and 'lower center block' is movable, try moving down first, then moving left twice and finally up if applicable. Mention this in your plan if you want to do so. \n"

        prompt += "\n"
        if self.rules is not None:
            prompt += self.rules
            prompt += "\n"
        response_format = """Please response with your thoughts, plan, and chosen action in the following format:
{
    // Describe your initial thoughts, like analysising the key steps towards game win, identifying your subgoals, comparing your candidate actions, analysising the progress of your teammate, assessing your previous plan and making future plan.
    "Thoughts": "Let's think step by step! [your analysis here]",
    
    // Make your future plan after you take action at this timestep. The plan will be the reference of your future decision making.
    // Do not include the current chosen action in the plan.
    "Plan": "[fill your future plan here]",
    
    // Your action, make sure to choose from 'up', 'down', 'left', 'right', 'stand'.
    "Chosen Action": "[fill your final action choice here]"   
}
"""


        
        if len(self.history) > 0:
            previous_plan = self.history[-1][0]
            previous_action = self.history[-1][1]
            prompt += "## Your last action: \n"
            prompt += f"\tLast timestep, you choose to move {previous_action} " if previous_action != "stand" else "Last timestep, you chose to remain on the original block. "
            prompt += '\n\n'
            if previous_plan is not None:
                prompt += "## Your plan at last timestep: \n"
                prompt += previous_plan 
                prompt += '\n\n'
            prompt += 'Please reaccess your situation and make decisions based on your current observations and the previous plan. If necessary, you can choose to act without considering your plan.\n'
            prompt += '\n'
        prompt += f"## Your current observation: \n{obs_description}\n\n"
        # prompt += "\n"
        prompt += response_format
        chat_input = []
        for his in self.history_chat[-3:]:
            chat_input.append({"role": "user", "content": his[0]})
            chat_input.append({"role": "assistant", "content": his[1]})
            
        chat_input.append({"role": "user", "content": prompt})

        response = self.chat(chat_input)
        print("=======")
        print(self.name)
        print(prompt)
        print(response)
        print("=======")
        # extract action and reason
        response_dict = parse_response(response)
        thoughts = response_dict['Thoughts']
        plan = response_dict['Plan']
        action_text = response_dict['Chosen Action']
        references = response_dict['Candidate Action Analysis']
        if action_text == "up" or action_text == "move up":
            action = 0
        elif action_text == "down" or action_text == "move down":
            action = 1
        elif action_text == "left" or action_text == "move left":
            action = 2
        elif action_text == "right" or action_text == "move right":
            action = 3
        elif action_text == "stand":
            action = 4
        elif action_text == "lower left" or action_text == "lower right":
            action = 1
            action_text = "down"
        elif action_text in ["upper left", "upper right", "upper right", "upper left", "up-left", "up-right"]:
            action = 0
            action_text = "up"
        else:
            action_text = "stand"
            action=4
            print(action_text)
            # raise ValueError("Invalid action")
            print('Bug')
        self.history.append((plan, action_text))
        self.history_chat.append((obs_description, action_text))
        return action, action_text, obs_description, thoughts, plan, references, prompt, response
    def random_act(self,):
        return random.choice([0, 1, 2, 3])
    def reset(self):
        self.history = []
        self.history_chat = []
        self.visited_subgoals = []

