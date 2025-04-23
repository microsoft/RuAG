# RuAG: Learned-Rule-Augmented Generation for Large Language Models

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue?logo=OpenReview)](https://openreview.net/forum?id=BpIbnXWfhL)

Welcome to the official code repository for our ICLR 2025 paper, **RuAG: Learned-rule-augmented Generation for Large Language Models**.


---

## 📘 Abstract

In-context learning (ICL) and Retrieval-Augmented Generation (RAG) have shown promise in enhancing LLMs’ reasoning by incorporating external knowledge. However, they are limited by context window size and often fail to inject sufficient information.
We propose **RuAG**, a framework that **automatically distills large-scale offline data into interpretable first-order logic rules**, which are then injected into LLM prompts to enhance reasoning capabilities. RuAG first relies on LLMs’ commonsense to define rule predicates, then applies **Monte Carlo Tree Search (MCTS)** to efficiently discover symbolic logic rules. These rules are translated into natural language and seamlessly integrated into downstream tasks.
We evaluate RuAG on both public and private industrial datasets across domains including **natural language processing**, **time-series**, and **decision-making**.

![Framework](./figures/framework.pdf)

---

## 🧪  Experiments

This repository provides code and data for **🧾 Relation Extraction**  and **🕹️ Decision Making**. 

### Installation

As the begining, please run the following for installation.

```
pip install -e .
```

### Relation Extraction

This repository contains the codebase for the `Relation Extraction` task in RuAG, structured into clear stages for ease of use. We use [DWIE](https://github.com/klimzaporojets/DWIE.git) (Deutsche Welle corpus for Information Extraction) in document-level multi-task Information Extraction (IE).
Our experiment is based on the top 20 relations(`./dataset/relation_dicts.json`) of the DWIE dataset and all the articles that do not violate the GPT protocol. Finally, each article is processed into the following format:

```json
{
  "content": "article content",
  "relations": [["entity1", "relation1", "entity2"], ...]
}
``` 
- Rule Searching: Navigate to the rule search directory and execute the following steps:

```shell
cd rule_search

python mcts_relation.py

```

- LLM's generation: Use the discovered rules to extract relation:

```shell
cd llm_gen
python extract_main.py
```


### Decision-Making
Here is the instruction for decision-making task in the paper. We choose the cooperative multi-agent game [Alice&Bob](https://github.com/zowiezhang/STAS), which requires both planning and collaboration. In the game, Alice and Bob work together to find the treasure and the optimal paths for both agents often go against intuition, as shown in the figure.
![Case Study](./figures/case_study.pdf)




- Offline data collection. You can configure the hyper-parameter `coef` that controls the possibilities to choosen optimal actions. Trajectories will be stored in `golden_history-stoch{coef}.json`
```shell

cd decision_making/data

python golden_player.py --coef 0.5

```

- Search and Process Rules 
```shell
cd decision_making/rule_search

# rule search
python MCTS_search.py

# post process
python post_process.py

# translate rules into natural languages
python rules2text.py

```

- LLM's generation
```shell
python play_game.py
```

---

## 📚 Citation

If you find this work helpful, please cite us:

```bibtex
@inproceedings{
    zhang2025ruag,
    title={Ru{AG}: Learned-rule-augmented Generation for Large Language Models},
    author={Yudi Zhang and Pei Xiao and Lu Wang and Chaoyun Zhang and Meng Fang and Yali Du and Yevgeniy Puzyrev and Randolph Yao and Si Qin and Qingwei Lin and Mykola Pechenizkiy and Dongmei Zhang and Saravan Rajmohan and Qi Zhang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=BpIbnXWfhL}
    }
```