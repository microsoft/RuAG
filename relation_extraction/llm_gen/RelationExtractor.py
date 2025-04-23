import json
import os.path
import re
import time
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

import concurrent.futures

# from utils import cloudgpt_aoai

relationships_description = open("./prompt/component/relationships.txt", 'r', encoding='utf-8').read()
CoT_rule_prompt = open("./prompt/ruag_prompt.txt", 'r', encoding='utf-8').read()


class RelationExtractor:
    def __init__(self, engine="gpt-4-20230613",rule_des_file = "../results/rule_description.text",MAX_TOKENS=1000):
        self.engine = engine
        self.MAX_TOKENS = MAX_TOKENS
        self.model = "RuAG"
        self.IS_THREAD = False
        self.temperature = 0  # 0 0.7
        self.top_p = 1  # 0 0.95
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.response_format = {"type": "json_object"}

        self.RuAG_rule_des = open(rule_des_file, 'r', encoding='utf-8').read()

        self.similarity_top_k = 10
        self.prompt_strategy = "RuAG"

        self.documents_path = "../dataset/entity_relations_pairs/test/"

        # prompt constant part
        with open("../dataset/relations_dict.json") as file:
            self.relations_dict = json.load(file)

        self.relations_list = [relation for relation, des in self.relations_dict.items()]
        self.relation_metrics = {relation: {"TP": 0, "FP": 0, "FN": 0} for relation in
                                 self.relations_dict.keys()}

    def reset(self):
        self.relation_metrics = {relation: {"TP": 0, "FP": 0, "FN": 0} for relation in
                                 self.relations_dict.keys()}

    def extract_entities_and_relation(self, data):
        related_entities = set()
        relations_truth = set()

        for triplet in data['relations']:
            entity1, entity2, relation = triplet
            related_entities.add(entity1)
            related_entities.add(entity2)

            if relation in self.relations_list:
                relations_truth.add((entity1, relation, entity2))
        print(', '.join(related_entities))
        return related_entities, relations_truth

    def get_top_k_embeddings(self, query_embedding, doc_embeddings, doc_ids, similarity_top_k):
        """Get top nodes by similarity to the query.
        (query_embedding: List[float],
        doc_embeddings: List[List[float]],
        doc_ids: List[str],
        similarity_top_k: int = 5,) -> Tuple[List[float], List]:
        """
        # dimensions: D
        qembed_np = np.array(query_embedding)
        # dimensions: N x D
        dembed_np = np.array(doc_embeddings)
        # dimensions: N
        dproduct_arr = np.dot(dembed_np, qembed_np)
        # dimensions: N
        norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
            dembed_np, axis=1, keepdims=False
        )
        # dimensions: N
        cos_sim_arr = dproduct_arr / norm_arr

        # now we have the N cosine similarities for each document
        # sort by top k cosine similarity, and return ids
        tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
        sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

        if similarity_top_k < len(sorted_tups):
            sorted_tups = sorted_tups[:similarity_top_k]

        result_similarities = [s for s, _ in sorted_tups]
        result_ids = [n for _, n in sorted_tups]
        return result_similarities, result_ids
    #
    # def rule_base_generate(self):
    #     with open("./dataset/logic_rules.json") as file:
    #         logic_rules_dict = json.load(file)
    #
    #     logic_rules = [i for i in logic_rules_dict.values()]
    #     logic_rules_key = [i for i in logic_rules_dict.keys()]
    #
    #     # to emb,get_top_k_embeddings
    #     rule_embs = cloudgpt_aoai.get_embedding(logic_rules)
    #     rule_embs = [rule_embs[i].embedding for i in range(len(rule_embs))]
    #     np.save('./dataset/logic_rule_embs.npy', rule_embs)

    def getPrompt(self, article_id):
        with open(self.documents_path + article_id, "r", encoding='utf-8') as file:
            document_data = json.load(file)
        related_entities, relations_truth = self.extract_entities_and_relation(document_data)
        rule_des = self.RuAG_rule_des

        prompt = CoT_rule_prompt.format(
            relationships=relationships_description,
            Document=document_data["content"],
            Entities='; '.join(related_entities) + ".\n",
            rules=rule_des
        )
        return prompt, related_entities, relations_truth

    def get_chat_completion_with_retry(self, engine, messages, max_retries=10, sleep_duration=5):
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = cloudgpt_aoai.get_chat_completion(
                    engine=engine,
                    messages=messages,
                    temperature=self.temperature,  # 0 0.7
                    # max_tokens=self.MAX_TOKENS,
                    top_p=self.top_p,  # 0 0.95
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=None,
                    response_format=self.response_format

                )

                return response
            except openai.RateLimitError:
                print(f"Rate limit reached. Retrying after {sleep_duration} seconds...")
                time.sleep(sleep_duration)
                retry_count += 1

        raise Exception("Request failed: Maximum number of retries exceeded.")

    def relation_extract(self, article_id):
        prompt_message, related_entities, relations_truth = self.getPrompt(article_id)

        print(prompt_message)
        test_chat_message = [{"role": "user", "content": prompt_message}]
        # "system": "You are an AI Assistant and always write the output of your response in json."}]
        response = self.get_chat_completion_with_retry(engine=self.engine, messages=test_chat_message)

        print("-----------message.content-----------")
        predicted_relations_str = response.choices[0].message.content  # ['choices'][0]['message']["content"]
        print(predicted_relations_str)

        if self.response_format == {"type": "json_object"}:
            llm_output_json = json.loads(predicted_relations_str)
            triplets = []
            for item in llm_output_json['result']:
                triplets.append([item["subject"], item["relation"], item["object"]])

        else:
            if predicted_relations_str:
                predicted_relations_str = predicted_relations_str.replace('"', "'")
            triplets_srt = re.findall(r'\((.*?)\)', predicted_relations_str)

            triplets = []
            for triplet in triplets_srt:
                items = [item.strip(" '") for item in triplet.split(", ")]
                triplets.append(items)

        relations_dict = self.prompt_data["relation_des"]["relation_dict"]
        relations_list = [relation for relation, des in relations_dict.items()]
        # print(f"test:{relations_list}")

        predicted_relations = set()
        for items in triplets:
            if len(items) < 3:
                print("len(items) < 3 ")
                continue
            if items[1] in relations_list:
                if items[0] in related_entities and items[2] in related_entities:
                    relation_triplet = tuple(items)
                    predicted_relations.add(relation_triplet)

            else:
                print(relations_list)

        # Update the True Positives (TP), False Positives (FP), and False Negatives (FN) for each relationship after processing each document.
        for relation in relations_dict:
            predicted = set(filter(lambda x: x[1] == relation, predicted_relations))
            truth = set(filter(lambda x: x[1] == relation, relations_truth))

            TP = len(predicted.intersection(truth))
            FP = len(predicted - truth)
            FN = len(truth - predicted)

            self.relation_metrics[relation]["TP"] += TP
            self.relation_metrics[relation]["FP"] += FP
            self.relation_metrics[relation]["FN"] += FN

        print("-----------predicted_relations-----------")
        print(predicted_relations)
        print("-----------relations_truth-----------")
        print(relations_truth)
        print("-----------predicted_relations-relations_truth-----------")
        print(predicted_relations - relations_truth)

        true_positives = predicted_relations.intersection(relations_truth)
        precision = len(true_positives) / len(predicted_relations) if predicted_relations else 0
        recall = len(true_positives) / len(relations_truth) if relations_truth else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        print(f"f1:{f1}; recall:{recall}; precision:{precision}")

        result_data = {
            "article_id": article_id,
            "F1": f1,
            "Recall": recall,
            "Precision": precision

        }
        result_set = {}
        result_set["predicted_relations"] = list(predicted_relations)
        result_set["relations_truth"] = list(relations_truth)
        return result_data, result_set

    def extract_main(self):
        directory_path = Path("../dataset/entity_relations_pairs/test")
        files = sorted(directory_path.glob('*.json'))
        file_names = [file.name for file in files]

        # These articles violate the GPT processing protocol.
        need_filtered = ["DW_16083654_relations.json", "DW_44141017_relations.json", "DW_17347807_relations.json",
                         "DW_17736433_relations.json", "DW_18751636_relations.json", "DW_19210651_relations.json",
                         "DW_39718698_relations.json"]  # ground truth为0

        results_df = pd.DataFrame(columns=["article_id", "F1", "Recall", "Precision"])

        error_articles = []

        def process_article(article_id):
            try:
                if article_id not in need_filtered:
                    print(f"****************{article_id}********************")
                    result_data, result_set = self.relation_extract(article_id=article_id)
                    return article_id, result_data, result_set
            except Exception as e:
                error_articles.append(article_id)
                print(f"Error processing {article_id}: {e}")
            return None, None, None

        result_sets = {}

        if not os.path.exists(f"../results/LLM_Gen/{self.engine}/"):
            os.makedirs(f"../results/LLM_Gen/{self.engine}/")


        if self.IS_THREAD:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(process_article, article_id): article_id for article_id in file_names}
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(file_names)):
                    try:
                        article_id, result_data, result_set = future.result()
                        if article_id is not None:
                            results_df = results_df._append(result_data, ignore_index=True)
                            result_sets[article_id] = result_set
                            results_df.to_excel(
                                f"../results/LLM_Gen/{self.engine}/{self.prompt_strategy}_{len(file_names)}.xlsx"
                            )
                    except Exception as e:
                        error_articles.append(futures[future])
                        print(f"Failed to process article {futures[future]}: {e}")

            with open(
                    f"../results/LLM_Gen/{self.engine}_{self.prompt_strategy}_{len(file_names)}_temp.json",
                    'w') as file:
                json.dump(result_sets, file, indent=4)
            if error_articles:
                print(f"These articles failed to process: {error_articles}")
            else:
                print()
        else:
            result_sets = {}
            for i, article_id in tqdm(enumerate(file_names), total=len(file_names)):
                if article_id not in need_filtered:
                    print(f"****************{article_id}********************")
                    result_data, result_set = self.relation_extract(article_id=article_id)
                    results_df = results_df._append(result_data, ignore_index=True)

                    result_sets[article_id] = result_set
                    results_df.to_excel(
                        f"./results/LLM_Gen/{self.engine}/{self.prompt_strategy}_{len(file_names)}.xlsx")

        total_TP, total_FP, total_FN = 0, 0, 0
        for relation, metrics in self.relation_metrics.items():
            TP, FP, FN = metrics["TP"], metrics["FP"], metrics["FN"]
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            total_TP += TP
            total_FP += FP
            total_FN += FN
            self.relation_metrics[relation].update({"Precision": precision, "Recall": recall, "F1": f1})

        # # Calculate the overall Precision, Recall, and F1 metrics.
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        result_data = {
            "article_id": "all",
            "F1": f1,
            "Recall": recall,
            "Precision": precision

        }

        results_df = results_df._append(result_data, ignore_index=True)
        results_df.to_excel(
            f"../results/LLM_Gen/{self.engine}/{self.prompt_strategy}_{len(file_names)}_f1_{f1:.3f}.xlsx")

        print(f"overall Precision: {precision}")
        print(f"overall Recall: {recall}")
        print(f"overall F1: {f1}")

        df = pd.DataFrame.from_dict(self.relation_metrics, orient='index')
        df.to_excel(f'../results/LLM_Gen/{self.engine}/{self.prompt_strategy}_relation_metrics_f1_{f1:.3f}.xlsx')

        return f1, recall, precision
