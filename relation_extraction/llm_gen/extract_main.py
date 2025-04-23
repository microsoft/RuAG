

import RelationExtractor

if __name__ == '__main__':

    # engine = "gpt-4o-20240513" # "gpt-35-turbo-16k-20230613"
    engine = "gpt-4-1106-preview" # "gpt-35-turbo-16k-20230613"
    MAX_TOKENS = 2000
    IS_THREAD = True # Is  multithreaded
    rule_des_file = "../results/rule_description.text"
    relationExtractor = RelationExtractor.RelationExtractor(engine=engine,
                                                                  MAX_TOKENS=MAX_TOKENS)
    relationExtractor.IS_THREAD= IS_THREAD

    relationExtractor.documents_path = "../dataset/entity_relations_pairs/test/"
    f1, re, pre = relationExtractor.extract_main()
    print(f"f1:{f1},re:{re},pr{pre}")


