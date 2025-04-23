import json

input_file = "../results/selected_rules_with_reward.json"
output_file = "../results/rule_description.text"
with open(input_file,'r') as f:
    data = json.load(f)


texts = []
for rule in data:
    body_predicates = rule["Rule"][0]
    target_predicate = rule["Rule"][1]
    confidence = rule["Confidence"]  # 置信度

    entity_count = len(body_predicates) + 1
    entities = [chr(65 + i) for i in range(entity_count)]  # A, B, C...

    conditions = []
    for i, predicate in enumerate(body_predicates):
        subject = entities[i]
        object_ = entities[i + 1]
        conditions.append(f"{subject} has relation {predicate} with {object_}")

    condition_str = " and ".join(conditions)

    conclusion = f"{entities[0]} and {entities[-1]} have relation {target_predicate}"

    description = (
        f"If {condition_str}, then {conclusion}, "
        f"with confidence {confidence:.4f}"  # 保留6位小数，可根据需求调整
    )

    print(description)
    texts.append(description)

with open(output_file,'w') as f:
    for i in texts:
        f.write(i)
        f.write("\n")

