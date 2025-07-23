import joblib
import numpy as np
from sklearn.tree import _tree

def load_model(model_path="model.joblib"):
    saved = joblib.load(model_path)
    return saved['model'], saved['features']

def extract_rules(model, feature_names):
    tree = model.tree_
    rules = []

    def recurse(node, conditions):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            # LEFT: Feature is False (<= 0.5)
            left_conditions = conditions + [f"{name} is FALSE"]
            recurse(tree.children_left[node], left_conditions)

            # RIGHT: Feature is True (> 0.5)
            right_conditions = conditions + [f"{name} is TRUE"]
            recurse(tree.children_right[node], right_conditions)
        else:
            # Leaf node — decide if it's allow/deny
            allow_prob = tree.value[node][0][1]
            deny_prob = tree.value[node][0][0]
            decision = "ALLOW" if allow_prob > deny_prob else "DENY"
            rules.append((conditions, decision))

    recurse(0, [])
    return rules

def format_rules(rules):
    formatted = []
    for conds, decision in rules:
        rule_text = "If " + ", and ".join(conds) + f", then access is {decision}"
        formatted.append(rule_text)
    return formatted
