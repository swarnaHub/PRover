from pulp import *
import numpy as np
import json

def solve_LP(edge_logits, fact_rule_identifier, node_labels):
    prob = LpProblem("Node edge consistency ", LpMaximize)
    all_vars = {}

    print(node_labels)

    # Optimization Problem
    opt_prob = None
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            if i == j:
                continue
            var0 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_0", 0, 1, LpInteger)
            var1 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_1", 0, 1, LpInteger)
            all_vars[(i, j, 0)] = var0
            all_vars[(i, j, 1)] = var1
            if opt_prob is None:
                opt_prob = (1 - edge_logits[i][j]) * all_vars[(i, j, 0)] + edge_logits[i][j] * all_vars[(i, j, 1)]
            else:
                opt_prob += (1 - edge_logits[i][j]) * all_vars[(i, j, 0)] + edge_logits[i][j] * all_vars[(i, j, 1)]

    prob += opt_prob, "Maximum Score"

    # Constraints
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            if i == j:
                continue
            # An edge can either be present or not present
            prob += all_vars[(i, j, 0)] + all_vars[(i, j, 1)] == 1, "Exist condition" + str(i) + "_" + str(j)

            # An edge is present only if the two corresponding nodes are present
            if node_labels[i] == 0 or node_labels[j] == 0:
                prob += all_vars[(i, j, 1)] == 0, "Edge presence" + str(i) + "_" + str(j)

            # No edge can exist between two facts (includes NAF too)
            if fact_rule_identifier[i] == 0 and fact_rule_identifier[j] == 0:
                prob += all_vars[(i, j, 1)] == 0, "Fact Fact" + str(i) + "_" + str(j)

            # No edge can exist from a rule to a fact
            if fact_rule_identifier[i] == 1 and fact_rule_identifier[j] == 0:
                prob += all_vars[(i, j, 1)] == 0, "Rule Fact" + str(i) + "_" + str(j)

    # Make sure there is an edge from every node only if the number of predicted nodes is > 1
    if node_labels.count(1) > 1:
        for i in range(len(edge_logits)):
            if node_labels[i] == 0:
                continue
            sum = None
            for j in range(len(edge_logits)):
                if i == j:
                    continue
                if sum is None:
                    sum = all_vars[(i, j, 1)] + all_vars[(j, i, 1)]
                else:
                    sum += all_vars[(i, j, 1)] + all_vars[(j, i, 1)]
    
            prob += sum >= 1, "Node connected "+str(i)

    #prob.writeLP("output/NodeEdgeConsistency.lp")
    prob.solve()

    edges = []
    for v in prob.variables():
        if v.varValue > 0 and v.name.endswith("1"):
            #print(v.name, "=", v.varValue)
            name = v.name.split("_")
            n_i = int(name[1]) - 1
            n_j = int(name[2]) - 1
            edges.append((n_i, n_j))
    print("Max score = ", value(prob.objective))

    return edges

def get_fact_rule_identifiers():
    data_dir = "../data/depth-5"
    test_file = os.path.join(data_dir, "test.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-test.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    fact_rule_identifiers = []
    for record, meta_record in zip(f1, f2):
        record = json.loads(record)
        meta_record = json.loads(meta_record)
        nfact = meta_record["NFact"]

        sentence_scramble = record["meta"]["sentenceScramble"]

        fact_rule_identifier = []
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                fact_rule_identifier.append(0)
            else:
                fact_rule_identifier.append(1)
        fact_rule_identifier.append(0) #NAF
        for (j, question) in enumerate(record["questions"]):
            if question["meta"]["QDep"] != 3:
                continue
            fact_rule_identifiers.append(fact_rule_identifier)

    return fact_rule_identifiers


if __name__ == '__main__':
    edge_logit_file = open("../output/cycle_model/prediction_edge_logits_dev.lst", "r", encoding="utf-8-sig")
    edge_logits = edge_logit_file.read().splitlines()

    node_pred_file = open("../output/cycle_model/prediction_nodes_dev.lst", "r", encoding="utf-8-sig")
    node_preds = node_pred_file.read().splitlines()

    fact_rule_identifiers = get_fact_rule_identifiers()

    assert len(edge_logits) == len(node_preds)
    assert len(edge_logits) == len(fact_rule_identifiers)

    edge_assignments = []
    f = open("../output/edge_assignment_identifiers_d3.lst", "w")
    for (i, (edge_logit, node_pred)) in enumerate(zip(edge_logits, node_preds)):
        print(i)
        edge_logit = edge_logit[1:-1].split(", ")
        edge_logit = [float(logit) for logit in edge_logit]

        node_pred = node_pred[1:-1].split(", ")
        node_pred = [int(pred) for pred in node_pred]

        edge_logit = edge_logit[:len(node_pred)*len(node_pred)]

        edge_logit = np.array(edge_logit).reshape(len(node_pred), len(node_pred))

        edges = solve_LP(edge_logit, fact_rule_identifiers[i], node_pred)

        print(edges)
        f.write(str(edges))
        f.write("\n")
    f.close()

