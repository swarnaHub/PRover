from pulp import *
import numpy as np
import json
import argparse

def solve_LP_unconstrained(edge_logits):
    prob = LpProblem("Node edge consistency ", LpMaximize)
    all_vars = {}

    # Optimization Problem
    opt_prob = None
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            var0 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_0", 0, 1, LpInteger)
            var1 = LpVariable("Edge_" + str(i + 1) + "_" + str(j + 1) + "_1", 0, 1, LpInteger)

            all_vars[(i, j, 0)] = var0
            all_vars[(i, j, 1)] = var1

            if opt_prob is None:
                opt_prob = (1 - edge_logits[i][j]) * all_vars[(i, j, 0)] + edge_logits[i][j] * all_vars[(i, j, 1)]
            else:
                opt_prob += (1 - edge_logits[i][j]) * all_vars[(i, j, 0)] + edge_logits[i][j] * all_vars[(i, j, 1)]

    prob += opt_prob, "Maximum Score"

    # An edge is either present or absent
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            prob += all_vars[(i, j, 0)] + all_vars[(i, j, 1)] == 1, "Exist condition" + str(i) + "_" + str(j)

    prob.solve()

    edges = []
    for v in prob.variables():
        if v.varValue > 0 and v.name.endswith("1") and v.name.startswith("Edge"):
            name = v.name.split("_")
            n_i = int(name[1]) - 1
            n_j = int(name[2]) - 1
            edges.append((n_i, n_j))
    print("Max score = ", value(prob.objective))

    return edges

def solve_LP(edge_logits, fact_rule_identifier, node_labels):
    prob = LpProblem("Node edge consistency ", LpMaximize)
    all_vars = {}

    all_flow_vars = {}

    source_id = -1
    sink_id = -2

    print(node_labels)

    # get ids of nodes that are present
    node_ids_present = []
    for i in range(len(node_labels)):
        if int(node_labels[i]) == 1:
            node_ids_present.append(i)

    if len(node_ids_present) == 0:
        return []

    # add flow from source to one node present
    # arbitarily choosing that node to be the last node
    # 1000 is infinity
    all_flow_vars[(source_id, node_ids_present[-1])] = \
        LpVariable("Flow_source_" + str(node_ids_present[-1] + 1), 0, 1000, LpInteger)

    # add flow from all nodes present to sink
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        all_flow_vars[(temp, sink_id)] = LpVariable("Flow_" + str(temp + 1) + "_sink", 0, 1000, LpInteger)

    # define capacities
    C = {}
    # capacity from source to 1st node is number of nodes in graph
    C[(source_id, node_ids_present[-1])] = len(node_ids_present)
    C[(node_ids_present[-1], source_id)] = 0

    # capacity from nodes in graph to sink is 1
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        C[(temp, sink_id)] = 1
        C[(sink_id, temp)] = 0

    # capacities inside graph are infinite or say 1000 in this case, except self loops and if the edge is not possible
    arcs = set()
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            if (i == j) or (i not in node_ids_present) or (j not in node_ids_present):
                C[(i, j)] = 0
            else:
                C[(i, j)] = 1000
                arcs.add((i, j))
                arcs.add((j, i))
    arcs = list(arcs)

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

            f_var = LpVariable("Flow_" + str(i + 1) + "_" + str(j + 1), 0, 1000, LpInteger)
            all_flow_vars[(i, j)] = f_var

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

            # flow less than capacity
            prob += all_flow_vars[(i, j)] <= C[(i, j)], "Capacity constraint " + str(i) + " " + str(j)

    # capacity constraint of source to 1st node
    prob += all_flow_vars[(source_id, node_ids_present[-1])] <= C[
        (source_id, node_ids_present[-1])], "Capacity constraint source " + str(node_ids_present[-1])

    # capacity constraint of nodes to sink
    for i in range(len(node_ids_present)):
        temp = node_ids_present[i]
        prob += all_flow_vars[(temp, sink_id)] == C[(temp, sink_id)], "Capacity constraint " + str(temp) + " sink"

    # node flow conservation constraint
    for n in range(len(edge_logits)):
        if n == node_ids_present[-1]:
            prob += (all_flow_vars[(source_id,n)] + lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if j == n]) ==
                     lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if i == n])+ all_flow_vars[(n,sink_id)]), \
                    "Flow Conservation in Node " + str(n)
        elif n in node_ids_present and n != node_ids_present[-1]:
            prob += (lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if j == n ]) ==
                     lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if i == n ]) + all_flow_vars[(n,sink_id)]), \
                    "Flow Conservation in Node " + str(n)
        else:
            prob += (lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if j == n]) ==
                     lpSum([all_flow_vars[(i, j)] for (i, j) in arcs if i == n])), \
                    "Flow Conservation in Node " + str(n)

    # Max flow should be equal to number of nodes in graph
    # to ensure this make the flow from source exactly equal to capacity
    # also ensure that the flow occurs only when the edge exists
    prob += all_flow_vars[(source_id, node_ids_present[-1])] == C[(source_id, node_ids_present[-1])]
    for i in range(len(edge_logits)):
        for j in range(len(edge_logits)):
            if i == j:
                continue
            prob += len(node_ids_present)*(all_vars[i, j, 1] + all_vars[j, i, 1]) - all_flow_vars[(i, j)] >= 0, "Valid flow " + str(
                    i + 1) + " " + str(j + 1)

    prob.solve()

    edges = []
    for v in prob.variables():
        if v.varValue > 0 and v.name.endswith("1") and v.name.startswith("Edge"):
            name = v.name.split("_")
            if name[1] == 'source' or name[2] == 'sink' or name[2] == 'source' or name[1] == 'sink':
                continue
            n_i = int(name[1]) - 1
            n_j = int(name[2]) - 1
            edges.append((n_i, n_j))
    print("Max score = ", value(prob.objective))

    return edges


def get_fact_rule_identifiers(data_dir):
    test_file = os.path.join(data_dir, "dev.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-dev.jsonl")

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
        fact_rule_identifier.append(0)  # NAF
        for (j, question) in enumerate(record["questions"]):
            #if question["meta"]["QDep"] != 5:
            #    continue
            fact_rule_identifiers.append(fact_rule_identifier)

    return fact_rule_identifiers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--node_preds", default=None, type=str, required=True)
    parser.add_argument("--edge_logits", default=None, type=str, required=True)
    parser.add_argument("--edge_preds", default=None, type=str, required=True)

    args = parser.parse_args()

    edge_logit_file = open(args.edge_logits, "r", encoding="utf-8-sig")
    edge_logits = edge_logit_file.read().splitlines()

    node_pred_file = open(args.node_preds, "r", encoding="utf-8-sig")
    node_preds = node_pred_file.read().splitlines()

    fact_rule_identifiers = get_fact_rule_identifiers(args.data_dir)

    assert len(edge_logits) == len(node_preds)
    assert len(edge_logits) == len(fact_rule_identifiers)

    edge_assignments = []
    f = open(args.edge_preds, "w")
    for (i, (edge_logit, node_pred)) in enumerate(zip(edge_logits, node_preds)):
        print(i)
        edge_logit = edge_logit[1:-1].split(", ")
        edge_logit = [float(logit) for logit in edge_logit]

        node_pred = node_pred[1:-1].split(", ")
        node_pred = [int(pred) for pred in node_pred]

        edge_logit = edge_logit[:len(node_pred) * len(node_pred)]

        edge_logit = np.array(edge_logit).reshape(len(node_pred), len(node_pred))

        edges = solve_LP(edge_logit, fact_rule_identifiers[i], node_pred)
        # Use this for unconstrained inference -- maximizes only the edge logits
        #edges = solve_LP_unconstrained(edge_logit)

        print(edges)
        f.write(str(edges))
        f.write("\n")
    f.close()

