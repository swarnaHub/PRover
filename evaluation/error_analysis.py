import json
import os
import networkx as nx

from proof_utils import get_proof_graph, get_proof_graph_with_fail

def get_node_edge_indices(proofs, sentence_scramble, nfact, nrule):
    all_node_indices, all_edge_indices = [], []
    for proof in proofs.split("OR"):
        node_indices = []
        edge_indices = []

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index - nfact)
            component_index_map[component] = i
        component_index_map["NAF"] = nfact+nrule

        for node in nodes:
            index = component_index_map[node]
            node_indices.append(index)

        edges = list(set(edges))
        for edge in edges:
            start_index = component_index_map[edge[0]]
            end_index = component_index_map[edge[1]]
            edge_indices.append((start_index, end_index))

        all_node_indices.append(node_indices)
        all_edge_indices.append(edge_indices)

    return all_node_indices, all_edge_indices

def get_gold_proof_nodes_edges(data_dir):
    test_file = os.path.join(data_dir, "test.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-test.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    gold_proofs = []
    for record, meta_record in zip(f1, f2):
        record = json.loads(record)
        meta_record = json.loads(meta_record)
        #if not record["id"].startswith("AttPosBirdsVar2"):
        #    continue

        sentence_scramble = record["meta"]["sentenceScramble"]
        for (j, question) in enumerate(record["questions"]):
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            proofs = meta_data["proofs"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]
            #if question["meta"]["QDep"] != 5:
            #    continue
            #print(question['label'])
            all_node_indices, all_edge_indices = get_node_edge_indices(proofs, sentence_scramble, nfact, nrule)
            gold_proofs.append((all_node_indices, all_edge_indices))

    return gold_proofs

def is_connected(edges):
    if len(edges) == 0:
        return True
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0], edge[1])

    return nx.is_connected(g)

def get_cc(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0], edge[1])

    return nx.number_connected_components((g))

def is_direction_wrong(gold_edge, pred_edge):
    if len(gold_edge) != len(pred_edge):
        return False

    for edge in gold_edge:
        if edge not in pred_edge and (edge[1], edge[0]) not in pred_edge:
            return False

    return True

if __name__ == '__main__':
    all_gold_proofs = get_gold_proof_nodes_edges("../data/depth-5")

    all_pred_edges = []
    with open("../output/edge_assignment_identifiers_overall_no_connec.lst", "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for line in lines:
            if line == "[]":
                all_pred_edges.append([])
            else:
                edges = line[2:-2].split("), (")
                pred_edges = []
                for edge in edges:
                    edge = edge.split(", ")
                    pred_edges.append((int(edge[0]), int(edge[1])))
                all_pred_edges.append(pred_edges)

    assert len(all_gold_proofs) == len(all_pred_edges)

    print("Num samples = " + str(len(all_gold_proofs)))

    count_disconnected = 0
    direction_wrong = 0
    missing_edges = {}
    for (i, gold_proofs) in enumerate(all_gold_proofs):
        gold_edges = gold_proofs[1]
        is_correct = False
        best_pred_edge = None
        best_gold_edge = None
        best_common_edge = -1
        for (j, gold_edge) in enumerate(gold_edges):
            pred_edge = all_pred_edges[i]

            if set(pred_edge) == set(gold_edge):
                is_correct = True
                break

            common_edge = len(set(pred_edge) & set(gold_edge))
            if common_edge > best_common_edge:
                best_common_edge = common_edge
                best_gold_edge = gold_edge
                best_pred_edge = pred_edge

        if not is_correct:
            if not is_connected(best_pred_edge):
                print(i)
                print(best_pred_edge)
                print(get_cc(best_pred_edge))
                count_disconnected += 1

            if is_direction_wrong(best_gold_edge, best_pred_edge):
                #print(best_gold_edge)
                #print(best_pred_edge)
                #print("\n")
                direction_wrong += 1



    print("Disconnected proofs = " + str(count_disconnected))
    print("Wrong direction proofs = " + str(direction_wrong))

