import argparse
import os
import json
import numpy as np

from proof_utils import get_proof_graph, get_proof_graph_with_fail

def get_node_edge_label(proofs, sentence_scramble, nfact, nrule):
    all_node_labels, all_edge_labels = [], []
    for proof in proofs.split("OR"):
        # print(proof)
        node_label = [0] * (nfact + nrule + 1)
        edge_label = np.zeros((nfact + nrule + 1, nfact + nrule + 1), dtype=int)

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)
        # print(nodes)
        # print(edges)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index - nfact)
            component_index_map[component] = i

        for node in nodes:
            if node != "NAF":
                index = component_index_map[node]
            else:
                index = nfact + nrule
            node_label[index] = 1

        edges = list(set(edges))
        for edge in edges:
            if edge[0] != "NAF":
                start_index = component_index_map[edge[0]]
            else:
                start_index = nfact + nrule
            if edge[1] != "NAF":
                end_index = component_index_map[edge[1]]
            else:
                end_index = nfact + nrule

            if start_index < end_index:
                edge_label[start_index][end_index] = 1
            else:
                edge_label[end_index][start_index] = 2

        # Set lower triangle labels to -100
        edge_label[np.tril_indices((nfact + nrule + 1), 0)] = -100

        all_node_labels.append(node_label)
        all_edge_labels.append(list(edge_label.flatten()))

    return all_node_labels, all_edge_labels

def get_gold_proof_nodes_edges(data_dir):
    test_file = os.path.join(data_dir, "test.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-test.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    gold_proofs = []
    for record, meta_record in zip(f1, f2):
        record = json.loads(record)
        meta_record = json.loads(meta_record)

        sentence_scramble = record["meta"]["sentenceScramble"]
        for (j, question) in enumerate(record["questions"]):
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            proofs = meta_data["proofs"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]
            all_node_labels, all_edge_labels = get_node_edge_label(proofs, sentence_scramble, nfact, nrule)
            gold_proofs.append((all_node_labels, all_edge_labels))

    return gold_proofs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--node_pred_file", default=None, type=str, required=True)
    parser.add_argument("--edge_pred_file", default=None, type=str, required=True)

    args = parser.parse_args()
    all_gold_proofs = get_gold_proof_nodes_edges(args.data_dir)

    all_pred_nodes = []
    with open(args.node_pred_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for line in lines:
            all_pred_nodes.append([int(x) for x in line[1:-1].split(",")])

    all_pred_edges = []
    with open(args.edge_pred_file, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for line in lines:
            all_pred_edges.append([int(x) for x in line[1:-1].split(",")])

    assert len(all_gold_proofs) == len(all_pred_nodes)
    assert len(all_gold_proofs) == len(all_pred_edges)

    print(len(all_gold_proofs))

    correct_nodes = 0
    correct_edges = 0
    correct_graphs = 0
    for (i, gold_proofs) in enumerate(all_gold_proofs):
        gold_nodes = gold_proofs[0]
        gold_edges = gold_proofs[1]

        for (j, gold_node) in enumerate(gold_nodes):
            if gold_node == all_pred_nodes[i]:
                correct_nodes += 1
                break

        for (j, gold_edge) in enumerate(gold_edges):
            pred_edge = all_pred_edges[i][:len(gold_edge)]
            for (k, edge) in enumerate(gold_edge):
                if edge == -100:
                    pred_edge[k] = -100

            if pred_edge == gold_edge:
                correct_edges += 1
                break


        for (j, (gold_node, gold_edge)) in enumerate(zip(gold_nodes, gold_edges)):
            is_correct_graph = False
            if gold_node == all_pred_nodes[i]:
                pred_edge = all_pred_edges[i][:len(gold_edge)]
                for (k, edge) in enumerate(gold_edge):
                    if edge == -100:
                        pred_edge[k] = -100

                if pred_edge == gold_edge:
                    correct_graphs += 1
                    is_correct_graph = True
                    break

            if is_correct_graph:
                break

    print("Node accuracy = " + str(correct_nodes/len(all_gold_proofs)))
    print("Edge accuracy = " + str(correct_edges / len(all_gold_proofs)))
    print("Graph accuracy = " + str(correct_graphs / len(all_gold_proofs)))

