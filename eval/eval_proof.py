import argparse
import os
import json

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

        sentence_scramble = record["meta"]["sentenceScramble"]
        for (j, question) in enumerate(record["questions"]):
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            proofs = meta_data["proofs"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]
            if question["meta"]["QDep"] != 3:
                continue
            all_node_indices, all_edge_indices = get_node_edge_indices(proofs, sentence_scramble, nfact, nrule)
            gold_proofs.append((all_node_indices, all_edge_indices))

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
            pred_nodes = [int(x) for x in line[1:-1].split(",")]
            pred_nodes = [i for i, x in enumerate(pred_nodes) if x == 1]
            all_pred_nodes.append(pred_nodes)

    all_pred_edges = []
    with open(args.edge_pred_file, "r", encoding="utf-8-sig") as f:
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

    assert len(all_gold_proofs) == len(all_pred_nodes)
    assert len(all_gold_proofs) == len(all_pred_edges)

    print("Num samples = " + str(len(all_gold_proofs)))

    correct_nodes = 0
    correct_edges = 0
    correct_graphs = 0
    for (i, gold_proofs) in enumerate(all_gold_proofs):
        gold_nodes = gold_proofs[0]
        gold_edges = gold_proofs[1]
        pred_node = all_pred_nodes[i]

        for (j, gold_node) in enumerate(gold_nodes):
            if set(gold_node) == set(pred_node):
                correct_nodes += 1
                break

        for (j, gold_edge) in enumerate(gold_edges):
            pred_edge = all_pred_edges[i]

            if set(pred_edge) == set(gold_edge):
                correct_edges += 1
                break


        for (j, (gold_node, gold_edge)) in enumerate(zip(gold_nodes, gold_edges)):
            is_correct_graph = False
            if set(gold_node) == set(pred_node):
                pred_edge = all_pred_edges[i]

                if set(pred_edge) == set(gold_edge):
                    correct_graphs += 1
                    is_correct_graph = True
                    break

            if is_correct_graph:
                break

    print("Node accuracy = " + str(correct_nodes/len(all_gold_proofs)))
    print("Edge accuracy = " + str(correct_edges / len(all_gold_proofs)))
    print("Graph accuracy = " + str(correct_graphs / len(all_gold_proofs)))

