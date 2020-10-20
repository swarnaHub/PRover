import argparse
import os
import json
import sys
from graphviz import Digraph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        component_index_map["NAF"] = nfact + nrule

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

def get_index_component_maps(data_dir):
    test_file = os.path.join(data_dir, "test.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-test.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    index_component_maps = []
    for record, meta_record in zip(f1, f2):
        record = json.loads(record)
        meta_record = json.loads(meta_record)
        nfact = meta_record["NFact"]
        nrule = meta_record["NRule"]
        # if not record["id"].startswith("AttPosBirds"):
        #     continue

        sentence_scramble = record["meta"]["sentenceScramble"]

        index_component_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index - nfact)
            index_component_map[i] = component
        index_component_map[nfact + nrule] = "NAF"

        for (j, question) in enumerate(record["questions"]):
            #if question["meta"]["QDep"] != 5:
            #    continue
            index_component_maps.append(index_component_map)

    return index_component_maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--node_pred_file", default=None, type=str, required=True)
    parser.add_argument("--edge_pred_file", default=None, type=str, required=True)
    parser.add_argument("--graph_path", default=None, type=str, required=True)

    args = parser.parse_args()
    index_component_maps = get_index_component_maps(args.data_dir)

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

    assert len(all_pred_nodes) == len(all_pred_edges)

    print("Num samples = " + str(len(all_pred_nodes)))

    for (i, (index_component_map, pred_nodes, pred_edges)) in enumerate(zip(index_component_maps, all_pred_nodes, all_pred_edges)):
        # Uncomment this to print the proof for a particular example by specifying the index
        # if i != index:
        #     break
        g = Digraph()
        print(pred_nodes)
        print(pred_edges)
        for pred_node in pred_nodes:
            g.node(index_component_map[pred_node])

        for pred_edge in pred_edges:
            g.edge(index_component_map[pred_edge[0]], index_component_map[pred_edge[1]])

        g.render(os.path.join(args.graph_path, "graph_" + str(i)), view=False)



