import argparse
import os
import json
from collections import OrderedDict
from nltk import sent_tokenize
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_utils import get_proof_graph, get_proof_graph_with_fail

def filter_context(context):
    sents = sent_tokenize(context)
    for sent in sents:
        if sent.count(".") > 1:
            print(context)
            return True

    return False

def get_natlang_mappings(natlang_metadata):
    natlang_mappings = OrderedDict()
    with open(natlang_metadata, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
        for (i, line) in enumerate(lines):
            if i == 0 or line == "":
                continue

            line = line.split("\t")
            id = line[0].split("-")[0].replace("AttNoneg", "")
            if id not in natlang_mappings:
                natlang_mappings[id] = {}
            for rf_id in line[6].split(" "):
                if rf_id == "":
                    continue
                natlang_mappings[id][rf_id] = (line[1], line[4])

    return natlang_mappings

def get_node_edge_indices(proofs, natlang_mapping):
    all_node_indices, all_edge_indices = [], []
    for proof in proofs.split("OR"):
        node_indices = []
        edge_indices = []

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)


        for node in nodes:
            sent_id = int(natlang_mapping[node][0].replace("sent", "")) - 1
            node_indices.append(sent_id)

        edges = list(set(edges))
        for edge in edges:
            start_index = int(natlang_mapping[edge[0]][0].replace("sent", "")) - 1
            end_index = int(natlang_mapping[edge[1]][0].replace("sent", "")) - 1
            edge_indices.append((start_index, end_index))

        all_node_indices.append(node_indices)
        all_edge_indices.append(edge_indices)

    return all_node_indices, all_edge_indices

def get_gold_proof_nodes_edges(data_dir, natlang_metadata):
    natlang_mappings = get_natlang_mappings(natlang_metadata)

    test_file = os.path.join(data_dir, "dev.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-dev.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    meta_record_mappings = {}
    for line in f2:
        meta_record = json.loads(line)
        meta_record_mappings[meta_record["id"]] = meta_record

    gold_proofs = []
    gold_labels = []
    for (i, record) in enumerate(f1):
        record = json.loads(record)
        meta_record = meta_record_mappings[record["id"]]

        context = record["context"]
        if filter_context(context):
            continue
        for (j, question) in enumerate(record["questions"]):
            # Uncomment to test at certain depth
            #if question["meta"]["QDep"] != 5:
            #    continue
            id = question["id"]
            # Testing only for NatLang samples
            if "NatLang" not in id:
                continue
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            proofs = meta_data["proofs"]
            label = question["label"]
            natlang_mapping = natlang_mappings[id.split("-")[1]]

            all_node_indices, all_edge_indices = get_node_edge_indices(proofs, natlang_mapping)
            gold_proofs.append((all_node_indices, all_edge_indices))
            gold_labels.append(label)

    return gold_proofs, gold_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--qa_pred_file", default=None, type=str, required=True)
    parser.add_argument("--node_pred_file", default=None, type=str, required=True)
    parser.add_argument("--edge_pred_file", default=None, type=str, required=True)
    parser.add_argument("--natlang_metadata", default=None, type=str, required=True)

    args = parser.parse_args()
    with open(args.qa_pred_file, "r", encoding="utf-8-sig") as f:
        all_pred_labels = f.read().splitlines()
    all_gold_proofs, all_gold_labels = get_gold_proof_nodes_edges(args.data_dir, args.natlang_metadata)

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
    assert len(all_gold_proofs) == len(all_gold_labels)
    assert len(all_gold_labels) == len(all_pred_labels)

    print("Num samples = " + str(len(all_gold_proofs)))

    correct_qa = 0
    correct_nodes = 0
    correct_edges = 0
    correct_proofs = 0
    correct_samples = 0

    for (i, gold_proofs) in enumerate(all_gold_proofs):
        is_correct_qa = False
        if str(all_gold_labels[i]) == all_pred_labels[i]:
            is_correct_qa = True
            correct_qa += 1
        
        gold_nodes = gold_proofs[0]
        gold_edges = gold_proofs[1]
        pred_node = all_pred_nodes[i]
        pred_edge = all_pred_edges[i]

        for (j, gold_node) in enumerate(gold_nodes):
            if set(gold_node) == set(pred_node):
                correct_nodes += 1
                break

        for (j, gold_edge) in enumerate(gold_edges):
            if set(pred_edge) == set(gold_edge):
                correct_edges += 1
                break

        is_correct_proof = False
        for (j, (gold_node, gold_edge)) in enumerate(zip(gold_nodes, gold_edges)):
            if set(gold_node) == set(pred_node) and set(pred_edge) == set(gold_edge):
                correct_proofs += 1
                is_correct_proof = True
                break

        if is_correct_proof and is_correct_qa:
            correct_samples += 1

    print("QA accuracy = " + str(correct_qa/len(all_gold_labels)))
    print("Node accuracy = " + str(correct_nodes/len(all_gold_proofs)))
    print("Edge accuracy = " + str(correct_edges / len(all_gold_proofs)))
    print("Proof accuracy = " + str(correct_proofs / len(all_gold_proofs)))
    print("Full accuracy = " + str(correct_samples / len(all_gold_proofs)))