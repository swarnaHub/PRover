import json
import numpy as np
from nltk.tokenize import sent_tokenize
from pulp import *
import inference
import argparse


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

def filter_context(context):
    sents = sent_tokenize(context)
    for sent in sents:
        if sent.count(".") > 1:
            print(context)
            return True

    return False

def get_fact_rule_identifiers(data_dir, natlang_metadata):
    test_file = os.path.join(data_dir, "dev.jsonl")
    meta_test_file = os.path.join(data_dir, "meta-dev.jsonl")

    f1 = open(test_file, "r", encoding="utf-8-sig")
    f2 = open(meta_test_file, "r", encoding="utf-8-sig")

    meta_record_mappings = {}
    for line in f2:
        meta_record = json.loads(line)
        meta_record_mappings[meta_record["id"]] = meta_record

    natlang_mappings = get_natlang_mappings(natlang_metadata)
    fact_rule_identifiers = []
    for (i, record) in enumerate(f1):
        # print(i)
        record = json.loads(record)
        meta_record = meta_record_mappings[record["id"]]
        assert record["id"] == meta_record["id"]
        context = record["context"]
        if filter_context(context):
            continue
        for (j, question) in enumerate(record["questions"]):
            #if question["meta"]["QDep"] != 5:
            #    continue
            id = question["id"]
            if "NatLang" not in id:
                continue
            meta_data = meta_record["questions"]["Q" + str(j + 1)]

            question = question["text"]
            assert (question == meta_data["question"])

            natlang_mapping = natlang_mappings[id.split("-")[1]]
            new_sents = {}
            for rf_id, (sid, orig_sents) in natlang_mapping.items():
                if rf_id.startswith("triple"):
                    new_sents[sid] = "fact"
                else:
                    new_sents[sid] = "rule"

            fact_rule_identifier = []
            for i in range(len(new_sents)):
                if new_sents["sent"+str(i+1)] == "fact":
                    fact_rule_identifier.append(0)
                else:
                    fact_rule_identifier.append(1)

            fact_rule_identifier.append(0)  # NAF
            fact_rule_identifiers.append(fact_rule_identifier)

    return fact_rule_identifiers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--node_preds", default=None, type=str, required=True)
    parser.add_argument("--edge_logits", default=None, type=str, required=True)
    parser.add_argument("--natlang_metadata", default=None, type=str, required=True)
    parser.add_argument("--edge_preds", default=None, type=str, required=True)

    args = parser.parse_args()

    edge_logit_file = open(args.edge_logits, "r", encoding="utf-8-sig")
    edge_logits = edge_logit_file.read().splitlines()

    node_pred_file = open(args.node_preds, "r", encoding="utf-8-sig")
    node_preds = node_pred_file.read().splitlines()

    fact_rule_identifiers = get_fact_rule_identifiers(args.data_dir, args.natlang_metadata)

    print(len(fact_rule_identifiers))
    print(len(node_preds))
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

        edges = inference.solve_LP(edge_logit, fact_rule_identifiers[i], node_pred)

        print(edges)
        f.write(str(edges))
        f.write("\n")
    f.close()
