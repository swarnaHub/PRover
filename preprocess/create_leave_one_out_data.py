import json
from proof_utils import get_proof_graph
from nltk import sent_tokenize

def is_node_in_all_proofs(all_proof_nodes, node):
    for proof_nodes in all_proof_nodes:
        if node not in proof_nodes:
            return False

    return True

if __name__ == '__main__':
    test = open("../data/depth-5/test.jsonl", "r", encoding="utf-8-sig")
    meta_test = open("../data/depth-5/meta-test.jsonl", "r", encoding="utf-8-sig")
    leave_one_out_test = open("../data/depth-5/leave-one-out.jsonl", "w", encoding="utf-8-sig")

    all_samples = 0
    for (record, meta_record) in zip(test, meta_test):
        record = json.loads(record)
        meta_record = json.loads(meta_record)

        assert record["id"] == meta_record["id"]
        record_id = record["id"]
        if "Noneg" in record["id"]:
            context = record["context"]
            context_sents = sent_tokenize(context)

            sentence_scramble = record["meta"]["sentenceScramble"]
            nfact = meta_record["NFact"]
            nrule = meta_record["NRule"]

            index_component_map = {}
            for (i, index) in enumerate(sentence_scramble):
                if index <= nfact:
                    component = "triple" + str(index)
                else:
                    component = "rule" + str(index - nfact)
                index_component_map[i] = component

            for (j, question) in enumerate(record["questions"]):
                qid = question["id"]
                label = question["label"]
                meta_data = meta_record["questions"]["Q" + str(j + 1)]
                proofs = meta_data["proofs"]
                if "CWA" in proofs:
                    continue
                question = question["text"]
                all_samples += len(sentence_scramble)

                assert len(sentence_scramble) == len(context_sents)

                #print(proofs)
                proofs = proofs.split("OR")
                #print(len(proofs))

                all_proof_nodes = []
                for proof in proofs:
                    nodes, _ = get_proof_graph(proof)
                    all_proof_nodes.append(nodes)

                critical_count = 0
                irrelevant_count = 0
                for k in range(len(sentence_scramble)):
                    curr_sentence = context_sents[k]
                    new_context = context.replace(curr_sentence, "")
                    if is_node_in_all_proofs(all_proof_nodes, index_component_map[k]):
                        critical_count += 1
                        new_sample = {
                            "id": record_id + "_" + qid + "_" + "c" + str(critical_count),
                            "context": new_context,
                            "question": question,
                            "label": not label
                        }
                    else:
                        irrelevant_count += 1
                        new_sample = {
                            "id": record_id + "_" + qid + "_" + "i" + str(irrelevant_count),
                            "context": new_context,
                            "question": question,
                            "label": label
                        }

                    leave_one_out_test.write(json.dumps(new_sample))
                    leave_one_out_test.write("\n")

    print(all_samples)