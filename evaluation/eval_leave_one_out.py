import json

def get_noneg_data():
    test = open("../data/depth-5/test.jsonl", "r", encoding="utf-8-sig")
    meta_test = open("../data/depth-5/meta-test.jsonl", "r", encoding="utf-8-sig")
    question_label_map = {}

    for (line, meta_line) in zip(test, meta_test):
        record = json.loads(line)
        meta_record = json.loads(meta_line)
        record_id = record["id"]
        if "Noneg" in record_id:
            for (j, question) in enumerate(record["questions"]):
                qid = question["id"]
                label = question["label"]
                meta_data = meta_record["questions"]["Q" + str(j + 1)]

                assert (question["text"] == meta_data["question"])

                proofs = meta_data["proofs"]
                if "CWA" in proofs:
                    continue
                question_label_map[record_id + "_" + qid] = label

    return question_label_map

def get_leave_one_out_preds():
    leave_one_out_data = open("../data/depth-5/leave-one-out.jsonl", "r", encoding="utf-8-sig")
    preds = open("../output/best_model/predictions_dev.lst", "r", encoding="utf-8-sig")

    question_pred_map = {}
    for leave_one_out_line, pred_line in zip(leave_one_out_data, preds):
        record = json.loads(leave_one_out_line)
        record_id = record["id"].split("_")
        qid = record_id[0] + "_" + record_id[1]
        if qid not in question_pred_map:
            question_pred_map[qid] = []
        if pred_line.strip() == "True":
            question_pred_map[qid].append((record_id[2], True))
        else:
            question_pred_map[qid].append((record_id[2], False))

    return question_pred_map

if __name__ == '__main__':
    noneg_question_label_map = get_noneg_data()
    question_pred_map = get_leave_one_out_preds()

    print(len(noneg_question_label_map))
    print(len(question_pred_map))

    critical_correct = 0
    macro_precision = 0
    macro_recall = 0
    for qid, gold_label in noneg_question_label_map.items():
        preds = question_pred_map[qid]
        is_full_correct = True
        critical_correct_sample = 0
        critical_gold = 0
        critical_pred = 0
        for cid, pred_label in preds:
            if cid[0] == "c":
                critical_gold += 1
            if pred_label != gold_label:
                critical_pred += 1
            if cid[0] == "c" and pred_label != gold_label:
                critical_correct_sample += 1
            if not ((cid[0] == "c" and pred_label != gold_label) or (cid[0] == "i" and pred_label == gold_label)):
                is_full_correct = False
                break
        if is_full_correct:
            critical_correct += 1
        if critical_pred > 0:
            macro_precision += critical_correct_sample/critical_pred
        else:
            macro_precision += 1.0
        if critical_gold > 0:
            macro_recall += critical_correct_sample/critical_gold
        else:
            macro_recall += 1.0

    macro_precision /= len(question_pred_map)
    macro_recall /= len(question_pred_map)

    print("Accuracy = " + str(critical_correct/len(question_pred_map)))
    print("Precision = " + str(macro_precision))
    print("Recall = " + str(macro_recall))
    print("F1 = " + str(2*macro_precision*macro_recall/(macro_precision+macro_recall)))
