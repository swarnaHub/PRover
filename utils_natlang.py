from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from collections import OrderedDict

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import json
from nltk.tokenize import sent_tokenize
import numpy as np

from proof_utils import get_proof_graph, get_proof_graph_with_fail

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class RRInputExample(object):
    def __init__(self, id, context, sent_offset, question, node_label, edge_label, label):
        self.id = id
        self.context = context
        self.sent_offset = sent_offset
        self.question = question
        self.node_label = node_label
        self.edge_label = edge_label
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class RRFeatures(object):
    def __init__(self, id, input_ids, input_mask, segment_ids, proof_offset, node_label, edge_label, label_id):
        self.id = id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.proof_offset = proof_offset
        self.node_label = node_label
        self.edge_label = edge_label
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records

class RRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        natlang_mappings = self._get_natlang_mappings(os.path.join(data_dir, "turk-questions-train-mappings.tsv"))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-train.jsonl")),
            natlang_mappings)

    def get_dev_examples(self, data_dir):
        natlang_mappings = self._get_natlang_mappings(os.path.join(data_dir, "turk-questions-test-mappings.tsv"))
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-test.jsonl")),
            natlang_mappings)

    def get_labels(self):
        return [True, False]

    def _get_node_edge_label_no_natlang(self, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]
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
        component_index_map["NAF"] = nfact+nrule

        for node in nodes:
            index = component_index_map[node]
            node_label[index] = 1

        edges = list(set(edges))
        for edge in edges:
            start_index = component_index_map[edge[0]]
            end_index = component_index_map[edge[1]]
            edge_label[start_index][end_index] = 1

        # Mask impossible edges
        for i in range(len(edge_label)):
            for j in range(len(edge_label)):
                # Ignore diagonal
                if i == j:
                    edge_label[i][j] = -100
                    continue

                # Ignore edges between non-nodes
                if node_label[i] == 0 or node_label[j] == 0:
                    edge_label[i][j] = -100
                    continue

                is_fact_start = False
                is_fact_end = False
                if i == len(edge_label)-1 or sentence_scramble[i] <= nfact:
                    is_fact_start = True
                if j == len(edge_label)-1 or sentence_scramble[j] <= nfact:
                    is_fact_end = True

                # No edge between fact/NAF -> fact/NAF
                if is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

                # No edge between Rule -> fact/NAF
                if not is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

        return node_label, list(edge_label.flatten())

    def _get_natlang_mappings(self, natlang_metadata):
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

    def _get_node_edge_label_natlang(self, id, proofs, natlang_mappings):
        natlang_mapping = natlang_mappings[id.split("-")[1]]
        new_sents = {}
        for rf_id, (sid, orig_sents) in natlang_mapping.items():
            if rf_id.startswith("triple"):
                new_sents[sid] = "fact"
            else:
                new_sents[sid] = "rule"

        proof = proofs.split("OR")[0]
        #print(proof)
        node_label = [0] * (len(new_sents) + 1)
        edge_label = np.zeros((len(new_sents) + 1, len(new_sents) + 1), dtype=int)

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)

        #print(edges)

        for node in nodes:
            sent_id = int(natlang_mapping[node][0].replace("sent", "")) - 1
            node_label[sent_id] = 1

        for edge in edges:
            start_sent_id = int(natlang_mapping[edge[0]][0].replace("sent", "")) - 1
            end_sent_id = int(natlang_mapping[edge[1]][0].replace("sent", "")) - 1
            edge_label[start_sent_id][end_sent_id] = 1

        # Edge masking
        for i in range(len(edge_label)):
            for j in range(len(edge_label)):
                # Ignore diagonal
                if i == j:
                    edge_label[i][j] = -100
                    continue

                # Ignore edges between non-nodes
                if node_label[i] == 0 or node_label[j] == 0:
                    edge_label[i][j] = -100
                    continue

                # Ignore edges between Fact -> Fact and Rule -> Fact
                is_fact_start = False
                is_fact_end = False

                if i == len(edge_label)-1 or new_sents["sent" + str(i+1)] == "fact":
                    is_fact_start = True
                if j == len(edge_label)-1 or new_sents["sent" + str(j+1)] == "fact":
                    is_fact_end = True

                # No edge between fact/NAF -> fact/NAF
                if is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

                # No edge between Rule -> fact/NAF
                if not is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

        return node_label, list(edge_label.flatten())

    def _create_examples(self, records, meta_records, natlang_mappings):
        meta_record_mappings = {}
        for meta_record in meta_records:
            meta_record_mappings[meta_record["id"]] = meta_record

        examples = []
        for (i, record) in enumerate(records):
            #print(i)
            meta_record = meta_record_mappings[record["id"]]
            assert record["id"] == meta_record["id"]
            context = record["context"]
            sentence_scramble = record["meta"]["sentenceScramble"]
            for (j, question) in enumerate(record["questions"]):
                # Uncomment to train/evaluate at a certain depth
                #if question["meta"]["QDep"] != 4:
                #    continue
                #if not record["id"].startswith("AttPosElectricityRB4"):
                #    continue
                id = question["id"]
                label = question["label"]
                question = question["text"]
                meta_data = meta_record["questions"]["Q"+str(j+1)]

                assert (question == meta_data["question"])

                proofs = meta_data["proofs"]
                nfact = meta_record["NFact"]
                nrule = meta_record["NRule"]
                if "NatLang" not in id:
                    continue
                    node_label, edge_label = self._get_node_edge_label_no_natlang(proofs, sentence_scramble, nfact, nrule)
                    sent_offset = [k for k in range(len(node_label)-1)]
                else:
                    node_label, edge_label = self._get_node_edge_label_natlang(id, proofs, natlang_mappings)
                    natlang_mapping = natlang_mappings[id.split("-")[1]]
                    sent_offset = []
                    sid_sents_map = {}
                    for rf_id, (sid, orig_sents) in natlang_mapping.items():
                        sid_sents_map[sid] = orig_sents

                    cumulative = 0
                    for k in range(len(sid_sents_map)):
                        sents = sid_sents_map["sent" + str(k+1)]
                        sent_offset.append(cumulative + len(sent_tokenize(sents))-1)
                        cumulative += len(sent_tokenize(sents))

                assert len(sent_offset) == len(node_label)-1
                examples.append(RRInputExample(id, context, sent_offset, question, node_label, edge_label, label))

        return examples

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 output_mode,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        special_tokens_count = 3 if sep_token_extra else 2
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count - 1)
        else:
            # Account for [CLS] and [SEP] with "- 2" or "-3" for RoBERTa
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def convert_examples_to_features_RR(examples,
                                 label_list,
                                 max_seq_length,
                                 max_node_length,
                                 max_edge_length,
                                 tokenizer,
                                 output_mode,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    max_size = 0
    error_tokenization = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        sentences = sent_tokenize(example.context)
        context_tokens = []
        proof_offset = []
        sent_offset = example.sent_offset
        curr_offset = 0
        is_error = False
        for offset in sent_offset:
            sentence_tokens = tokenizer.tokenize(' '.join(sentences[curr_offset:(offset+1)]))
            if len(sentence_tokens) == 0:
                is_error = True
                error_tokenization += 1
                break
            #assert len(sentence_tokens) > 0
            context_tokens.extend(sentence_tokens)
            proof_offset.append(len(context_tokens))
            curr_offset = offset+1
            #proof_offset.append(len(sentence_tokens)) # Uncomment this for the efficient model

        if is_error:
            print(example.context)
            continue
        max_size = max(max_size, len(context_tokens))

        question_tokens = tokenizer.tokenize(example.question)

        special_tokens_count = 3 if sep_token_extra else 2
        _truncate_seq_pair(context_tokens, question_tokens, max_seq_length - special_tokens_count - 1)

        tokens = context_tokens + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens += question_tokens + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(question_tokens) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        proof_offset = proof_offset + [0] * (max_node_length - len(proof_offset))
        node_label = example.node_label
        node_label = node_label + [-100] * (max_node_length - len(node_label))

        edge_label = example.edge_label
        edge_label = edge_label + [-100] * (max_edge_length - len(edge_label))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(proof_offset) == max_node_length
        assert len(node_label) == max_node_length
        assert len(edge_label) == max_edge_length

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s" % (example.id))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("proof_offset: %s" % " ".join([str(x) for x in proof_offset]))
            logger.info("node_label: %s" % " ".join([str(x) for x in node_label]))
            logger.info("edge_label: %s" % " ".join([str(x) for x in edge_label]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            RRFeatures(id=id,
                       input_ids=input_ids,
                       input_mask=input_mask,
                       segment_ids=segment_ids,
                       proof_offset=proof_offset,
                       node_label=node_label,
                       edge_label=edge_label,
                       label_id=label_id))

        #if ex_index > 1000:
        #    break

    print(error_tokenization)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        max_len = max(len(tokens_a), len(tokens_b), len(tokens_c))
        if max_len == len(tokens_a):
            tokens_a.pop()
        elif max_len == len(tokens_b):
            tokens_b.pop()
        else:
            tokens_c.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "rr" or task_name == "rr_qa":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

def compute_graph_metrics(task_name, node_preds, out_node_label_ids, edge_preds, out_edge_label_ids):
    assert len(node_preds) == len(out_node_label_ids)
    assert len(edge_preds) == len(out_edge_label_ids)
    assert len(node_preds) == len(edge_preds)
    correct_node, correct_edge, correct_graph = 0, 0, 0
    for i in range(len(out_node_label_ids)):
        for j in range(len(out_node_label_ids[i])):
            if out_node_label_ids[i][j] == -100: # Ignore index, so copy it
                node_preds[i][j] = -100
                continue

        for j in range(len(out_edge_label_ids[i])):
            if out_edge_label_ids[i][j] == -100:
                edge_preds[i][j] = -100

        # If they match exactly, then it's fully correct
        if np.array_equal(out_node_label_ids[i], node_preds[i]):
            correct_node += 1

        if np.array_equal(out_edge_label_ids[i], edge_preds[i]):
            correct_edge += 1

        if np.array_equal(out_node_label_ids[i], node_preds[i]) and np.array_equal(out_edge_label_ids[i], edge_preds[i]):
            correct_graph += 1

    return {"node_acc": correct_node/len(node_preds),
            "edge_acc": correct_edge/len(edge_preds),
            "graph_acc": correct_graph/len(edge_preds)}

def compute_graph_metrics_node_based(task_name, node_preds, out_node_label_ids, edge_preds, out_edge_label_ids):
    assert len(node_preds) == len(out_node_label_ids)
    assert len(edge_preds) == len(out_edge_label_ids)
    assert len(node_preds) == len(edge_preds)
    correct_node, correct_edge, correct_graph = 0, 0, 0
    for i in range(len(out_node_label_ids)):
        node_pred_ids = []
        count_valid_nodes = 0
        for j in range(len(out_node_label_ids[i])):
            if out_node_label_ids[i][j] == -100: # Ignore index, so copy it
                node_preds[i][j] = -100
                continue
            else:
                count_valid_nodes += 1
                if node_preds[i][j] == 1:
                    node_pred_ids.append(j)

        invalid_edges_start = count_valid_nodes*count_valid_nodes
        for j in range(len(out_edge_label_ids[i])):
            row = int(j/count_valid_nodes)
            col = j % count_valid_nodes
            # Invalid cases
            if j >= invalid_edges_start:
                edge_preds[i][j] = -100
            # Lower triangle
            elif row >= col:
                edge_preds[i][j] = -100
            # Edges to nodes not part of node prediction, so can be ignored
            elif row not in node_pred_ids or col not in node_pred_ids:
                edge_preds[i][j] = out_edge_label_ids[i][j]

        # If they match exactly, then it's fully correct
        if np.array_equal(out_node_label_ids[i], node_preds[i]):
            correct_node += 1

        if np.array_equal(out_edge_label_ids[i], edge_preds[i]):
            correct_edge += 1

        if np.array_equal(out_node_label_ids[i], node_preds[i]) and np.array_equal(out_edge_label_ids[i], edge_preds[i]):
            correct_graph += 1

    return {"node_acc": correct_node/len(node_preds),
            "edge_acc": correct_edge/len(edge_preds),
            "graph_acc": correct_edge/len(edge_preds)}


processors = {
    "rr": RRProcessor
}

output_modes = {
    "rr": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "rr": 2
}