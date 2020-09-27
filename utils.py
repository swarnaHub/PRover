from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

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

class RRInputExampleQA(object):
    def __init__(self, id, context, question, label):
        self.id = id
        self.context = context
        self.question = question
        self.label = label

class RRInputExample(object):
    def __init__(self, id, context, question, node_label, edge_label, label):
        self.id = id
        self.context = context
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

class RRFeaturesQA(object):
    def __init__(self, id, input_ids, input_mask, segment_ids, label_id):
        self.id = id
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

class RRProcessorQA(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-train.jsonl")))

    def get_dev_examples(self, data_dir):
        # Change these to test paths for test results
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-dev.jsonl")))
        '''
        return self._create_examples_leave_one_out(
            self._read_jsonl(os.path.join(data_dir, "leave-one-out.jsonl"))
        )
        '''

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-test.jsonl")))

    def get_labels(self):
        return [True, False]

    def _create_examples(self, records, meta_records):
        examples = []
        for (i, (record, meta_record)) in enumerate(zip(records, meta_records)):
            print(i)
            assert record["id"] == meta_record["id"]
            context = record["context"]
            for (j, question) in enumerate(record["questions"]):
                id = question["id"]
                label = question["label"]
                meta_data = meta_record["questions"]["Q" + str(j + 1)]
                proofs = meta_data["proofs"]
                #if "CWA" in proofs:
                #    continue
                #if question["meta"]["QDep"] != 3:
                #    continue
                question = question["text"]

                examples.append(RRInputExampleQA(id, context, question, label))

        return examples

    def _create_examples_leave_one_out(self, records):
        examples = []
        for record in records:
            examples.append(RRInputExampleQA(record["id"], record["context"], record["question"], record["label"]))

        return examples


class RRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-train.jsonl")))

    def get_dev_examples(self, data_dir):
        # Change these to test paths for test results
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-test.jsonl")))

    def get_labels(self):
        return [True, False]

    # Unconstrained training, use this for ablation
    def _get_node_edge_label_unconstrained(self, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]
        #print(proof)
        node_label = [0] * (nfact + nrule + 1)
        edge_label = np.zeros((nfact+nrule+1, nfact+nrule+1), dtype=int)

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)
        #print(nodes)
        #print(edges)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index-nfact)
            component_index_map[component] = i

        for node in nodes:
            if node != "NAF":
                index = component_index_map[node]
            else:
                index = nfact+nrule
            node_label[index] = 1

        edges = list(set(edges))
        for edge in edges:
            if edge[0] != "NAF":
                start_index = component_index_map[edge[0]]
            else:
                start_index = nfact+nrule
            if edge[1] != "NAF":
                end_index = component_index_map[edge[1]]
            else:
                end_index = nfact+nrule

            edge_label[start_index][end_index] = 1

        return node_label, list(edge_label.flatten())

    def _get_node_edge_label_constrained(self, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]
        #print(proof)
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


    def _create_examples(self, records, meta_records):
        examples = []
        for (i, (record, meta_record)) in enumerate(zip(records, meta_records)):
            #print(i)
            assert record["id"] == meta_record["id"]
            context = record["context"]
            sentence_scramble = record["meta"]["sentenceScramble"]
            for (j, question) in enumerate(record["questions"]):
                # Uncomment to train/evaluate at a certain depth
                #if question["meta"]["QDep"] != 5:
                #    continue
                # Uncomment to test at a specific subset of Birds-Electricity dataset
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
                node_label, edge_label = self._get_node_edge_label_constrained(proofs, sentence_scramble, nfact, nrule)

                examples.append(RRInputExample(id, context, question, node_label, edge_label, label))

        return examples

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
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        sentences = sent_tokenize(example.context)
        context_tokens = []
        proof_offset = []
        for sentence in sentences:
            sentence_tokens = tokenizer.tokenize(sentence)
            context_tokens.extend(sentence_tokens)
            proof_offset.append(len(context_tokens))
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

processors = {
    "rr": RRProcessor,
    "rr_qa": RRProcessorQA
}

output_modes = {
    "rr": "classification",
    "rr_qa": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "rr": 2,
    "rr_qa": 2
}
