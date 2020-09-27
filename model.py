from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class RobertaForRR(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRR, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None,
                head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (qa_loss,) + outputs

        return outputs  # qa_loss, logits, (hidden_states), (attentions)

class RobertaForRRWithNodeLoss(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRRWithNodeLoss, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier_node = NodeClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)
        logits = self.classifier(sequence_output)

        max_node_length = node_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to("cuda")
        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding, torch.zeros((max_node_length-count-1, embedding_dim)).to("cuda")), dim=0)
            batch_node_embedding[batch_index, :, :] = sample_node_embedding

        node_logits = self.classifier_node(batch_node_embedding)

        outputs = (logits, node_logits) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            node_loss = loss_fct(node_logits.view(-1, self.num_labels), node_label.view(-1))
            total_loss = qa_loss + node_loss
            outputs = (total_loss, qa_loss, node_loss) + outputs

        return outputs  # (total_loss), qa_loss, node_loss, logits, node_logits, (hidden_states), (attentions)

class NodeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(NodeClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EdgeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(EdgeClassificationHead, self).__init__()
        self.dense = nn.Linear(3*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForRRWithNodeEdgeLoss(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRRWithNodeEdgeLoss, self).__init__(config)

        self.num_labels = config.num_labels
        self.num_labels_edge = 2
        self.roberta = RobertaModel(config)
        self.naf_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_node = NodeClassificationHead(config)
        self.classifier_edge = EdgeClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, node_label=None,
                edge_label=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]
        naf_output = self.naf_layer(cls_output)
        logits = self.classifier(sequence_output)

        max_node_length = node_label.shape[1]
        max_edge_length = edge_label.shape[1]
        batch_size = node_label.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_node_embedding = torch.zeros((batch_size, max_node_length, embedding_dim)).to("cuda")
        batch_edge_embedding = torch.zeros((batch_size, max_edge_length, 3*embedding_dim)).to("cuda")

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat((sample_node_embedding, rf_embedding), dim=0)

            # Add the NAF output at the end
            sample_node_embedding = torch.cat((sample_node_embedding, naf_output[batch_index].unsqueeze(0)), dim=0)

            repeat1 = sample_node_embedding.unsqueeze(0).repeat(len(sample_node_embedding), 1, 1)
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(1, len(sample_node_embedding), 1)
            sample_edge_embedding = torch.cat((repeat1, repeat2, (repeat1-repeat2)), dim=2)

            sample_edge_embedding = sample_edge_embedding.view(-1, sample_edge_embedding.shape[-1])

            # Append 0s at the end (these will be ignored for loss)
            sample_node_embedding = torch.cat((sample_node_embedding,
                                               torch.zeros((max_node_length-count-1, embedding_dim)).to("cuda")), dim=0)
            sample_edge_embedding = torch.cat((sample_edge_embedding,
                                               torch.zeros((max_edge_length-len(sample_edge_embedding), 3*embedding_dim)).to("cuda")), dim=0)

            batch_node_embedding[batch_index, :, :] = sample_node_embedding
            batch_edge_embedding[batch_index, :, :] = sample_edge_embedding

        node_logits = self.classifier_node(batch_node_embedding)
        edge_logits = self.classifier_edge(batch_edge_embedding)

        outputs = (logits, node_logits, edge_logits) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            node_loss = loss_fct(node_logits.view(-1, self.num_labels), node_label.view(-1))
            edge_loss = loss_fct(edge_logits.view(-1, self.num_labels_edge), edge_label.view(-1))
            total_loss = qa_loss + node_loss + edge_loss
            outputs = (total_loss, qa_loss, node_loss, edge_loss) + outputs

        return outputs  # (total_loss), qa_loss, node_loss, edge_loss, logits, node_logits, edge_logits, (hidden_states), (attentions)