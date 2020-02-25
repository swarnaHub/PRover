from pytorch_transformers import BertPreTrainedModel, RobertaConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, RobertaModel
from pytorch_transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss
import torch

class RobertaForRR(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForRR, self).__init__(config)

        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_sequence = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, proof_offset=None, proof_label=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        max_seq_length = proof_label.shape[1]
        batch_size = proof_label.shape[0]
        embedding_dim = sequence_output.shape[2]

        batch_output = torch.zeros((batch_size, max_seq_length, embedding_dim)).to("cuda")
        for batch_index in range(sequence_output.shape[0]):
            prev_index = 0
            sample_output = None
            count = 0
            for offset in proof_offset[batch_index]:
                if offset == 0:
                    break
                else:
                    sentence_output = torch.mean(sequence_output[batch_index, prev_index:(offset+1), :], dim=0).unsqueeze(0)
                    prev_index = offset+1
                    count += 1
                    if sample_output is None:
                        sample_output = sentence_output
                    else:
                        sample_output = torch.cat((sample_output, sentence_output), dim=0)
            sample_output = torch.cat((sample_output, torch.zeros((max_seq_length-count, embedding_dim)).to("cuda")), dim=0)
            batch_output[batch_index, :, :] = sample_output

        sequence_logits = self.classifier_sequence(batch_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            cr_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            seq_loss = loss_fct(sequence_logits.view(-1, self.num_labels), proof_label.view(-1))
            total_loss = cr_loss + seq_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)




class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.classifier_new = RobertaClassificationHead(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        # flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask)
        sequence_output = outputs[0]

        logits = self.classifier_new(sequence_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
