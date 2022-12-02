from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.utils import ModelOutput


@dataclass
class Output(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None

class BertClassifier(BertPreTrainedModel):
    def __init__(self, config=BertConfig):
        super(BertClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, every_loss=False):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if every_loss:
                loss_fct = CrossEntropyLoss(reduction='none')
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return Output(loss=loss, logits=logits)
