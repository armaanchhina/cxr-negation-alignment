
import torch.nn as nn
from transformers import AutoModel
from torch.optim import AdamW

class ClinicalBERTClassifier(nn.Module):
    def __init__(self, num_classess=3, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classess)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)
        return logits
    
