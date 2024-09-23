import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

from transformers import DistilBertModel, DistilBertConfig

# Sample data
input_data = torch.randn(3, 15, 1024)

input_distilbert = torch.randn(3, 15, 768)

# Initialize DistilBert model
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Feed data to DistilBert
output = distilbert(inputs_embeds=input_distilbert)

print(output)
