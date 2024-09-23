import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

from transformers import DistilBertModel, DistilBertConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        self.output_projection = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project inputs to query, key, and value
        Q = self.query_projection(query)  # shape: (batch_size, seq_len, d_model)
        K = self.key_projection(key)      # shape: (batch_size, seq_len, d_model)
        V = self.value_projection(value)  # shape: (batch_size, seq_len, d_model)
        
        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32).to(query.device))  # shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply attention weights to value projections
        attention_output = torch.matmul(attention_weights, V)  # shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape attention output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # shape: (batch_size, seq_len, d_model)
        
        # Apply output projection
        multihead_output = self.output_projection(attention_output)  # shape: (batch_size, seq_len, d_model)
        
        return multihead_output, attention_weights

class DummyImageEncoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(DummyImageEncoder, self).__init__()
        # resnet = models.resnet152(pretrained=True)
        # modules = list(resnet.children())[:-1]  # delete the last fc layer.
        # self.resnet = nn.Sequential(*modules)
        # self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(resnet.fc.in_features, momentum=0.01)

        swin = models.swin_t(pretrained=True)
        modules = list(swin.children())[:-1]
        self.swin = nn.Sequential(*modules)
        self.linear = nn.Linear(swin.norm.normalized_shape[0], embed_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(swin.norm.normalized_shape[0], momentum=0.01)


    def get_trainable_parameters(self):
        return list(self.bn.parameters()) + list(self.linear.parameters())

    # def load_resnet(self, resnet=None):
    #     if resnet is None:
    #         resnet = models.resnet152(pretrained=True)
    #         modules = list(resnet.children())[:-1]  # delete the last fc layer.
    #         self.resnet = nn.Sequential(*modules)
    #         self.resnet_in_features = resnet.fc.in_features
    #     else:
    #         self.resnet = resnet
    #     return
    
    def load_swin_transformer(self, swin_transformer=None):
        if swin_transformer is None:
            swin_transformer = models.swin_transformer(pretrained=True)
            modules = list(swin_transformer.children())[:-1]
            self.swin_transformer = nn.Sequential(*modules)
            self.swin_transformer_in_features = swin_transformer.fc.in_features
        else:
            self.swin_transformer = swin_transformer
        return


    def delete_swin_transformer(self):
        # resnet = self.resnet
        # self.resnet = None
        # return resnet
        swin = self.swin
        self.swin = None
        return swin
    
    def forward(self, image):
        with torch.no_grad():
            img_ft = self.swin(image)

        out = self.linear(self.bn(img_ft.reshape(img_ft.size(0), -1)))
        return out

    # def forward(self, image):
    #     with torch.no_grad():
    #         img_ft = self.swin(image)

    #     out = self.linear(self.bn(img_ft.reshape(img_ft.size(0), -1)))
    #     return out
    


class DistilBertEncoder(nn.Module):
    def __init__(self, embed_size, vocab_size, vocab_embed_size):
        super(DistilBertEncoder, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.out_linear = nn.Linear(self.distilbert.config.hidden_size, embed_size) 
        self.embed = nn.Embedding(vocab_size, vocab_embed_size)
        self.attention = MultiHeadAttention(embed_size, 8)
    
    def forward(self, input, lengths):
        input = self.embed(input)
        output = self.distilbert(inputs_embeds=input)
        output = self.out_linear(output.last_hidden_state)
        # [3,21,768] -> [3,768]
        output, _ = self.attention(output, output, output)
        output = output.mean(dim=1)
        
        return output

        
    def get_trainable_parameters(self):
        return list(self.parameters())

    

class DummyCaptionEncoder(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, embed_size):
        super(DummyCaptionEncoder, self).__init__()
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.rnn = nn.GRU(vocab_embed_size, embed_size)
        self.embed = nn.Embedding(vocab_size, vocab_embed_size)
        
    def forward(self, input, lengths):
        input = self.embed(input)
        lengths = torch.LongTensor(lengths)
        [_, sort_ids] = torch.sort(lengths, descending=True)
        sorted_input = input[sort_ids]
        sorted_length = lengths[sort_ids]
        reverse_sort_ids = sort_ids.clone()
        for i in range(sort_ids.size(0)):
            reverse_sort_ids[sort_ids[i]] = i
        packed = pack_padded_sequence(sorted_input, sorted_length, batch_first=True)
        output, _ = self.rnn(packed)
        padded, output_length = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = [padded[output_length[i]-1, i, :] for i in range(len(output_length))]
        output = torch.stack([output[reverse_sort_ids[i]] for i in range(len(output))], dim=0)
        output = self.out_linear(output)
        return output

    def get_trainable_parameters(self):
        return list(self.parameters())



#
# model = DummyCaptionEncoder(100, 64, 10)
#
# x1 = [
#     [45, 4, 7, 9, 2, 0, 0],
#     [11, 2, 3, 4, 5, 6, 7],
#     [99, 98, 97, 96, 7, 8, 0],
#     [89, 87, 86, 2, 0, 0, 0]
#     ]
# len1 = [5, 2, 3, 2]
# x1 = torch.tensor(x1)
# y1 = model(x1, len1)
#
# x2 = [
#     [56, 56, 3, 0, 0, 0, 0],
#     [89, 87, 86, 1, 0, 0, 0],
#     [1, 36, 4, 7, 8, 4, 0],
#     [99, 98, 97, 96, 4, 0, 0]
#     ]
# len2 = [2, 2, 5, 3]
# x2 = torch.tensor(x2)
# y2 = model(x2, len2)
#
# print('max dif 1', (y1[3,:] - y2[1,:]).max(), (y1[3,:] - y2[1,:]).min())
# print('max dif 2', (y1[2,:] - y2[3,:]).max(), (y1[2,:] - y2[3,:]).min())