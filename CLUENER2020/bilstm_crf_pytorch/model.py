from torch.nn import LayerNorm
import torch.nn as nn
from crf import CRF

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

# object instantialize:
# model = NERModel(vocab_size=len(processor.vocab), embedding_size=args.embedding_size,
#                  hidden_size=args.hidden_size,device=args.device,label2id=args.label2id)

# embedding_size = 128, hidden_size = 384, label2id is a dictionary
# embedding_size是词向量的维度
# hidden_size是LSTM中隐层的维度
class NERModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(NERModel, self).__init__()
        self.embedding_size = embedding_size
        # nn.Embedding is a simple lookup table that stores embeddings of a fixed dictionary and size.
        # This module is often used to store word embeddings and retrieve them using indices.
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # nn.LSTM中，
        # input_size：输入数据的特征维度，通常就是词向量的维度
        # hidden_size：LSTM中隐层的维度
        # batch_first=True确保输入的数据维度顺序为(batch_size, seq_length, embedding_dim)
        # num_layers：循环神经网络的层数
        # dropout：用dropout的比例
        # bidirectional=True表示使用双向LSTM
        self.bilstm = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,
                              batch_first=True,num_layers=2,dropout=drop_p,
                              bidirectional=True)
        # 
        self.dropout = SpatialDropout(drop_p)
        # LayerNorm applies Layer Normalization over a mini-batch of inputs
        self.layer_norm = LayerNorm(hidden_size * 2)
        # nn.Linear作为self.classifier，输入向量长度hidden_size * 2，输出向量长度len(label2id)，即所有类别的个数
        self.classifier = nn.Linear(hidden_size * 2,len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device)

    def forward(self, inputs_ids, input_mask):
        # 词到inputs_ids的嵌入
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        # bilstm层
        seqence_output, _ = self.bilstm(embs)
        # 归一化层
        seqence_output= self.layer_norm(seqence_output)
        # 分类层
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features