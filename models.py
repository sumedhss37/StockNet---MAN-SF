import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last

class attn(nn.Module):
    def __init__(self,in_shape, out_shape ):
        super(attn, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape ,out_shape)
        self.V = nn.Linear(in_shape,1)
    def forward(self, full, last):
        score = self.V(torch.tanh(self.W1(last) + self.W2(full)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * full
        context_vector =torch.sum(context_vector, dim=1) 
        return context_vector


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha, stock_num):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.grup = nn.ModuleList([gru(3, 64) for _ in range(stock_num)])
        self.attnp = nn.ModuleList([attn(64, 64) for _ in range(stock_num)])
        self.tweet_gru = nn.ModuleList([gru(512, 64) for _ in range(stock_num)])
        self.attn_tweet = nn.ModuleList([attn(64, 64) for _ in range(stock_num)])
        self.grut = nn.ModuleList([gru(64, 64) for _ in range(stock_num)])
        self.attnt = nn.ModuleList([attn(64, 64) for _ in range(stock_num)])
        self.bilinear = nn.ModuleList([nn.Bilinear(64, 64, 64) for _ in range(stock_num)])
        self.layer_normt = nn.ModuleList([nn.LayerNorm((64,)) for _ in range(stock_num)])
        self.layer_normp = nn.ModuleList([nn.LayerNorm((64,)) for _ in range(stock_num)])
        self.linear_x = nn.ModuleList([nn.Linear(64, 2) for _ in range(stock_num)])

        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, text_input, price_input, adj):
        num_tw = text_input.size(2)     # num of tweets
        num_d = price_input.size(1)     # num of days
        pr_ft = price_input.size(2)     # price features (3)
        num_stocks = price_input.size(0)

        li = []

        for i in range(num_stocks):
            ### Price encoder
            price_feat = price_input[i].unsqueeze(0)  # (1, num_d, 3)
            price_full, price_last = self.grup[i](price_feat)
            price_vec = self.attnp[i](price_full, price_last).reshape(1, 64)
            price_vec = self.layer_normp[i](price_vec)

            ### Tweet encoders per day
            han_li1 = []
            for j in range(num_d):
                tweet_day = text_input[i, j]  # (8, 512)
                tweet_day = tweet_day.unsqueeze(0)  # (1, 8, 512)
                tweet_full, tweet_last = self.tweet_gru[i](tweet_day)
                tweet_vec = self.attn_tweet[i](tweet_full, tweet_last).reshape(1, 64)
                han_li1.append(tweet_vec)

            news_vector = torch.stack(han_li1, dim=1)  # (1, num_d, 64)

            ### Sequential Tweet Encoder
            tweet_full, tweet_last = self.grut[i](news_vector)
            tweet_vec = self.attnt[i](tweet_full, tweet_last).reshape(1, 64)
            tweet_vec = self.layer_normt[i](tweet_vec)

            ### Combine with Bilinear
            combined = torch.tanh(self.bilinear[i](tweet_vec, price_vec))
            li.append(combined)  # (1, 64)

        ft_vec = torch.cat(li, dim=0)  # (num_stocks, 64)

        ### Feedforward + GAT
        out_1 = torch.stack([torch.tanh(self.linear_x[i](ft_vec[i])) for i in range(num_stocks)], dim=0)

        x = F.dropout(ft_vec, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return x + out_1

    """
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    """


# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [SpGraphAttentionLayer(nfeat, 
#                                                  nhid, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = SpGraphAttentionLayer(nhid * nheads, 
#                                              nclass, 
#                                              dropout=dropout, 
#                                              alpha=alpha, 
#                                              concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)

