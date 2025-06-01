import math
import torch
from torch import nn
import dgl
from model.model_layers import CompGCNCov
import torch.nn.functional as F



class KGGCN(nn.Module):  # GCN表示学习
    def __init__(self, num_ent, num_rel, gcn_layers, init_dim, gcn_dim, edge_type, edge_norm,
                 gcn_drop=0., opn='mult', act=None):
        super(KGGCN, self).__init__()
        # self.act = torch.tanh
        # self.act = F.gelu
        # self.act = torch.relu
        # self.act = torch.prelu
        # self.loss = nn.BCELoss()
        self.act = act
        self.num_ent, self.num_rel = num_ent, num_rel
        self.init_dim, self.gcn_dim = init_dim, gcn_dim
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]

        self.init_embed = self.get_param([self.num_ent, self.init_dim])  # initial embedding for entities
        self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])  # relation

        layers = []
        for i in range(gcn_layers):
            if i == 0:
                layer = CompGCNCov(self.init_dim, self.gcn_dim, self.act, gcn_drop, opn)
            else:
                layer = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, gcn_drop, opn)
            layers.append(layer)
        self.gcn_layers = nn.ModuleList(layers)
        self.drop = torch.nn.Dropout(self.gcn_drop)


    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    # def calc_loss(self, pred, label):
    #     return self.loss(pred, label)

    def forward(self, g, subj, rel):
        x, r = self.init_embed, self.init_rel  # embedding of relations
        for layer in self.gcn_layers:
            x, r = layer(g, x, r, self.edge_type, self.edge_norm)
            x = self.drop(x)
            r = self.drop(r)
        sub_emb = torch.index_select(x, 0, subj)  # filter out embeddings of subjects in this batch
        rel_emb = torch.index_select(r, 0, rel)  # filter out embeddings of relations in this batch
        return sub_emb, rel_emb, x

    def cal_ent_embeds(self, g):
        x, r = self.init_embed, self.init_rel  # embedding of relations
        for layer in self.gcn_layers:
            x, r = layer(g, x, r, self.edge_type, self.edge_norm)
            x = self.drop(x)
            r = self.drop(r)
        return x


class VQGCN(nn.Module):
    def __init__(self, num_ent, num_rel, gcn_layers, tf_layers, init_dim, gcn_dim, edge_type, edge_norm,
                 gcn_drop=0., act='tanh', opn='mult', seq_len=16, num_code=1024, att_head=2):
        super(VQGCN, self).__init__()
        if act == 'tanh':
            self.act = torch.tanh
        elif act == 'relu':
            self.act = torch.relu
        self.seq_len = seq_len
        self.gcn_dim = gcn_dim
        self.kggcn = KGGCN(num_ent, num_rel, gcn_layers, init_dim, gcn_dim, edge_type, edge_norm, gcn_drop, opn, self.act)
        # self.decoder = TransformerDecoder(gcn_dim)
        self.encoder = TransformerBlock(gcn_dim, num_heads=att_head, transformer_layers=tf_layers)
        self.entagg = TransformerBlock(gcn_dim, num_heads=att_head, transformer_layers=tf_layers)
        # self.entrelagg = TransformerBlock(gcn_dim)
        self.position_enc = PositionalEncoding(gcn_dim)
        self.codebook = Codebook(num_codes=num_code, latent_dim=gcn_dim)

        # cl loss
        # self.line1 = nn.Linear(init_dim, gcn_dim)
        # self.line2 = nn.Linear(gcn_dim, gcn_dim)

        self.bias = nn.Parameter(torch.zeros(num_ent))  # 预测结果时实体bias
        self.cls_emb_head = self.get_param([1, gcn_dim])  # CLS初始化 1, D
        # self.cls_emb_rel = self.get_param([1, gcn_dim])  # CLS初始化 1, D
        self.loss = nn.BCELoss()
        # self.cl_loss = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(gcn_drop)

        # convE  # 不用
        self.conv_drop = nn.Dropout(gcn_drop)
        self.line_conv1 = nn.Linear(2*gcn_dim, 400)
        # self.line_conv2 = nn.Linear(200*14*14, gcn_dim)
        self.line_conv2 = nn.Linear(200*14*14, gcn_dim)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=200,
                                      kernel_size=(7, 7), stride=1, padding=0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)  # one channel, do bn on initial embedding
        self.bn1 = torch.nn.BatchNorm2d(200)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(gcn_dim)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)


    def concat(self, ent_embed, rel_embed):
        ent_embed = ent_embed.view(-1, 1, self.gcn_dim)
        rel_embed = rel_embed.view(-1, 1, self.gcn_dim)
        stack_input = torch.cat([ent_embed, rel_embed], 1)  # [batch_size, 2, embed_dim]
        stack_input = stack_input.reshape(-1, 1, 20, 20)  # reshape to 2D [batch, 1, 2*k_h, k_w]
        return stack_input


    def forward(self, g, subj, rel, stage=1):
        sub_emb, rel_emb, all_ent = self.kggcn(g, subj, rel)  # B D, B D, M D

        batch_size = sub_emb.size(0)

        sub_exp = sub_emb.unsqueeze(1).repeat(1,self.seq_len-1,1)  # B L-1 D
        # rel_exp = rel_emb.unsqueeze(1).repeat(1,self.seq_len,1)

        # 添加一个CLS token
        cls_emb_head = self.cls_emb_head.unsqueeze(0).repeat(batch_size, 1, 1)  # B 1 D
        sub_trans = torch.cat([cls_emb_head, sub_exp], dim=1)  # B L D
        pos_emds = self.position_enc(sub_trans)  # 1 L D
        sub_trans = sub_trans + pos_emds
        sub_trans = self.encoder(sub_trans)  # 实体表示  之后进行VQ
        sub_trans = self.drop(self.act(sub_trans))

        sub_trans_q, codebook_indices, q_loss = self.codebook(sub_trans)  # 不取CLS codebook表示

        # sub_trans = sub_trans_q  # 增加原始CLS
        # if stage != 1:  # 2 3
        #     sub_trans = sub_trans_q  # 增加原始CLS
        sub_trans = self.entagg(sub_trans)  # 重新使用
        sub_trans = self.drop(self.act(sub_trans))  #
        sub_final = sub_trans[:, 0, :]  # B D 最终实体表示


        # sub_final = sub_emb

        # conve
        # subrel = torch.cat([sub_final, rel_emb], dim=1)
        # subrel = self.line_conv1(subrel).reshape(batch_size, 1, 20, 20)  # 用于卷积
        subrel = self.concat(sub_final, rel_emb)
        # subrel = self.concat(sub_emb, rel_emb)  # 原始gcn输出
        x = self.bn0(subrel)
        x = self.conv2d(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv_drop(x)
        x = x.view(batch_size, -1)
        x = self.line_conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv_drop(x)  # 提前
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]

        # distmult
        # obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        # # obj_emb = sub_final * rel_emb  # [batch_size, emb_dim]
        # x = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, ent_num]

        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        # 对比损失
        # logits, labels = self.cal_cls_loss(sub_final, subj, temperature=0.07)  # 0.07
        # cl_loss = self.cl_loss(logits, labels)

        # return score, cl_loss, cl_loss
        return score, q_loss, q_loss
        # return score, score, score


    def cal_allent_codes(self, g):
        all_ent = self.kggcn.cal_ent_embeds(g)  # M D
        sub_exp = all_ent.unsqueeze(1).repeat(1,self.seq_len,1)  # B L D
        # rel_exp = rel_emb.unsqueeze(1).repeat(1,self.seq_len,1)

        # 添加一个CLS token
        # cls_emb_head = self.cls_emb_head.unsqueeze(0).repeat(batch_size, 1, 1)  #B 1 D
        # sub_trans = torch.cat([cls_emb_head, sub_exp], dim=1)  # B L D
        sub_trans = sub_exp
        pos_emds = self.position_enc(sub_trans)  # 1 L D
        sub_trans = sub_trans + pos_emds
        sub_trans = self.encoder(sub_trans)  # 实体表示  之后进行VQ
        # codebook_indices = self.codebook.cal_codes(sub_trans[:, 1:, :])  # 不取CLS
        codebook_indices = self.codebook.cal_codes(sub_trans)  # 不取CLS
        return codebook_indices


    def cal_cls_loss(self, subs, sub_index, temperature):  # 原始index  对比损失计算
        device = subs.device
        b = subs.size(0)
        ent_len = self.kggcn.num_ent
        input_ents = self.line1(self.kggcn.init_embed)  # M D
        target_ents = self.line2(subs)  # 实体表示  N D

        input_ents = F.normalize(input_ents, dim=1)
        target_ents = F.normalize(target_ents, dim=1)  # 正则

        pos_q = torch.index_select(input_ents, 0, sub_index)  # N D
        neg_q = input_ents.unsqueeze(0).repeat(b,1,1)  # N M D
        mask = torch.ones((b, ent_len), dtype=torch.bool, device=device)  # N M
        b_range = torch.arange(b, device=device)
        mask[b_range, sub_index] = False
        neg_q = neg_q[mask].reshape(b, ent_len-1, -1)

        # print(target_ents.size(), pos_q.size(), neg_q.size())
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [target_ents, pos_q]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,nkc->nk", [target_ents, neg_q])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  # 位置

        return logits, labels





# class TransformerDecoder(nn.Module):  # decoder生成 用于预测结果
#     def __init__(self, embed_dim, num_heads=4, transformer_layers=2):
#         super(TransformerDecoder, self).__init__()
#         layers = []
#         for i in range(transformer_layers):
#             layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True))
#         self.layers = nn.ModuleList(layers)
#
#     def forward(self, x):  # B L D  使用mask或者position embedding
#         # L = x.size(1)
#         # attn_mask = torch.triu(torch.ones(L, L), diagonal=1)  # 下三角不mask
#         # attn_mask = attn_mask == 1
#         # attn_mask = attn_mask.to(x.device)
#         for layer in self.layers:
#             # x, attn_weights = layer(query=x, key=x, value=x, attn_mask=attn_mask)
#             x, attn_weights = layer(query=x, key=x, value=x)
#         return x


# class TransformerBlock(nn.Module):  # decoder聚合结果
#     def __init__(self, embed_dim, num_heads=2, transformer_layers=2):
#         super(TransformerBlock, self).__init__()
#         layers = []
#         for i in range(transformer_layers):
#             layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True))
#         self.layers = nn.ModuleList(layers)
#
#     def forward(self, x, y, z):  # B L D
#         for layer in self.layers:
#             x, attn_weights = layer(query=y, key=z, value=x)
#         return x

class TransformerBlock(nn.Module):  # decoder聚合结果
    def __init__(self, embed_dim, num_heads=4, transformer_layers=2):
        super(TransformerBlock, self).__init__()
        layers = []
        for i in range(transformer_layers):
            layers.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):  # B L D
        for layer in self.layers:
            x, attn_weights = layer(query=x, key=x, value=x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=1024):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = nn.Parameter(pe, requires_grad=False)
        # pe = pe.view(1, max_len, d_model)  # 1 L D
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        emds = self.pe[:L, :].unsqueeze(0)  # 1 L D
        return emds


class Codebook(nn.Module):  # embedding
    def __init__(self, num_codes, latent_dim, beta=0.25):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codes
        self.latent_dim = latent_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)  # ND
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)  # 初始化
        # self.embedding.weight.data.uniform_(-1.0 / 20, 1.0 / 20)  # 初始化

    def forward(self, z):  # B L D
        z_flattened = z.reshape(-1, self.latent_dim)  # M D
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))  # M,1 N -> MN - MN -> MN

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # code embedding

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # print(torch.mean(z), torch.mean(self.embedding.weight), loss)
        # print(torch.mean(z**2), torch.mean(self.embedding.weight**2), loss)

        z_q = z + (z_q - z).detach()  # 为什么加这个？
        return z_q, min_encoding_indices, loss  # codebook的loss

    def cal_codes(self, z):  # 生成code B L D
        z_flattened = z.reshape(-1, self.latent_dim)  # M,D
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened, self.embedding.weight.t()))  # M,1 N -> MN - MN -> MN
        min_encoding_indices = torch.argmin(d, dim=1)  # 变换
        min_encoding_indices = min_encoding_indices.view(z.shape[:2])  # 两个维度
        return min_encoding_indices

