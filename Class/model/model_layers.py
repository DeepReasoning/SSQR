import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np


class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, drop_rate=0., opn='mult'):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        self.rel = rel_repr
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        x = self.bn(x)

        rel_repr = self.act(torch.matmul(self.rel, self.w_rel))
        return self.act(x), rel_repr


if __name__ == '__main__':
    compgcn = CompGCNCov(in_channels=10, out_channels=5)
    src, tgt = [0, 1, 0, 3, 2], [1, 3, 3, 4, 4]
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges(src, tgt)  # src -> tgt
    g.add_edges(tgt, src)  # tgt -> src
    edge_type = torch.tensor([0, 0, 0, 1, 1] + [2, 2, 2, 3, 3])
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    g.ndata['xxx'] = torch.tensor(norm)
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    edge_norm = g.edata.pop('xxx').squeeze()

    x = torch.randn([5, 10])
    rel = torch.randn([4, 10])  # 2*2+1
    x, rel = compgcn(g, x, rel, edge_type, edge_norm)
    print(x.shape, rel.shape)
