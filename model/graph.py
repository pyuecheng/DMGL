import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class NodeFeaExtractor(nn.Module):
    def __init__(self, hidden_size_st, fc_dim):
        super(NodeFeaExtractor, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, 10, stride=1)  
        self.conv2 = torch.nn.Conv1d(8, 16, 10, stride=1) 
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size_st)
        self.fc = torch.nn.Linear(fc_dim, hidden_size_st)
       

    def forward(self, node_fea):
        t, n = node_fea.shape
        x = node_fea.transpose(1, 0).reshape(n, 1, -1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.bn3(x)
        return x

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class EvolvingGraphLearner(nn.Module):
    def __init__(self, input_size: int, dg_hidden_size: int):  #conv_channels,dy_embedding_dim
        super(EvolvingGraphLearner, self).__init__()
        self.rz_gate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size * 2)
        self.dg_hidden_size = dg_hidden_size
        self.h_candidate = nn.Linear(input_size + dg_hidden_size, dg_hidden_size)
        self.conv = normal_conv(dg_hidden_size)
        self.conv2 = normal_conv(dg_hidden_size)
        self.R = nn.Parameter(torch.zeros(16, 266,20 ), requires_grad=True)
    def forward(self, inputs: Tensor, states):
        """
        :param inputs: inputs to cal dynamic relations   [B,N,C] 16,288,32
        :param states: recurrent state [B, N,C]
        :return:  graph[B,N,N]       states[B,N,C]  
        """
        
        # states = self.attention(states)
        # states = torch.squeeze(torch.mean(states, dim=0))


        b,n,c = states.shape
        # print(inputs.shape)266，40；16，266，32  temporal feature
        # print(states.shape)  #16,266,20
        # print(self.dg_hidden_size) 20
        # GRU
        r_z = torch.sigmoid(self.rz_gate(torch.cat([inputs, states], -1)))
        r, z = r_z.split(self.dg_hidden_size, -1)
        h_ = torch.tanh(self.h_candidate(torch.cat([inputs, r * states], -1)))
        new_state = z * states + (1 - z) * h_
        # new_state = z * states + (1 - z) * h_+self.R
        # new_state=torch.squeeze(new_state)
        # print(new_state.shape)

        # node connection
        dy_sent = torch.unsqueeze(torch.relu(new_state), dim=-2).repeat(1, 1, n, 1)
        # print(dy_sent.shape) 16,266,266,20
        dy_revi = dy_sent.transpose(1,2)
        y = torch.cat([dy_sent, dy_revi], dim=-1)
        # print(y.shape) 16.266.266.40
        support = self.conv(y, (b, n, n))      
        mask = self.conv2(y,(b,n,n))
        support = support * torch.sigmoid(mask)
            
        return support, new_state  # B,N,N
    
   

class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device, alpha): #k=subgraph 20; dim=node_embedding 40; layer_num=3;
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num
        
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k  #subgraph size 
        self.dim = dim
        self.alpha = alpha
        self.idx=torch.arange(self.nnodes).to(device)
        
    def forward(self, x, scale_set):
        # b,idx,c = x.shape
        idx=self.idx
        # print(idx)
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(self.idx)

        # adj_set = []

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

        # nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1*scale_set))
        # nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2*scale_set))
        # a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        # adj0 = F.relu(torch.tanh(self.alpha*a))
            
        
        # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))
        # s1,t1 = adj0.topk(self.k,1)  #subgraph size
        # mask.scatter_(1,t1,s1.fill_(1))
        # # print(mask)
        # adj = adj0*mask
        # # adj_set.append(adj)

        # print(adj.shape)
        # return adj

