import torch
from torch import nn, Tensor
from torch.nn import functional as F


from .esg_utils import Dilated_Inception,Res2Net, Res2Net_e, MixProp, LayerNorm,mixprop
from .graph import  NodeFeaExtractor, graph_constructor,EvolvingGraphLearner
import pickle

class TConv(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, dropout:float):
        super(TConv, self).__init__()
        # self.filter_conv = Dilated_Inception(residual_channels, conv_channels,kernel_set, dilation_factor)
        # self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor)
        self.filter_conv = Res2Net(residual_channels, conv_channels,dilation_factor)
        self.gate_conv = Res2Net(residual_channels, conv_channels, dilation_factor)
        # self.filter_conv = Res2Net_e(residual_channels, conv_channels,dilation_factor)
        # self.gate_conv = Res2Net_e(residual_channels, conv_channels, dilation_factor)

        self.dropout = dropout

    def forward(self, x: Tensor):
        _filter = self.filter_conv(x)
        filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        gate = torch.sigmoid(_gate)
        x = filter * gate  
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        return x


class Evolving_GConv(nn.Module):
    def __init__(self,device,num_nodes:int, conv_channels: int, residual_channels: int, gcn_depth: int,  st_embedding_dim: int, 
                dy_embedding_dim: int, dy_interval: int, scale_set, dropout=0.3, propalpha=0.05):
        super(Evolving_GConv, self).__init__()
        self.linear_s2d = nn.Linear(st_embedding_dim, dy_embedding_dim)
        # self.scale_spc_EGL = graph_constructor(num_nodes,20, st_embedding_dim,3,device,3)
        self.scale_spc_EGL =EvolvingGraphLearner(conv_channels,dy_embedding_dim)
        self.dy_interval = dy_interval         
        self.scale_set=scale_set

        self.gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
       
        # with open('./data/adj_mx_taxi.pkl', 'rb') as f:
        #     self.adj = torch.Tensor(pickle.load(f)[2]).cuda(1)
    def forward(self, x, st_node_fea):

        b, _, n, t = x.shape 
        # print(x.shape) 16，32，266，11；6，1
        # print(st_node_fea.shape) 266,40
        dy_node_fea = self.linear_s2d(st_node_fea).unsqueeze(0)  
        states_dy = dy_node_fea.repeat( b, 1, 1) #[B, N, C]
        # print(states_dy.shape) 16 266 20
        x_out = []
    
        
        # dy_graph= self.scale_spc_EGL(x,self.scale_set)

        for i_t in range(0,t,self.dy_interval):     
            x_i =x[...,i_t:min(i_t+self.dy_interval,t)]
            # print(x_i.shape) 16，32，266，1
            input_state_i = torch.mean(x_i.transpose(1,2),dim=-1)
            # print(input_state_i.shape) 16，266，32

            dy_graph, states_dy= self.scale_spc_EGL(input_state_i, states_dy)
            # dy_graph= self.scale_spc_EGL(input_state_i, states_dy)
            # print(dy_graph)
            # print(dy_graph.shape)# 16,266,266
            # dy_graph=self.adj.unsqueeze(0).repeat(16,1,1)
            x_out.append(self.gconv(x_i, dy_graph))  #GCN
        
        # x_out = self.gconv(x, dy_graph) #[B, c_out, N, T]
        # print(len(x_out)) 11,6,1
        # print(x_out[0][0].shape) 32,266,1
        x_out = torch.cat(x_out, dim= -1) #[B, c_out, N, T] 16,32,266,6
    

        return x_out

class Extractor(nn.Module):
    def __init__(self, residual_channels: int, conv_channels: int, kernel_set, dilation_factor: int, gcn_depth: int, 
                st_embedding_dim, dy_embedding_dim, 
           skip_channels:int, t_len: int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int,device,scale_set:float):
        super(Extractor, self).__init__()

        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))
      
        self.s_conv = Evolving_GConv(device,num_nodes,conv_channels, residual_channels, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                                    dy_interval,scale_set, dropout, propalpha)

        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
        
        self.norm = LayerNorm((residual_channels, num_nodes, t_len),elementwise_affine=layer_norm_affline)
       

    def forward(self, x: Tensor,  st_node_fea: Tensor):
        residual = x # [B, F, N, T]
        # dilated convolution
        x = self.t_conv(x)       
        # parametrized skip connection
        skip = self.skip_conv(x)
        #graph convolution
        # print(x.shape) 16,32,266,11
        # print(st_node_fea.shape) 266,40
        x = self.s_conv(x,  st_node_fea)  
        # print(x.shape)     16,32,266,11 
     
        #residual connection
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip


class Block(nn.ModuleList):
    def __init__(self, block_id: int, total_t_len : int, kernel_set, dilation_exp: int, n_layers: int, residual_channels: int, conv_channels: int,
    gcn_depth: int, st_embedding_dim, dy_embedding_dim,  skip_channels:int, num_nodes: int, layer_norm_affline, propalpha: float, dropout:float, dy_interval: int, device):
        super(Block, self).__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1+ block_id*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            rf_block = block_id*n_layers*(kernel_size-1) + 1
        
        dilation_factor = 1
        self.scale_set = [1, 0.8, 0.6, 0.5,0.3]
        for i in range(1, n_layers+1):            
            if dilation_exp>1:
                rf_size_i = int(rf_block + (kernel_size-1)*(dilation_exp**i-1)/(dilation_exp-1))
            else:
                rf_size_i = rf_block + i*(kernel_size-1)
            t_len_i = total_t_len - rf_size_i +1

            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, st_embedding_dim, dy_embedding_dim, 
                 skip_channels, t_len_i, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval[i-1],device,self.scale_set[i-1])
            )
            dilation_factor *= dilation_exp

    def forward(self, x: Tensor, st_node_fea: Tensor, skip_list):
        flag = 0
        for layer in self:
            flag +=1
            x, skip = layer(x, st_node_fea)
            skip_list.append(skip)           
        return x, skip_list


class ESG(nn.Module):
    def __init__(self,                 
                 dy_embedding_dim: int,#20
                 dy_interval: list, #[1,1,1]
                 num_nodes: int,
                 seq_length: int, #12
                 pred_len : int,#12
                 in_dim: int, #2
                 out_dim: int, #2
                 n_blocks: int, #1
                 n_layers: int,   #3             
                 conv_channels: int, #32
                 residual_channels: int, #32
                 skip_channels: int, #32
                 end_channels: int, #128
                 kernel_set: list, #[2,6]
                 dilation_exp: int, #1
                 gcn_depth: int,       #2                          
                 device,
                 fc_dim: int, #95744
                 st_embedding_dim=40,
                 static_feat=None, 
                 dropout=0.3, 
                 propalpha=0.05,
                 layer_norm_affline=True            
                 ):
        super(ESG, self).__init__()
       
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.num_nodes = num_nodes        
        self.device = device
        self.pred_len = pred_len
        self.st_embedding_dim = st_embedding_dim
        self.seq_length = seq_length
        kernel_size = kernel_set[-1]
        if dilation_exp>1:
            self.receptive_field = int(1+n_blocks*(kernel_size-1)*(dilation_exp**n_layers-1)/(dilation_exp-1))
        else:
            self.receptive_field = n_blocks*n_layers*(kernel_size-1) + 1
        self.total_t_len = max(self.receptive_field, self.seq_length)      
      
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        for block_id in range(n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, kernel_set, dilation_exp, n_layers, residual_channels, conv_channels, gcn_depth,
                 st_embedding_dim, dy_embedding_dim, skip_channels, num_nodes, layer_norm_affline, propalpha, dropout, dy_interval, device))

        self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.total_t_len), bias=True)
        self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.total_t_len-self.receptive_field+1), bias=True)
        

        in_channels = skip_channels
        final_channels = pred_len * out_dim

        
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, end_channels, kernel_size=(1,1), bias=True),
            nn.ReLU(),
            nn.Conv2d(end_channels, final_channels, kernel_size=(1,1), bias=True)     
        )
        self.stfea_encode = NodeFeaExtractor(st_embedding_dim, fc_dim)
        self.static_feat = static_feat
       

    def forward(self, input):
        """
        :param input: [B, in_dim, N, n_hist]
        :return: [B, n_pred, N, out_dim]
        """

        b, _, n, t = input.shape
        # print(input.shape) 16,2,266,12
        assert t==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = F.pad(input,(self.receptive_field-self.seq_length,0,0,0), mode='replicate')
        
        x = self.start_conv(input)
        # print(x.shape) 16.32,266,16
        st_node_fea = self.stfea_encode(self.static_feat)


        skip_list = [self.skip0(F.dropout(input, self.dropout, training=self.training))]
        for j in range(self.n_blocks):    
            x, skip_list= self.blocks[j](x, st_node_fea , skip_list)
                    
        skip_list.append(self.skipE(x)) 
        skip_list = torch.cat(skip_list, -1)#[B, skip_channels, N, n_layers+2]
       
        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)  #[B, skip_channels, N, 1]
        x = self.out(skip_sum) #[B, pred_len* out_dim, N, 1] 
        x = x.reshape(b, self.pred_len, -1, n).transpose(-1, -2) #[B, pred_len, N, out_dim]
        return x 





        







    