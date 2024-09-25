import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor=2):
        super(Dilated_Inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        #input [B, D, N, T]
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        # print(x.shape)
        return x

#--------------------start----------------------------
class Res2Net(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(Res2Net, self).__init__()
        
        # self.conv1 = nn.Conv2d(cin, cout, (1, 5), dilation=(1, dilation_factor))
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv_1_20 = nn.Conv2d(32, 32, kernel_size=(1, 85), stride=(1, 1))
        self.conv_1_135 = nn.Conv2d(32, 32, kernel_size=(1, 135), stride=(1, 1))
        self.conv_1_170 = nn.Conv2d(32, 32, kernel_size=(1, 170), stride=(1, 1))
        self.conv_86_14 = nn.Conv2d(32, 32, kernel_size=(1, 55), stride=(1, 1))
        self.conv_1_72 = nn.Conv2d(32, 32, kernel_size=(1, 72), stride=(1, 1))
        self.conv_72_37 = nn.Conv2d(32, 32, kernel_size=(1, 37), stride=(1, 1))
        self.conv_44_8 = nn.Conv2d(32, 32, kernel_size=(1, 25), stride=(1, 1))
        self.conv_end = nn.Conv2d(32, 32, kernel_size=(1, 6), stride=(1, 1))
        self.Upsample = nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
        self.dconv = nn.ConvTranspose2d(32, 32, kernel_size=(1, 4), stride=(1, 1))
    def forward(self, input):
        b, c, n, f = input.shape
        # print(input.shape)
        input = self.conv(input)
        # print(input.shape)
        x1 = input
        x2 = input
        x3 = input
        x4 = input
        x2 = self.conv2(x2)
        # print(x2.shape)
        x3 = torch.cat([x2, x3], dim=3)
        x3 = self.conv3(x3)
        # print(x3.shape)
        x4 = torch.cat([x3, x4], dim=3)
        x4 = self.conv4(x4)
        # print(x4.shape)
        x = torch.cat([x1, x2, x3, x4], dim=3)
        _, _, _, f1 = x.shape
        # print(x.shape)
        if f1 == 65:
            x =  self.conv_86_14(x)
        elif f1 == 30:
            x =  self.conv_44_8(x)
        elif f1==72:
            x=self.conv_1_72(x)
        elif f1==37:
            x=self.conv_72_37(x)
        elif f1==135:
            x=self.conv_1_135(x)
        elif f1==170:
            x=self.conv_1_170(x)
        else:
            x = self.conv_1_20(x)
        # print(x.shape)
        x = x + input
        x = self.conv_end(x)
       

        return x
#--------------------end-----------------------------


class Res2Net_e(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(Res2Net_e, self).__init__()
        
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1))
        self.conv = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv_1_20 = nn.Conv2d(32, 32, kernel_size=(1, 109), stride=(1, 1))
        self.conv_86_14 = nn.Conv2d(32, 32, kernel_size=(1, 73), stride=(1, 1))
        self.conv_44_8 = nn.Conv2d(32, 32, kernel_size=(1, 37), stride=(1, 1))
        self.conv_end = nn.Conv2d(32, 32, kernel_size=(1, 6), stride=(1, 1))
        self.Upsample = nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
        self.dconv = nn.ConvTranspose2d(32, 32, kernel_size=(1, 4), stride=(1, 1))
    def forward(self, input):
        x = self.conv(input)
        return x
    
    
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        """
         :param X: tensor, [B, D, N, T]
         :param A: tensor [N, N] , [B, N, N] or [T*, B, N, N]
         :return: tensor [B, D, N, T]        
        """
        #x = torch.einsum('ncwl,vw->ncvl',(x,A))
        if len(A.shape) == 2:
            a_ ='vw'
        elif len(A.shape) == 3:
            a_ ='bvw'
        else:
            a_ = 'tbvw'
        x = torch.einsum(f'bcwt,{a_}->bcvt',(x,A))
        return x.contiguous()
       

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class MixProp(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(MixProp, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
    
    def forward(self,x,adj):
        # print(adj.shape) 16,266,266
        h = x
        # out = [h]
        # for i in range(self.gdep):
        #     h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj)
        #     out.append(h)
        # ho = torch.cat(out,dim=1)# [B, D*(1+gdep), N, T]
        # ho = self.mlp(ho) #[B, c_out, N, T]
        # return ho


        h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj)
        return h
        # return self.nconv(h,adj)

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)

        ho = torch.cat(out,dim=1)

        ho = self.mlp(ho)

        return ho
  
class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)




