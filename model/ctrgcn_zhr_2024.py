import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        # print('TemporalConv里x:', x.shape)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
            # print('out形状:',out.shape)

        out = torch.cat(branch_outs, dim=1)
        out += res
        # print('最终out的形状 ',out.shape)
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        # if self.in_channels == 3:
        #     print('CTRGC最初的x ',x.shape) #torch.Size([2, 3, 64, 25])
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # if self.in_channels == 3:
        #     print('CTRGC最初的x1 ',x1.shape) #torch.Size([2, 8, 25])
        #     print('CTRGC最初的x2 ',x2.shape) #torch.Size([2, 8, 25])
        #     print('CTRGC最初的x3 ',x3.shape) #torch.Size([2, 64, 64, 25])
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # if self.in_channels == 3:
        #     print('CTRGC的x1 ',x1.shape)#torch.Size([2, 8, 25, 25])
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        # if self.in_channels == 3:
        #     print('CTRGC的x1 ',x1.shape)#torch.Size([2, 64, 25, 25])
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        # if self.in_channels == 3:
        #     print('CTRGC的x1 ',x1.shape)#torch.Size([2, 64, 64, 25])
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha) #torch.Size([2, 64, 64, 25])
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        # print('y的形状',y.shape) (人体数，当前通道数，帧数，关键点数)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        # print('y的形状 ',y.shape)
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=1, adaptive=adaptive) #此处stride=2使得时间帧数减半
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=1, adaptive=adaptive) #此处stride=2使得时间帧数减半
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.attentionlayer = attentionModel(256,25)

        #  创建一个多头注意力层
        self.mha = MultiHeadAttention(256, 8)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, label, index):
        # print('shape1',x.shape) #torch.Size([1, 3, 64, 25, 2])
        # print('label',label.shape)
        # print('index',index.shape)
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        # print("小z先生")
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        # print('shape2',x.shape)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        # print('shape3',x.shape) #torch.Size([2, 3, 64, 25])
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

# 增加注意力模块start
        attentionTensor = x.clone() #torch.Size([2, 256, 64, 25])
        attentionTensor = attentionTensor.mean(2).mean(0).unsqueeze(0)
        attentionTensor = self.attentionlayer(attentionTensor) #torch.Size([1, 256, 25])
# 增加注意力模块end
        
# 尝试多头注意力模块start
        attentionTensor = x.clone() #torch.Size([2, 256, 64, 25])
        attentionTensor = attentionTensor.mean(2).mean(0).unsqueeze(0)
        attentionTensor = attentionTensor.permute(0,2,1)
        # attentionTensor = attentionTensor.repeat(8,  1,  1)
        output, attention_weights = self.mha(attentionTensor, attentionTensor, attentionTensor)
        print(output.shape)  #  输出：torch.Size([1,  25,  1])
        # print(output)  
        output = output.unsqueeze(0).permute(0,1,3,2)

        x = x*output
        # print(attention_weights.shape) 
# 尝试多头注意力模块end

        # print('shape4',x.shape) #torch.Size([2, 256, 64, 25])
        # N*M,C,T,V
        c_new = x.size(1)
        Z = x
        x = x.view(N, M, c_new, -1)
        

        # print('shape5',x.shape) #torch.Size([64, 2, 256, 400])
        x = x.mean(3).mean(1)
        # print('shape6',x.shape) #torch.Size([64, 256])
        x = self.drop_out(x)
        # print('shape7',x.shape) #torch.Size([64, 256])
        # print('self.fc(x)',self.fc(x).shape)
        z_size = x.size(0)
        print('z的形状1:',Z.shape)
        Z = Z.view(z_size, 2, 256, T, 25).permute(0,3,1,4,2)
        print('z的形状2:',Z.shape)
        Z = Z[0]
        print('z的形状3:',Z.shape)
        return Z, self.fc(x) # Z.shape = ([1, 64, 2, 25, 256])
    

class attentionModel(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(attentionModel, self).__init__()
        chennel = 64
        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_feature = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=1)

        self.conv2_3_k = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)
        self.conv2_3_v = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv1d(chennel, ch_out, kernel_size=1, stride=1, padding=0)

        self.opt = nn.ReLU()

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        #输出格式为64个权重值
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.bmm(value, y_weight)
        sum = value[:, :, -1:] + y_sum
        return torch.cat((value[:, :, :-1], sum), dim=-1)

    def forward(self, input):
        # input = input.permute(0, 2, 1)

        k = self.conv1_3_k(input)
        v = self.conv1_3_v(input)
        #运行得到的各个张量大小
        # input torch.Size([1, 256, 25])
        # k torch.Size([1, 64, 25])
        # k[:, :, -1:] torch.Size([1, 64, 1])
        y_weight = self.weight(k, k[:, :, -1:])
        feat1 = self.sum(v, y_weight)
        feat1 = self.opt(feat1)

        k = self.conv2_3_k(feat1)
        v = self.conv2_3_v(feat1)
        y_weight = self.weight(k, k[:, :, -1:])
        feat2 = self.sum(v, y_weight)
        feat2 = self.opt(feat2)

        return self.conv3_1(feat2)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads ==  0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, 1)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1,  2)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / self.depth**0.5

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        print('attention_weights',attention_weights.shape)

        output = torch.matmul(attention_weights, v)

        output = output.transpose(1,  2).contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(output)
        output = self.fc(output)
        output = F.softmax(output, dim=-1)

        return output, attention_weights









class gululu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(gululu, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x): #torch.Size([2, 256, 64, 25])
        x = self.bn(self.conv(x))
        return x

class Temporal_CNN_Model(nn.Module):
    def __init__(self):
        super(Temporal_CNN_Model, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层卷积层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三层卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四层卷积层
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)  # 输出单元数设置为1
        
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 前向传播
        print('第一层')
        print(x.shape)
        print(x)
        x = self.conv1(x)
        print(x.shape)
        x = self.relu1(x)
        print(x.shape)
        # x = self.pool1(x)
        print(x.shape)
        
        print('第二层')
        x = self.conv2(x)
        print(x.shape)
        x = self.relu2(x)
        print(x.shape)
        # x = self.pool2(x)
        print(x.shape)
        
        print('第三层')
        x = self.conv3(x)
        print(x.shape)
        x = self.relu3(x)
        print(x.shape)
        # x = self.pool3(x)
        print(x.shape)
        
        print('第四层')
        x = self.conv4(x)
        print(x.shape)
        x = self.relu4(x)
        print(x.shape)
        # x = self.pool4(x)
        print(x.shape)
        
        # print('奇怪层')
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        
        # print('全连接层')
        # x = self.fc1(x)
        # print(x.shape)
        # x = self.relu5(x)
        # print(x.shape)
        # x = self.fc2(x)
        # print(x.shape)
        
        # 使用Sigmoid激活函数，将输出值映射到0到1之间

        print(x)
        x = self.sigmoid(x)
        print(x)
        
        return x