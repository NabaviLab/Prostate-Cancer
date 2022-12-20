import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D,Add
from tensorflow.keras.layers import Conv2DTranspose,Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation,MaxPool2D
from tensorflow.keras.layers import Concatenate,Dense,Multiply,Flatten
from tensorflow.keras.layers import Dropout,Reshape,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
#from utils.utils import initialize_weights
import numpy as np
import torch
from torch.nn.parameter import Parameter
import math


def Autoencoder(input_shape=(512, 512, 3)):

    backbone = tf.keras.applications.EfficientNetB7(weights='imagenet',
                            include_top=False
                            ,input_shape=input_shape)
    
    input = backbone.input
    start_neurons = 8

    conv4 = backbone.layers[-3].output
    
    conv4 = LeakyReLU(alpha=0.1)(conv4)

    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(conv4)
    convm = LeakyReLU(alpha=0.1)(convm)    
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(deconv4)    
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)    
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(deconv3)    
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)    
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(deconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)    
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(deconv1) 
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)    
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    output_layer = Conv2D(3, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = tf.keras.models.Model(input, output_layer)
    

    return model

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
            
            
            
            
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 2560, D = 512, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.4))
            self.attention_b.append(nn.Dropout(0.4))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 512, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.4))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes
    
    
    
class Model1(nn.Module):
   # def __init__(self):
        #super(Model, self).__init__()
    def __init__(self, gate = True, hidden_dim = 512, out_dim = 256,size_arg = "small", dropout = False, n_classes = 5, lr = 5e-5,weight_decay = 10E-6):
        super(Model1, self).__init__()
        self.size_dict = {"small": [2560, 512, 256], "big": [1024, 1, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], hidden_dim), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.4))

        if gate:
            attention_net = Attn_Net_Gated(L = hidden_dim, D = out_dim, dropout = dropout, n_classes = 1)

        else:
            attention_net = Attn_Net(L = hidden_dim, D = out_dim, dropout = dropout, n_classes = 1)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(hidden_dim, n_classes)

        initialize_weights(self)

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.NLLLoss()

        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h) 
        #print(A)
        A = torch.transpose(A, 1, 0) 
        
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
       # print(A)
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return logits, Y_prob, Y_hat, A_raw, results_dict,M  
    
    
    
class MIL_Attention_fc_surv(Model1):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 5):
        super(MIL_Attention_fc_surv, self).__init__(gate = gate, size_arg = size_arg, dropout = dropout, n_classes = n_classes)

    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h) 
        logits  = self.classifier(M) 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        return hazards, S, Y_hat, A_raw, results_dict, M
    
    
    
    
    
    
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight    # X * W
        output = adj @ support           # A * X * W
        if self.bias is not None:        # A * X * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, gcn_feature_dim = 2560, gcn_hid_dim = 512, gcn_out_dim = 64):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(gcn_feature_dim, gcn_hid_dim)
        self.ln1 = nn.LayerNorm(gcn_hid_dim)
        self.gc3 = GraphConvolution(gcn_hid_dim, gcn_hid_dim)
        self.ln3 = nn.LayerNorm(gcn_hid_dim)
        self.gc2 = GraphConvolution(gcn_hid_dim, gcn_out_dim)
        self.ln2 = nn.LayerNorm(gcn_out_dim)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)
        self.relu3 = nn.LeakyReLU(0.2,inplace=True)
        self.relu4 = nn.LeakyReLU(0.2,inplace=True)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, adj):  			# x.shape = (seq_len, GCN_FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = self.gc1(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu1(self.ln1(x))
        x = self.gc3(x, adj)  				# x.shape = (seq_len, GCN_HIDDEN_DIM)
        x = self.relu3(self.ln3(x))
        x = self.gc2(x, adj)
        output = self.relu2(self.ln2(x))	# output.shape = (seq_len, GCN_OUTPUT_DIM)
        return output


class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention


class Model2(nn.Module):
    def __init__(self, gcn_feature_dim = 2560, gcn_hid_dim = 512, gcn_out_dim = 64,dense_dim = 16,n_heads = 6, n_class = 5, lr = 1e-5,weight_decay = 10E-7):
        super(Model2, self).__init__()

        self.gcn = GCN(gcn_feature_dim = 2560, gcn_hid_dim = 512, gcn_out_dim = 64)
        self.attention = Attention(gcn_out_dim, dense_dim, n_heads)
        self.fc_final = nn.Linear(gcn_out_dim, n_class)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, adj):  											# x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        x = self.gcn(x, adj)  												# x.shape = (seq_len, GAT_OUTPUT_DIM)

        x = x.unsqueeze(0).float()  										# x.shape = (1, seq_len, GAT_OUTPUT_DIM)
        att = self.attention(x)  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ x 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        logits = torch.sigmoid(self.fc_final(node_feature_embedding_avg))  	# output.shape = (1, NUM_CLASSES)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = torch.softmax(logits, dim = 1)
        return logits, Y_hat, Y_prob    
    
    
    
class GraphAttentionLayer(nn.Module):
    def __init__(self, inp, out, slope):
        
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(inp, out, bias=False)
        self.a = nn.Linear(out*2, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)
        self.softmax = nn.Softmax(dim=1)
  
    def forward(self, h, adj):
        Wh = self.W(h)
        Whcat = self.Wh_concat(Wh, adj)
        e = self.leakyrelu(self.a(Whcat).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = self.softmax(attention)
        h_hat = torch.mm(attention, Wh)

        return h_hat
 
    def Wh_concat(self, Wh, adj):
        N = Wh.size(0)
        Whi = Wh.repeat_interleave(N, dim=0)
        Whj = Wh.repeat(N, 1)
        WhiWhj = torch.cat([Whi, Whj], dim=1)
        WhiWhj = WhiWhj.view(N, N, Wh.size(1)*2)

        return WhiWhj
 
class MultiHeadGAT(nn.Module):
    def __init__(self, inp, out, heads, slope):
        super(MultiHeadGAT, self).__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(inp, out, slope) for _ in range(heads)])
        self.tanh = nn.Tanh()
  
    def forward(self, h, adj):
        heads_out = [att(h, adj) for att in self.attentions]
        out = torch.stack(heads_out, dim=0).mean(0)
    
        return self.tanh(out)
 
class GAT(nn.Module):
    def __init__(self, gcn_feature_dim= 2560, gcn_hid_dim = 256, gcn_out_dim =64,dense_dim = 16, gat_heads=4, slope=0.01,n_heads = 6, n_class = 5, lr = 1e-5,weight_decay = 10E-7):
        super(GAT, self).__init__()
        self.gat1 = MultiHeadGAT(gcn_feature_dim, gcn_hid_dim, gat_heads, slope)
        self.gat2 = MultiHeadGAT(gcn_hid_dim, gcn_out_dim, gat_heads, slope)
        self.attention = Attention(gcn_out_dim, dense_dim, n_heads)
        #self.fc_final = nn.Linear(GCN_OUTPUT_DIM, NUM_CLASSES)
        self.fc_final = nn.Linear(gcn_out_dim, n_class)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
  
    def forward(self, h, adj):
        out = self.gat1(h, adj)
        out = self.gat2(out, adj)
        att = self.attention(out.unsqueeze(0).float())  											# att.shape = (1, ATTENTION_HEADS, seq_len)
        node_feature_embedding = att @ out 									# output.shape = (1, ATTENTION_HEADS, GAT_OUTPUT_DIM)
        node_feature_embedding_avg = torch.sum(node_feature_embedding,
                                               1) / self.attention.n_heads  # node_feature_embedding_avg.shape = (1, GAT_OUTPUT_DIM)
        logits = torch.sigmoid(self.fc_final(node_feature_embedding_avg))  	# output.shape = (1, NUM_CLASSES)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = torch.softmax(logits, dim = 1)
        return logits, Y_hat, Y_prob
        #return out