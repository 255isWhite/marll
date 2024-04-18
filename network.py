import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def orthogonal_init(layers, gain=1.0):
    for name,param in layers.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param,gain=gain)
        elif 'bias' in name:
            nn.init.constant_(param,0)

class Q_Net(nn.Module):
    def __init__(self,args):
        super(Q_Net,self).__init__()
        
        self.fc1 = nn.Linear(args.input_dim,args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim,args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim,args.action_dim)
        
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Q_Net_RNN(nn.Module):
    def __init__(self,args):
        super(Q_Net_RNN,self).__init__()
        
        self.rnn_hidden = None
        self.fc1 = nn.Linear(args.input_dim,args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim,args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim,args.action_dim)
        
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        self.rnn_hidden = self.rnn(x,self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        return x
    
class QMIX_Net(nn.Module):
    def __init__(self,args):
        super(QMIX_Net,self).__init__()
        
        self.hyper_w1 = nn.Sequential(nn.Linear(args.state_dim,args.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.hyper_hidden_dim,args.qmix_hidden_dim*args.N)
                                      )
        self.hyper_w2 = nn.Sequential(nn.Linear(args.state_dim,args.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.hyper_hidden_dim,args.qmix_hidden_dim)
                                      )
        self.hyper_b1 = nn.Linear(args.state_dim,args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_dim,args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim,1)
                                      )
        self.args = args
        # if args.use_orthogonal_init:
        #     orthogonal_init(self.hyper_w1)
        #     orthogonal_init(self.hyper_w2)
        #     orthogonal_init(self.hyper_b1)
        #     orthogonal_init(self.hyper_b2)

    def forward(self,q,s):
        # q (batch_size,max_episode_len,N)
        # s (batch_size,max_episode_len,state_dim)
        
        q = q.view(-1,1,self.args.N) # (batch_size*max_episode_len,1,N)
        s = s.reshape(-1,s.shape[-1]) # (batch_size*max_episode_len,state_dim)
        w1 = torch.abs(self.hyper_w1(s)) # (batch_size*max_episode_len,N*qmix_hidden_dim)
        w1 = w1.view(-1,self.args.N,self.args.qmix_hidden_dim) # (batch_size*max_episode_len,N,qmix_hidden_dim)
        b1 = self.hyper_b1(s) # (batch_size*max_episode_len,qmix_hidden_dim)
        b1 = b1.view(-1,1,self.args.qmix_hidden_dim) # (batch_size*max_episode_len,1,qmix_hidden_dim)
        
        q_hidden = F.elu(torch.bmm(q,w1) + b1) # (batch_size*max_episode_len,1,qmix_hidden_dim)
        
        w2 = self.hyper_w2(s) # (batch_size*max_episode_len,qmix_hidden_dim)
        w2 = w2.view(-1,self.args.qmix_hidden_dim,1) # (batch_size*max_episode_len,qmix_hidden_dim,1)
        b2 = self.hyper_b2(s) # (batch_size*max_episode_len,1)
        b2 = b2.view(-1,1,1) # (batch_size*max_episode_len,1,1)
        
        q_tot = torch.bmm(q_hidden,w2) + b2 # (batch_size*max_episode_len,1,1)
        q_tot = q_tot.view(self.args.batch_size,-1,1) # (batch_size,max_episode_len,1)
        
        return q_tot
    
class VDN_Net(nn.Module):
    def __init__(self,args):
        super(VDN_Net,self).__init__()
        
    def forward(self,x):
        x = torch.sum(x,dim=-1,keepdim=True)
        return True