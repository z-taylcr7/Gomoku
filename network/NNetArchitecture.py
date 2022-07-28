from importlib.metadata import requires
from matplotlib.cbook import flatten
from numpy import mean, size
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')


class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params

        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.pb1=nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.pb2=nn.ReLU(inplace=True)
        self.pb3=nn.Conv2d(16,64,kernel_size=3,padding=1)
        self.pb4=nn.ReLU(inplace=True)
        self.pb7=nn.Conv2d(64,256,kernel_size=3,padding=1)
        self.pb8=nn.ReLU(inplace=True)

        self.line1=nn.Conv2d(256,2,kernel_size=1,padding=0)
        self.line2=nn.ReLU(inplace=True)
        self.line3=nn.Linear(162,128)
        self.line4=nn.ReLU(inplace=True)
        
        self.l1=nn.Conv2d(256,4,kernel_size=1,padding=0)
        self.l2=nn.ReLU(inplace=True)
        
        
        self.out=nn.Linear(324,82)
        self.out_val=nn.Linear(128,1)
        self.seq_pb=nn.Sequential(
            self.pb1,self.pb2,self.pb3,
            self.pb4,
            self.pb7,self.pb8
        )
        self.seq=nn.Sequential(
            #nn.Conv2d(3,64,kernel_size=3,padding=0),nn.ReLU(inplace=True),nn.Conv2d(64,4,kernel_size=1,padding=0),nn.ReLU(inplace=True)
            self.l1,self.l2
        )
        self.seq_val=nn.Sequential(
            self.line1,self.line2,self.line3,self.line4
        )

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   
        #p1=self.conv1(s.clone())
        s=self.seq_pb(s)
        s1=self.seq(s)
        s1=s1.view(-1,324)
        pi=self.out(s1)

        #v1=self.conv_val(s.clone())
        s2=self.line1(s)
        s2=self.line2(s2)
        s2=s2.view(-1,162)
        s2=self.line3(s2)
        s2=self.line4(s2)
        v=self.out_val(s2)
        
        """
            TODO: Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.

            pi = ...
            v = ...
        """

        # Think: What are the advantages of using log_softmax ?
        return F.log_softmax(pi, dim=1), torch.tanh(v)