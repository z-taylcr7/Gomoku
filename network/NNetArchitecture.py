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
        self.line1=nn.Conv2d(3,64,kernel_size=3,padding=0)
        self.line2=nn.ReLU(inplace=True)
        self.line3=nn.Conv2d(64,4,kernel_size=3,padding=0)
        self.line4=nn.ReLU(inplace=True)
        self.l1=nn.Conv2d(3,64,kernel_size=3,padding=0)
        self.l2=nn.ReLU(inplace=True)
        self.l3=nn.Conv2d(64,128,kernel_size=3,padding=0)
        self.l4=nn.ReLU(inplace=True)
        self.l5=nn.Conv2d(128,512,kernel_size=1,padding=0)
        self.l6=nn.ReLU(inplace=True)
        self.l7=nn.Conv2d(512,4,kernel_size=1,padding=0)
        self.l8=nn.ReLU(inplace=True)
        
        self.out=nn.Linear(12800,82)
        self.out_val=nn.Linear(100,1)###why 100?
        self.seq=nn.Sequential(
            #nn.Conv2d(3,64,kernel_size=3,padding=0),nn.ReLU(inplace=True),nn.Conv2d(64,4,kernel_size=1,padding=0),nn.ReLU(inplace=True)
            self.l1,self.l2,self.l3,self.l4,self.l5,self.l6
        )
        self.seq_val=nn.Sequential(
            self.line1,self.line2,self.line3,self.line4
        )



        

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   
        #p1=self.conv1(s.clone())
        s1=self.seq(s.clone())
        
        s1=s1.clone().view(s1.clone().size(0),-1)
        pi=self.out(s1.clone())

        #v1=self.conv_val(s.clone())
        s=self.seq_val(s.clone())
       
        s=s.clone().view(s.clone().size(0),-1)
        
        v=self.out_val(s.clone())
        
        """
            TODO: Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.

            pi = ...
            v = ...
        """

        # Think: What are the advantages of using log_softmax ?
        return F.log_softmax(pi, dim=1), torch.tanh(v)