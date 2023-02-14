import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaModel
import math
import pandas as pd
from torch import optim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
head_model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
head_transformer = head_model.transformer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
for param in head_model.parameters():
    
   param.requires_grad = False


lm_head = head_model.lm_head

class CrossAttentionSingle(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        """
        Single head cross attention block scaled
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
        
        self.WQ = torch.randn((self.d_dim, self.attention_dim), requires_grad=True).to(device)
        self.WK = torch.randn((self.e_dim, self.attention_dim), requires_grad=True).to(device)
        self.WV = torch.randn((self.e_dim, self.attention_dim), requires_grad=True).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        

    def forward(self, encoder_x, decoder_x):
        
        #print(f"self.WQ: {self.WQ}")
        Q = torch.mm(decoder_x.to(device), self.WQ ).to(device)
        #print(f"Q shape {Q.shape}")
        #print(f"Q {Q}")
        K = torch.mm(encoder_x.to(device), self.WK ).to(device)
        #print(f"K shape {K.shape}")
        #print(f"K {K}")
        V = torch.mm(encoder_x.to(device), self.WV ) .to(device)
        #print(f"V shape {V.shape}")
        #print(f"V {V}")
        QKT = torch.mm(Q, K.t()).to(device)
        #print(f"QKT shape {QKT.shape}")
        #print(f"QKT  {QKT}")
      
        # Q d_lenXd_dim
        # K e_lenXd_dim
        # V e_lenXd_dim
        QKT_div = torch.div(QKT,math.sqrt(self.d_dim))
        
        SM = self.softmax(QKT_div).to(device) # may need the div from my earlier transformer
        #print(f"SM  {SM}")
        
        attention = torch.mm(SM, V).to(device) 
        #print(f"attention shape {attention.shape}")
        return attention
    
    
class ProposedModel(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
        self.cross_a = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
        self.FF = nn.Linear(self.attention_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        attention = self.cross_a(encoder_x, decoder_x)
        adjustment = self.FF(attention)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjusted_output = adjustment + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return output
    
class Block(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
        self.cross_a = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.attention_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        attention = self.cross_a(encoder_x, decoder_x)
        attention_w_skip = attention + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
        
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        adjusted_output = adjust_add_normed + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        # output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return adjusted_output
class DeeperModel(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.proposed_model1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.proposed_model2 = Block(self.proposed_model1.attention_dim,
                                     self.proposed_model1.attention_dim)
        self.proposed_model3 = ProposedModel(self.proposed_model2.attention_dim,
                                             self.proposed_model2.attention_dim)
    
    def forward(self, encoder_x, decoder_x):
        out = self.proposed_model1(encoder_x, decoder_x)
        out =  self.proposed_model2(out, out)
        out = self.proposed_model3(out, out)
        return out
        
        
# class Test_skip_norm_model(nn.Module):
    
#     def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
#         super().__init__()
#         self.proposed_model1 = Block(encoder_dim, decoder_dim, attention_dim)
#         self.proposed_model2 = Block(self.proposed_model1.attention_dim,
#                                      self.proposed_model1.attention_dim)
#         self.proposed_model3 = ProposedModel(self.proposed_model2.attention_dim,
#                                              self.proposed_model2.attention_dim)
    
#     def forward(self, encoder_x, decoder_x):
#         out = self.proposed_model1(encoder_x, decoder_x)
#         out =  self.proposed_model2(out, out)
#         out = self.proposed_model3(out, out)
#         return out
        
