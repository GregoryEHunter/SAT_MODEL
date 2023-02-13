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

class CrossAttentionSingle(nn.Module): # a selfmade crossattention module
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
    
    
class ProposedModel(nn.Module): # a very basic intial attempt at the proposed model from the midterm
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
        # print(attention.shape)
        # print(decoder_x.shape)
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
    
class BlockFixedSkipM(nn.Module): # trying different skip layers
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
        # print(attention.shape)
        # print(decoder_x.shape)
        attention_w_skip = attention + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
        
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        # adjusted_output = adjust_add_normed + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        # output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return adjust_add_normed
    
class DeeperModel(nn.Module): # Stacking multiple old models on top of each other
    
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
        
        
class Test_skip_norm_model(nn.Module): # adding add norms etc and testing it
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.block_1(encoder_x, decoder_x)
        out = self.lm_head(out)
    
        return out
        
class DeeperModel_skip(nn.Module): # a deeper model with skips
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = Block(self.block_1.attention_dim,
             self.block_1.attention_dim)
        self.lm_head = lm_head
        
        
    def forward(self, encoder_x, decoder_x):
        out = self.block_1(encoder_x, decoder_x)
        out =  self.block_2(out, out)
        out = self.lm_head(out)
        return out

class WiderModel_skip(nn.Module): # using multiple heads with skips (these are wider than the final model to reduce paramertization)
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_3 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = Block(encoder_dim, decoder_dim, attention_dim)
        self.narrow = nn.Linear(decoder_dim + decoder_dim + decoder_dim + decoder_dim, decoder_dim).to(device)
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, decoder_x)
        out_3 = self.block_3(encoder_x, decoder_x)
        out_4 = self.block_4(encoder_x, decoder_x)
        combined_heads = torch.cat((out_1, out_2, out_3, out_4), 1)
        out = self.narrow(combined_heads)
        out = self.lm_head(out)
        return out
    
class WiderBlock(nn.Module): # making the wider model into blocks in order to stack them
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_3 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = Block(encoder_dim, decoder_dim, attention_dim)
        self.narrow = nn.Linear(decoder_dim + decoder_dim + decoder_dim + decoder_dim, decoder_dim).to(device)
        
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, decoder_x)
        out_3 = self.block_3(encoder_x, decoder_x)
        out_4 = self.block_4(encoder_x, decoder_x)
        combined_heads = torch.cat((out_1, out_2, out_3, out_4), 1)
        out = self.narrow(combined_heads)
        return out

    
class WiderDeeperModel_skip(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = WiderBlock(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = WiderBlock(decoder_dim, decoder_dim, attention_dim)
        self.block_3 = WiderBlock(decoder_dim, decoder_dim, attention_dim)
        self.block_4 = WiderBlock(decoder_dim, decoder_dim, attention_dim)
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(out_1, out_1)
        out_3 = self.block_3(out_2, out_2)
        out_4 = self.block_4(out_3, out_3)
        out = self.lm_head(out_4)
        return out
    
    
class WiderDeeperModel_Alt(nn.Module): # alternating cross attention and self attention
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = WiderBlock(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = WiderBlock(decoder_dim, decoder_dim, attention_dim)
        self.block_3 = WiderBlock(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = WiderBlock(decoder_dim, decoder_dim, attention_dim)
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(out_1, out_1)
        out_3 = self.block_3(encoder_x, out_2)
        out_4 = self.block_4(out_3, out_3)
        out = self.lm_head(out_4)
        return out
    
class WiderDeeperModel_2(nn.Module): # a 2 deep model utilizing the wider self made blocks
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = WiderBlock(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = WiderBlock(encoder_dim, decoder_dim, attention_dim)
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, out_1)
        out = self.lm_head(out_2)
        return out
    
class WiderBlock_8(nn.Module): # 8 wide model not stacked again not using pytorch yet
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_3 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_5 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_6 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_7 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_8 = Block(encoder_dim, decoder_dim, attention_dim)
        self.narrow = nn.Linear(decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim, decoder_dim).to(device)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, decoder_x)
        out_3 = self.block_3(encoder_x, decoder_x)
        out_4 = self.block_4(encoder_x, decoder_x)
        
        out_5 = self.block_5(encoder_x, decoder_x)
        out_6 = self.block_6(encoder_x, decoder_x)
        out_7 = self.block_7(encoder_x, decoder_x)
        out_8 = self.block_8(encoder_x, decoder_x)
        
        combined_heads = torch.cat((out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8), 1)
        out = self.narrow(combined_heads)
        out = self.lm_head(out)
        return out
class WiderBlock_8_AllSkip(nn.Module): # adding skips
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_3 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_5 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_6 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_7 = Block(encoder_dim, decoder_dim, attention_dim)
        self.block_8 = Block(encoder_dim, decoder_dim, attention_dim)
        self.narrow = nn.Linear(decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim, decoder_dim).to(device)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, decoder_x)
        out_3 = self.block_3(encoder_x, decoder_x)
        out_4 = self.block_4(encoder_x, decoder_x)
        
        out_5 = self.block_5(encoder_x, decoder_x)
        out_6 = self.block_6(encoder_x, decoder_x)
        out_7 = self.block_7(encoder_x, decoder_x)
        out_8 = self.block_8(encoder_x, decoder_x)
        
        combined_heads = torch.cat((out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8), 1)
        out = self.narrow(combined_heads)
        out = decoder_x + out
        out = self.lm_head(out)
        return out
    
class WiderBlock_8_FixedM(nn.Module): # different skips that were more toward what i intially thout it would be
    def __init__(self, encoder_dim, decoder_dim, attention_dim = None):
        super().__init__()
        self.block_1 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_2 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_3 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_4 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_5 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_6 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_7 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.block_8 = BlockFixedSkipM(encoder_dim, decoder_dim, attention_dim)
        self.narrow = nn.Linear(decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim + decoder_dim, decoder_dim).to(device)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out_1 = self.block_1(encoder_x, decoder_x)
        out_2 = self.block_2(encoder_x, decoder_x)
        out_3 = self.block_3(encoder_x, decoder_x)
        out_4 = self.block_4(encoder_x, decoder_x)
        
        out_5 = self.block_5(encoder_x, decoder_x)
        out_6 = self.block_6(encoder_x, decoder_x)
        out_7 = self.block_7(encoder_x, decoder_x)
        out_8 = self.block_8(encoder_x, decoder_x)
        
        combined_heads = torch.cat((out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8), 1)
        out = self.narrow(combined_heads)
        # out = decoder_x + out
        out = self.lm_head(out)
        return out

class MultiHeadBlock(nn.Module): # multi head block for stacking with a concatination stradegy again expiementation
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.heads = heads
        self.head_list = []
        # self.layer_norm_list = []
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
            
        #Make heads and put into list
        for head in range(self.heads):
            current_head = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
            self.head_list.append(current_head)
        # WO weights
        self.W_O =  nn.Linear(self.attention_dim*self.heads, self.d_dim).to(device)
        
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        concat_list = []
        for head in self.head_list:
            concat_list.append(head(encoder_x, decoder_x))
        
        concated_heads = torch.cat((concat_list), 1)
        attention = self.W_O(concated_heads)
        
        # print(attention.shape)
        # print(decoder_x.shape)
        
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
        # return adjust_add_normed
class MultiHeadBlockExtraSkip(nn.Module): # testing performance with more skip layers
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.heads = heads
        self.head_list = []
        # self.layer_norm_list = []
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
            
        #Make heads and put into list
        for head in range(self.heads):
            current_head = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
            self.head_list.append(current_head)
        # WO weights
        self.W_O =  nn.Linear(self.attention_dim*self.heads, self.d_dim).to(device)
        
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        concat_list = []
        for head in self.head_list:
            concat_list.append(head(encoder_x, decoder_x) + decoder_x)
        
        concated_heads = torch.cat((concat_list), 1)
        attention = self.W_O(concated_heads)
        
        # print(attention.shape)
        # print(decoder_x.shape)
        
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
    
class MultiHeadModel(nn.Module): # combinding these expiermental blocks
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.MBlock_1 = MultiHeadBlock(encoder_dim, decoder_dim, heads, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.MBlock_1(encoder_x, decoder_x)
        out = self.lm_head(out)
        
        return out
    
    
class MultiHeadBlock_WOFinalSkip(nn.Module): # playing with skips because performance is effected
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.heads = heads
        self.head_list = []
        # self.layer_norm_list = []
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
            
        #Make heads and put into list
        for head in range(self.heads):
            current_head = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
            self.head_list.append(current_head)
        # WO weights
        self.W_O =  nn.Linear(self.attention_dim*self.heads, self.d_dim).to(device)
        
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        concat_list = []
        for head in self.head_list:
            concat_list.append(head(encoder_x, decoder_x))
        
        concated_heads = torch.cat((concat_list), 1)
        attention = self.W_O(concated_heads)
        
        # print(attention.shape)
        # print(decoder_x.shape)
        
        attention_w_skip = attention + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
        
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        #adjusted_output = adjust_add_normed + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        # output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return adjust_add_normed

class MultiHeadBlock_WOAnySkip(nn.Module): # expierment without any skip layers
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.heads = heads
        self.head_list = []
        # self.layer_norm_list = []
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
            
        #Make heads and put into list
        for head in range(self.heads):
            current_head = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
            self.head_list.append(current_head)
        # WO weights
        self.W_O =  nn.Linear(self.attention_dim*self.heads, self.d_dim).to(device)
        
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        concat_list = []
        for head in self.head_list:
            concat_list.append(head(encoder_x, decoder_x))
        
        concated_heads = torch.cat((concat_list), 1)
        attention = self.W_O(concated_heads)
        
        # print(attention.shape)
        # print(decoder_x.shape)
        
        #attention_w_skip = attention + decoder_x
        attention_add_normed = self.layer_norm_1(attention)
        
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        #adjusted_output = adjust_add_normed + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        # output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return adjust_add_normed
class MultiHeadBlock_WTwoSkip(nn.Module): # expierment with two skips identified to be important
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.heads = heads
        self.head_list = []
        # self.layer_norm_list = []
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        if attention_dim is None:
            self.attention_dim = decoder_dim
        else:
            self.attention_dim = attention_dim
            
        #Make heads and put into list
        for head in range(self.heads):
            current_head = CrossAttentionSingle(self.e_dim, self.d_dim, self.attention_dim).to(device)
            self.head_list.append(current_head)
        # WO weights
        self.W_O =  nn.Linear(self.attention_dim*self.heads, self.d_dim).to(device)
        
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        concat_list = []
        for head in self.head_list:
            concat_list.append(head(encoder_x, decoder_x) + decoder_x)
        
        concated_heads = torch.cat((concat_list), 1)
        attention = self.W_O(concated_heads)
        
        # print(attention.shape)
        # print(decoder_x.shape)
        
        attention_w_skip = attention + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
        
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        #adjusted_output = adjust_add_normed + decoder_x
        # ######
        # adjusted_output = decoder_x
        # ######
        # output = self.lm_head(adjusted_output)
        # print(attention.shape)
        # print(adjusted_output.shape)
        # print(output.shape)
        return adjust_add_normed
    
class MultiHeadModel_WOFinalBlockSkip(nn.Module): # playing with skips
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.MBlock_1 = MultiHeadBlock_WOFinalSkip(encoder_dim, decoder_dim, heads, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.MBlock_1(encoder_x, decoder_x)
        out = self.lm_head(out)
        
        return out
class MultiHeadModel_WOAnyBlockSkip(nn.Module): #playing with skips
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.MBlock_1 = MultiHeadBlock_WOAnySkip(encoder_dim, decoder_dim, heads, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.MBlock_1(encoder_x, decoder_x)
        out = self.lm_head(out)
        
        return out
    
class MultiHeadModel_TwoSkip(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.MBlock_1 = MultiHeadBlock_WTwoSkip(encoder_dim, decoder_dim, heads, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.MBlock_1(encoder_x, decoder_x)
        out = self.lm_head(out)
        
        return out
class MultiHeadModel_ExtraSkip(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.MBlock_1 = MultiHeadBlockExtraSkip(encoder_dim, decoder_dim, heads, attention_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        out = self.MBlock_1(encoder_x, decoder_x)
        out = self.lm_head(out)
        
        return out
    
    
    
###################### pytorch implementations ############ now using pytorch for efficiency and to ensure no mistakes in implmentations
class MultiHeadModel_PyTorch(nn.Module): # a multi headed version of the SAT model with skips and using pytorchs multiheadattention
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        # print(attention)
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        #adjusted_output = adjust_add_normed + decoder_x # final skip layer intially proposed but not used

        out = self.lm_head(adjust_add_normed)
        
        return out
    
###################### pytorch implementations
class MultiHeadModel_PyTorch(nn.Module): # redefiniton with minor change to skip layer now commented out
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.5, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        # print(attention)
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        # adjusted_output = adjust_add_normed + decoder_x # TRYING ON GATSBY ORGINS

        out = self.lm_head(adjust_add_normed)
        
        # out = self.lm_head(adjusted_output)
        
        return out
    
    
# START: COPIED with edits FROM <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html>
 
class PositionalEncoding(nn.Module): # a positional ecoding I found that I thought may be helpful

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x): # invetibale batch error
        # print(x.shape)
        # print(self.pe[:, :x.size(1)].shape)
        # print(self.pe[:, :x.size(0)].squeeze(0).shape)
        x = x + self.pe[:, :x.size(0)].squeeze(0)
        return x
    
 # END: COPIED and editted FROM <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html>    
    
    
    
    
    
    
class MultiHeadModel_PyTorch_Positional(nn.Module): # one SAT cross attentionw with positiona embeddings.
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        # print(self.d_dim/self.heads)
        self.positional_encoding = PositionalEncoding(d_model=int(self.d_dim))
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.2, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        self.lm_head = lm_head
    def forward(self, encoder_x, decoder_x):
        decoder_x = self.positional_encoding(decoder_x)
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        # print(attention)
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        
        #adjusted_output = adjust_add_normed + decoder_x

        out = self.lm_head(adjust_add_normed)
        
        return out
    
    
    
    
    
    
class MultiHeadModel_PyTorch_Stacked_One_Alt(nn.Module): # attempting to use self attention as well
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        # print(self.d_dim/self.heads)
        self.positional_encoding = PositionalEncoding(d_model=int(self.d_dim))
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        
        self.Multi_Head_Cross_Attention_2 = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.d_dim, vdim=self.d_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_3 = nn.LayerNorm(self.d_dim)
        self.FF3 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu_2 = nn.ReLU().to(device)
        self.FF4 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_4 = nn.LayerNorm(self.d_dim)
        
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        decoder_x = self.positional_encoding(decoder_x)
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        # ----------------------------------------------------
        
        attention_2 = self.Multi_Head_Cross_Attention_2(adjust_add_normed, adjust_add_normed, adjust_add_normed,  need_weights=False)
        attention_w_skip_2 = attention_2[0] + adjust_add_normed
        attention_add_normed_2 = self.layer_norm_3(attention_w_skip_2)
        
        adjustment_2 = self.FF3(attention_add_normed_2)
        non_lin_adjustment_2 = self.Relu_2(adjustment_2)
        adjustment_2 = self.FF4(non_lin_adjustment_2)
        
        
        adjust_w_skip_2 = adjustment_2 + attention_add_normed_2
        adjust_add_normed_2 =  self.layer_norm_4(adjust_w_skip_2)
        
        # out = adjust_add_normed_2 + decoder_x
        out = self.lm_head(adjust_add_normed_2)

        # out = self.lm_head(out)

        return out
    
    
    
class MultiHeadModel_PyTorch_Stacked(nn.Module): # two SAT cross attention modules stacked
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        # print(self.d_dim/self.heads)
        #self.positional_encoding = PositionalEncoding(d_model=int(self.d_dim))
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        
        self.Multi_Head_Cross_Attention_2 =  torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_3 = nn.LayerNorm(self.d_dim)
        self.FF3 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu_2 = nn.ReLU().to(device)
        self.FF4 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_4 = nn.LayerNorm(self.d_dim)
        
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        #decoder_x = self.positional_encoding(decoder_x)
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        # ----------------------------------------------------
        
        attention_2 = self.Multi_Head_Cross_Attention_2(adjust_add_normed, encoder_x, encoder_x,  need_weights=False)
        attention_w_skip_2 = attention_2[0] + adjust_add_normed
        attention_add_normed_2 = self.layer_norm_3(attention_w_skip_2)
        
        adjustment_2 = self.FF3(attention_add_normed_2)
        non_lin_adjustment_2 = self.Relu_2(adjustment_2)
        adjustment_2 = self.FF4(non_lin_adjustment_2)
        
        
        adjust_w_skip_2 = adjustment_2 + attention_add_normed_2
        adjust_add_normed_2 =  self.layer_norm_4(adjust_w_skip_2)
        
        # out = adjust_add_normed_2 + decoder_x
        out = self.lm_head(adjust_add_normed_2)

        # out = self.lm_head(out)

        return out
    
class MultiHeadModel_PyTorch_Stacked_Positional(nn.Module): # **model used in paper!!!** two SAT cross attention modules stacked
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        # print(self.d_dim/self.heads)
        self.positional_encoding = PositionalEncoding(d_model=int(self.d_dim)) # adding positional encoding
        # below is the cross attention head the norm layers and the feed forward module (position-wise)
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        # second SAT cross attention module
        self.Multi_Head_Cross_Attention_2 =  torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_3 = nn.LayerNorm(self.d_dim)
        self.FF3 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu_2 = nn.ReLU().to(device)
        self.FF4 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_4 = nn.LayerNorm(self.d_dim)
        # GPT LM head used but frozen at top of file
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        decoder_x = self.positional_encoding(decoder_x)
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        # ----------------------------------------------------
        
        attention_2 = self.Multi_Head_Cross_Attention_2(adjust_add_normed, encoder_x, encoder_x,  need_weights=False)
        attention_w_skip_2 = attention_2[0] + adjust_add_normed
        attention_add_normed_2 = self.layer_norm_3(attention_w_skip_2)
        
        adjustment_2 = self.FF3(attention_add_normed_2)
        non_lin_adjustment_2 = self.Relu_2(adjustment_2)
        adjustment_2 = self.FF4(non_lin_adjustment_2)
        
        
        adjust_w_skip_2 = adjustment_2 + attention_add_normed_2
        adjust_add_normed_2 =  self.layer_norm_4(adjust_w_skip_2)
        
        # out = adjust_add_normed_2 + decoder_x
        out = self.lm_head(adjust_add_normed_2)

        # out = self.lm_head(out)

        return out
    
    
# other attempts below are still being expiermented with and have various levels of performance.
class MultiHeadModel_PyTorch_Stacked_Triple_Alt(nn.Module):
    # def __init__(self, max_length):
    def __init__(self, encoder_dim, decoder_dim, heads = 4, attention_dim = None):
        """
        Part by part feed forward
        """
        super().__init__()
        self.e_dim = encoder_dim
        self.d_dim = decoder_dim
        self.heads = heads
        # print(self.d_dim/self.heads)
        #self.positional_encoding = PositionalEncoding(d_model=int(self.d_dim))
        self.Multi_Head_Cross_Attention = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_1 = nn.LayerNorm(self.d_dim)
        self.FF = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu = nn.ReLU().to(device)
        self.FF2 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.d_dim)
        
        self.Multi_Head_Cross_Attention_2 = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=True, add_zero_attn=False, kdim=self.d_dim, vdim=self.d_dim, batch_first=False, device=device, dtype=None) # added bias kv true...?
        self.layer_norm_3 = nn.LayerNorm(self.d_dim)
        self.FF3 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu_2 = nn.ReLU().to(device)
        self.FF4 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_4 = nn.LayerNorm(self.d_dim)
        
        self.Multi_Head_Cross_Attention_3 = torch.nn.MultiheadAttention(self.d_dim, self.heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=self.e_dim, vdim=self.e_dim, batch_first=False, device=device, dtype=None)
        self.layer_norm_5 = nn.LayerNorm(self.d_dim)
        self.FF5 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.Relu_3 = nn.ReLU().to(device)
        self.FF6 = nn.Linear(self.d_dim, self.d_dim).to(device)
        self.layer_norm_6 = nn.LayerNorm(self.d_dim)
        
        self.lm_head = lm_head
        
    def forward(self, encoder_x, decoder_x):
        #decoder_x = self.positional_encoding(decoder_x)
        attention = self.Multi_Head_Cross_Attention(decoder_x, encoder_x, encoder_x,  need_weights=False)
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True
        attention_w_skip = attention[0] + decoder_x
        attention_add_normed = self.layer_norm_1(attention_w_skip)
    
        adjustment = self.FF(attention_add_normed)
        non_lin_adjustment = self.Relu(adjustment)
        adjustment = self.FF2(non_lin_adjustment)
        
        adjust_w_skip = adjustment + attention_add_normed
        adjust_add_normed =  self.layer_norm_2(adjust_w_skip)
        # ----------------------------------------------------
        
        attention_2 = self.Multi_Head_Cross_Attention_2(adjust_add_normed, adjust_add_normed, adjust_add_normed,  need_weights=False)
        attention_w_skip_2 = attention_2[0] + adjust_add_normed
        attention_add_normed_2 = self.layer_norm_3(attention_w_skip_2)
        
        adjustment_2 = self.FF3(attention_add_normed_2)
        non_lin_adjustment_2 = self.Relu_2(adjustment_2)
        adjustment_2 = self.FF4(non_lin_adjustment_2)
        
        
        adjust_w_skip_2 = adjustment_2 + attention_add_normed_2
        adjust_add_normed_2 =  self.layer_norm_4(adjust_w_skip_2)
        
        # ----------------------------------------------------
        
        attention_3 = self.Multi_Head_Cross_Attention_3(adjust_add_normed_2, encoder_x, encoder_x,  need_weights=False)
        attention_w_skip_3 = attention_3[0] + adjust_add_normed_2
        attention_add_normed_3 = self.layer_norm_5(attention_w_skip_3)
        
        adjustment_3 = self.FF5(attention_add_normed_3)
        non_lin_adjustment_3 = self.Relu_3(adjustment_3)
        adjustment_3 = self.FF6(non_lin_adjustment_3)
        
        
        adjust_w_skip_3 = adjustment_3 + attention_add_normed_3
        adjust_add_normed_3 =  self.layer_norm_6(adjust_w_skip_3)
        
        out = adjust_add_normed_3 + decoder_x # something to think about.. very weird performance diff in gen and loss
        # out = self.lm_head(adjust_add_normed_3)

        out = self.lm_head(out)

        return out
    
    
