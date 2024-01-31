'''
手撕 transformer 
https://blog.csdn.net/qq_37418807/article/details/120302612
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from IPython.display import Image
seaborn.set_context(context = 'talk')

def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    '''
    (1) input embedding
    输入:[batch_size,len]
    输出:[batch_size,len,embedding_dim]
    '''
    def __init__(self,d_model,vocab):
        super(Embeddings,self).__init__()
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model
    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)
emb = Embeddings(10,1000)
print(emb(torch.Tensor([[1,2,3],[4,5,6]]).long()).shape)  # torch.Size([2, 3, 10])


class PositionalEncoding(nn.Module):
    '''
    (2) positional encoding
    输入:[batch_size,len,embedding_dim]
    输出:[batch_size,len,embedding_dim]
    '''
    def __init__(self,d_model,dropout,max_len = 5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len,d_model)    # [max_len,d_model]
        position = torch.arange(0.,max_len).unsqueeze(1)    # [max_len,1]
        div_term1 = torch.exp(torch.arange(0.,d_model,2) * -(math.log(10000.0) / d_model))    # [d_model // 2]
        div_term2 = torch.exp(torch.arange(1.,d_model,2) * -(math.log(10000.0) / d_model))    # [d_model // 2]
        pe[:,0::2] = torch.sin(position * div_term1)    # [max_len,d_model // 2]
        pe[:,1::2] = torch.cos(position * div_term2)    # [max_len,d_model // 2]
        pe = pe.unsqueeze(0)  # [1,max_len,d_model // 2]
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad = True)
        return self.dropout(x)
plt.figure(figsize = (15,5))
pe = PositionalEncoding(19,0)
y = pe.forward(Variable(torch.zeros(1,100,19)))
print(y.shape)
plt.plot(np.arange(100),y[0,:,4:8].data.numpy())
plt.legend(['dim %d' % p for p in [4,5,6,7]])


def attention(query,key,value,mask = None,dropout = None):
    '''
    (1) attention操作
    输入:[batch_size,num_heads,max_length,dim]
    输出:[batch_size,num_heads,max_length,dim]
    '''
    # shape:query = key = value ---->[batch_size,8,max_length,64]
    d_k = query.size(-1)
    # k(after change):[batch,8,64,max_length],scores:[batch_size,8,max_length,max_length]
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0,-1e9)
    p_attn = F.softmax(scores,dim = -1)    # [batch,n_head,l,l]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value),p_attn   # [batch,n_head,l,model_dim],[batch,n_head,l,l]
q = k = v = torch.ones([4,8,10,100])
attention(q,k,v)[0].shape  # torch.Size([4, 8, 10, 100])


class MultiHeadedAttention(nn.Module):
    '''
    (2) MultiHeadedAttention
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,h,d_model,dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask = None):
        # shape:query = key = value ----> :[batch_size,max_length,embedding_dim = 512]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 将q,k,v分别与Wq,Wk,Wv矩阵进行相乘
        # Wq = Wk = Wv ---->[512,512]
        # 切分：shape:[batch_Size,max_length,8,64]
        # change dim:填充到第一个维度shape:[batch_size,8,max_length,64]
        query,key,value = [l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key,value))]
        x,self.attn = attention(query,key,value,mask = mask,dropout = self.dropout)    # x:[batch,8,max_length,64]
        x = x.transpose(1,2).contiguous().view(nbatches,-1,self.h * self.d_k)    # [batch,max_length,d_model]
        return self.linears[-1](x)
m = MultiHeadedAttention(4,100)
q = k = v = torch.ones([5,10,100])
print(m(q,k,v).shape)  # torch.Size([5, 10, 100])


class SublayerConnection(nn.Module):
    '''
    3.SublayerConnection
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    '''
    (1) EncoderLayer
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,size,self_attn,feed_forward,dropout=0.2):
        super(EncoderLayer,self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size
    def forward(self,x,mask = None):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))  
        return self.sublayer[1](x,self.feed_forward)
self_attn = MultiHeadedAttention(4,100)
size = 100
feed_forward = nn.Linear(100,100)
dropout = 0.2
encoderLayer = EncoderLayer(size,self_attn,feed_forward,dropout)
x = torch.ones(5,10,100)
print(encoderLayer(x).shape)


class Encoder(nn.Module):
    '''
    (2) Encoder
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,mask = None):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
self_attn = MultiHeadedAttention(4,100)
feed_forward = nn.Linear(100,100)
layer = EncoderLayer(size,self_attn,feed_forward)
encoder = Encoder(layer,6)
x = torch.ones(5,10,100)
print(encoder(x).shape)


class DecoderLayer(nn.Module):
    '''
    (1) DecoderLayer
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout = 0.2):
        super(DecoderLayer,self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),3)
    def forward(self,x,memory,src_mask = None,tgt_mask = None):
        m = memory
        # q = k = v
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        # q != k = v
        x = self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)
self_attn = MultiHeadedAttention(4,100)
src_attn = MultiHeadedAttention(10,100)
feed_forward = nn.Linear(100,100)
decoderLayer = DecoderLayer(100,self_attn,src_attn,feed_forward)
x = torch.ones(5,10,100)
memory = torch.ones(5,10,100)
print(decoderLayer(x,memory).shape)


class Decoder(nn.Module):
    '''
    (2) Decoder
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,embedding_dim]
    '''
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,src_mask = None,tgt_mask = None):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)
self_attn = MultiHeadedAttention(4,100)
src_attn = MultiHeadedAttention(10,100)
decoderLayer = DecoderLayer(100,self_attn,src_attn,feed_forward)
decoder = Decoder(decoderLayer,6)
print(decoder(torch.ones(4,10,100),torch.ones(4,10,100)).shape)


class Generator(nn.Module):
    '''
    (1) Generator
    输入:[batch_size,max_len,embedding_dim]
    输出:[batch_size,max_len,vocab_size]
    '''
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,vocab)
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim = -1)


class EncoderDecoder(nn.Module):
    '''
    (2) EncoderDecoder
    输入:
    src(input): [batch_size,src_len]
    tgt(output): [batch_size,tgt_len]
    输出:
    [batch_size,tgt_len]
    '''
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def encode(self,src,src_mask = None):
        return self.encoder(self.src_embed(src),src_mask)
    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    def forward(self,src,tgt,src_mask = None,tgt_mask = None):
        return self.generator(self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask))
self_attn = MultiHeadedAttention(4,100)
src_attn = MultiHeadedAttention(10,100)
feed_forward = nn.Linear(100,100)
layer = EncoderLayer(100,self_attn,feed_forward)
encoder = Encoder(layer,6)
decoderLayer = DecoderLayer(100,self_attn,src_attn,feed_forward)
decoder = Decoder(decoderLayer,6)
src_embed = nn.Embedding(1000,100)
tgt_embed = nn.Embedding(1000,100)
generator = Generator(100,1000)
encoderDecoder = EncoderDecoder(encoder,decoder,src_embed,tgt_embed,generator)
src = torch.ones(5,10).long()
tgt = torch.ones(5,15).long()
print(encoderDecoder(src,tgt).shape)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
           因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度. 
           最后一个是dropout置0比率."""
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn的Dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出"""
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))


'''
训练
'''
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
class EncodedDecoded_Dataset(Dataset):
    def __init__(self,encoded_X,decoded_X,decoded_Y,pad):
        self.pad = pad
        self.encoded_X = encoded_X
        self.decoded_X = decoded_X
        self.decoded_Y = decoded_Y
        assert decoded_X.shape[0] == decoded_Y.shape[0]
    def __len__(self):
        return encoded_X.shape[0]
    def __getitem__(self,index):
        encoded_x = self.encoded_X[index]
        decoded_x = self.decoded_X[index]
        decoded_y = self.decoded_Y[index]
        decoded_mask = self.make_std_mask(decoded_x,self.pad)
        ntokens = (decoded_y != self.pad).data.sum()
        return {'encoded_X':encoded_x,'decoded_X':decoded_x,
                'decoded_Y':decoded_y,'decoded_mask':decoded_mask,'ntokens':ntokens}
    def make_std_mask(self,tgt,pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    def subsequent_mask(self,size):
        attn_shape = (1,size,size)
        subsequent_mask = np.triu(np.ones(attn_shape),k = 1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
V = 10
num = 100
encoded_X = torch.from_numpy(np.random.randint(1,V,size = (num,30)))
#encoded_X[:,0] = 1
target = copy.deepcopy(encoded_X)
#encoded_X = torch.ones(num,100).long()
#target = torch.ones(num,19).long()
decoded_X = target[:,:-1]
decoded_Y = target[:,1:]
decoded_Y[:,0] = 1
dataset = EncodedDecoded_Dataset(encoded_X,decoded_X,decoded_Y,pad = 0)
dataloader = DataLoader(dataset,batch_size = 10)

# 定义模型
h = 8
N = 6
d_model = 512
d_ff = 2048
dropout = 0.1
src_vocab = 10
tgt_vocab = 10
c = copy.deepcopy
attn = MultiHeadedAttention(h,d_model)
ff = PositionwiseFeedForward(d_model,d_ff,dropout)
position = PositionalEncoding(d_model,dropout)

model = EncoderDecoder(
    Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
    Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
    nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
    nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
    Generator(d_model,tgt_vocab)
)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.1,betas = (0.9,0.98),eps = 1e-9)
iter_ = list(iter(dataloader))[0]
#model(iter_['encoded_X'],iter_['decoded_Y']).shape
# train
total_tokens = 0
compute_loss = nn.CrossEntropyLoss()
for _ in range(1):
    with tqdm(dataloader) as board:
        for i,batch in enumerate(board):
            out = model(batch['encoded_X'],batch['decoded_X'])
            print(out.shape)
            loss = compute_loss(out.view(-1,out.shape[-1]),batch['decoded_Y'].view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            board.set_description(str(loss.data))
            board.close()


'''
预测
'''
def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),k = 1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
def greedy_decode(model,src,src_mask,max_len,start_symbol):
    memory = model.encode(src,src_mask)
    ys = torch.ones(1,1).fill_(start_symbol).type_as(src.data)
    #print(subsequent_mask(ys.size(1)).shape)
    for i in range(max_len - 1):
        #print(subsequent_mask(ys.size(1)).shape)
        out = model.decode(memory,src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        #print(out)
        prob = model.generator(out[:,-1])
        #print(prob)
        _,next_word = torch.max(prob,dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,torch.ones(1,1).type_as(src.data).fill_(next_word)],dim = 1)
        print('ys:'+str(ys))
    return ys
model.eval()
src = torch.Tensor([[9,2,2,1,4]]).long()
src_mask = Variable(torch.ones(1, 1, 5))
# print("ys:"+str(ys))
print(greedy_decode(model, src, src_mask, max_len=5, start_symbol=0))
