import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor



class PatchEmbeding(nn.Module) :
  def __init__(self,config) :
    super().__init__()
    self.image_size = config["image_size"]
    self.patch_size = config["patch_size"]
    self.num_channels = config["num_channels"]
    self.hidden_size = config["hidden_size"]
    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.projection = nn.Sequential(
        #break down the image in s1 x s2 patches and flat them
        Rearrange('b c (h p1 )(w p2) -> b(h w) (p1 p2 c)',p1 = self.patch_size , p2 = self.patch_size),
        nn.Linear(self.patch_size * self.patch_size*self.num_channels,self.hidden_size)
    )


  def forward(self,x:Tensor):
    return self.projection(x)
  


class Embeddings (nn.Module) :
  def __init__(self,config) :
    super().__init__()
    self.config = config
    self.patch_embeddings = PatchEmbeding(config)
    #add a cls token which is a learnable parameter , it will be use to classify the entire sequence
    self.cls_token = nn.Parameter(torch.randn(1,1,config["hidden_size"]))
    self.positional_embeddings= nn.Parameter(torch.randn(1,self.patch_embeddings.num_patches+1 , config["hidden_size"])) #You are adding 1 to the count of image patches to create the correct number of position embeddings for the entire input sequence.
    self.dropout = nn.Dropout(config["hidden_dropout_prob"])

  def forward(self , x) :
    x = self.patch_embeddings(x)
    batch_size = x.shape[0]
    #expand the cls token to the batch size  , basically copy this single token for each image in the batch
    #(1,1,hidden_size ) --> (batch_size,1,hidden_size)
    #patch embeddings x has a shape of [batch_size, num_patches, hidden_size],
    #you need to make the cls_token have a compatible shape of [batch_size, 1, hidden_size] so you can concatenate them.
    cls_token  = self.cls_token.expand(batch_size , -1,-1)

    x = torch.cat((cls_token, x) , dim = 1)
    positional_embeddings = self.positional_embeddings
    x = x + positional_embeddings
    x = self.dropout(x)
    return x
  

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        affinity_score = torch.matmul(query, key.transpose(-1, -2))

        scale_factor = self.attention_head_size ** 0.5
        affinity_score = affinity_score / scale_factor
       

        attention_prob = nn.functional.softmax(affinity_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        return torch.matmul(attention_prob, value)

class MultiHeadAttention(nn.Module):

  def __init__(self,config) :
    super().__init__()
    self.hidden_size = config["hidden_size"]
    self.num_attention_heads = config["num_attention_heads"]

    self.attention_head_size = self.hidden_size // self.num_attention_heads
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    self.qkv_bias =  config["qkv_bias"]

    self.heads = nn.ModuleList([])

    for _ in range(self.num_attention_heads) :
      head = AttentionHead(
          self.hidden_size ,
          self.attention_head_size ,
          config["attention_probs_dropout_prob"],
          self.qkv_bias
      )
      self.heads.append(head)

    self.output_projection =nn.Linear(self.all_head_size , self.hidden_size)
    self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])



  def forward(self, x ) :
    attention_ouput = torch.cat([head(x) for head in self.heads], dim =-1)
    attention_ouput = self.output_projection(attention_ouput)
    attention_ouput = self.output_dropout(attention_ouput)
    return attention_ouput
  



class MLP(nn.Module) :
  def __init__(self,config) :
    super().__init__()
    self.Layer1 = nn.Linear(config["hidden_size"],config["intermediate_size"])
    self.activation = nn.GELU()
    self.Layer2 = nn.Linear(config["intermediate_size"],config["hidden_size"])
    self.dropout = nn.Dropout(config["hidden_dropout_prob"])

  def forward (self,x) :
    return self.dropout(self.Layer2(self.activation(self.Layer1(x))))
  

class Block(nn.Module) :
  def __init__(self,config) :
    super().__init__()
    self.attention = MultiHeadAttention(config)
    self.Norm1 = nn.LayerNorm(config["hidden_size"])
    self.mlp = MLP(config)
    self.Norm2 = nn.LayerNorm(config["hidden_size"])


  def forward(self,x) :
    attention_ouput = self.attention(self.Norm1(x))
    x = x + attention_ouput

    x = x + self.mlp(self.Norm2(x))
    return x

class Encode (nn.Module):
  def __init__(self,config) :
    super().__init__()
    self.blocks = nn.ModuleList([])

    for _ in range(config["num_hidden_layers"]) :
      block = Block(config)
      self.blocks.append(block)

  def forward(self,x) :
    for block in self.blocks :
      x = block(x)
    return x
  



class VitForClassification(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.image_size = config["image_size"]
    self.hidden_size = config["hidden_size"]
    self.num_classes = config["num_classes"]

    self.embeddings = Embeddings(config)
    self.encoder = Encode(config)
    self.classification_head  =  nn.Linear(self.hidden_size,self.num_classes)
    self.apply(self._init_weights)




  def _init_weights(self, module):
        """Initializes the weights of the model."""
        if isinstance(module, nn.Linear):
            # Uses a truncated normal distribution to initialize weights
            torch.nn.init.trunc_normal_(
                module.weight, std=self.config["initializer_range"]
            )
            if module.bias is not None:
                # Initializes bias to zero
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            # Initializes LayerNorm bias to zero and weight to one
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0 )


  def forward(self, x):
    embedding_out = self.embeddings(x)
    encoder_out = self.encoder(embedding_out)
    cls_token_out = encoder_out[:,0]
    logits = self.classification_head(cls_token_out)
    return logits

