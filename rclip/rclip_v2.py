import sys
sys.path.append('..')

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from clip.clip import tokenize

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = True, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    '''
    input:
        
    output:
        
    '''
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for _ in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask))
              
    def forward(self, x: torch.Tensor):

        for b in self.resblocks:
            x = b(x)
        
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        
        # Vision Transformer
        self.transformer = Transformer(width, 12, heads)
        
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        # tokens
        self.token_1_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_2_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_3_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_4_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_5_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_6_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_7_embedding = nn.Parameter(scale * torch.randn(width))
        self.token_8_embedding = nn.Parameter(scale * torch.randn(width))
        
        # rCLIP
        self.avg = torch.nn.AdaptiveAvgPool1d(1)
        
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=2)

        self.proj_k = nn.Parameter(scale * torch.randn(width, width))
        self.proj_q = nn.Parameter(scale * torch.randn(width, width))
        self.proj_v = nn.Parameter(scale * torch.randn(width, width))
        
        self.proj_k2 = nn.Parameter(scale * torch.randn(width, width))
        self.proj_q2 = nn.Parameter(scale * torch.randn(width, width))
        self.proj_v2 = nn.Parameter(scale * torch.randn(width, width))
        
        self.ln_selection = LayerNorm(width)
        
        self.ln_s1 = LayerNorm(width)
        self.ln_s2 = LayerNorm(width)
        self.ln_s3 = LayerNorm(width)
        self.ln_s4 = LayerNorm(width)
        
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        
        self.proj_cls =  nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_cls_mid =  nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_s1 =  nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_s2 =  nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_s3 =  nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj_s4 =  nn.Parameter(scale * torch.randn(width, output_dim))
        
        # init weights
        self._init_weights()
    
    def _init_weights(self):
        return
    
    def transformer_1(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        for num, block in enumerate(self.transformer.resblocks):
            if num in (0,1,2,3,4,5,6,7): # 0 ~ 7
                x = block(x)
        x = x.permute(1, 0, 2)  #  -> NLD
        return x
    
    def transformer_2(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        for num, block in enumerate(self.transformer.resblocks):
            #if num in (7,8,9,10): # 8 ~ 10
            if num in (8,9,10): # 8 ~ 10
                x = block(x)
        x = x.permute(1, 0, 2)  #  -> NLD
        return x   
    
    def transformer_last(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND
        block = self.transformer.resblocks[-1]
        x = block(x)
        x = x.permute(1, 0, 2)  #  -> NLD
        return x
    
    def select(self, x, select=False, num_select=0, visualization=False):
        # select most relevant tokens
        num_tokens = 8
        x_token_cls = x[:, num_tokens, :] # cls token, (100, 768), global feature
        group_tokens = x[:, 0:num_tokens, :]
        patch_tokens = x[:, num_tokens + 1:, :]
        
        # similarities: 
        if num_select == 0:  
            q = torch.matmul(group_tokens, self.proj_q) # 100, 6, 768 @ 768, 768
            k = torch.matmul(patch_tokens, self.proj_k) # 100, 49, 768 @ 768, 768
            v = torch.matmul(patch_tokens, self.proj_v) # 100, 49, 768 @ 768, 768  m1126#2     
        else:
            q = torch.matmul(group_tokens, self.proj_q2) # 100, 6, 768 @ 768, 768
            k = torch.matmul(patch_tokens, self.proj_k2) # 100, 49, 768 @ 768, 768
            v = torch.matmul(patch_tokens, self.proj_v2) # 100, 49, 768 @ 768, 768  m1126#2               
        
        similarities = torch.bmm(q, k.permute(0,2,1)) # (100, 5, 768) * (100, 768, 49) = (100, 5, 49)
        similarities = similarities/(768**(0.5))
        similarities = gumbel_softmax(similarities, dim=2) # (100, 5, 49), last dim -> one hot 
        
        result = torch.bmm(similarities, v) # 100, 6, 768
        
        result = (result + group_tokens) / 2 
        
        if select == True: 
            x = torch.cat((result, x_token_cls.unsqueeze(1)), dim=1) # 6 + 1

        else:
            x = torch.cat((result, x_token_cls.unsqueeze(1)), dim=1) # 6 + 1
            x = torch.cat((x, patch_tokens), dim=1) # 6 + 1 + 49
        
        # >>>>>>>>>>>> visualization >>>>>>>>>>>>
        if visualization == True:
            print('stage: ', num_select)
            for i in range(len(similarities)):
                temp0 = torch.argmax(similarities[i].squeeze(), dim=1)
                print(temp0)  
            if num_select == 1:
                exit()
            
        return x
        
    def forward(self, x: torch.Tensor, patches=False, visualization=False):
        num_tokens = 8
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2], width=768
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width], (100, 49+1, 768)
        
        x = x + self.positional_embedding.to(x.dtype)
        
        # add selection tokens
        x = torch.cat([self.token_1_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = torch.cat([self.token_2_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = torch.cat([self.token_3_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = torch.cat([self.token_4_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = torch.cat([self.token_5_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = torch.cat([self.token_6_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = torch.cat([self.token_7_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = torch.cat([self.token_8_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        
        x = self.ln_pre(x) # (100, 5 + 1 + 49, 768)
        
        x = self.transformer_1(x) # go through 8 transformer layers
        
        x = self.select(x, visualization=visualization)
        
        x = self.transformer_2(x)
        
        x = self.select(x, select=True, num_select=1, visualization=visualization)
        
        x = self.transformer_last(x) # 100, 6 + 1, 768
        
        x_contra = x[:, num_tokens, :]
        x_contra = self.ln_post(x_contra)
        
        x_select = x[:, 0:num_tokens-1, :]
        x_s1 = x_select[:, 0:2, :]
        x_s2 = x_select[:, 2:4, :]
        x_s3 = x_select[:, 4:6, :]
        x_s4 = x_select[:, 6:8, :]
        x_s1 = self.ln_s1(x_s1)
        x_s2 = self.ln_s2(x_s2)
        x_s3 = self.ln_s3(x_s3)
        x_s4 = self.ln_s4(x_s4)
        
        x_s1 = self.avg(x_s1.permute(0,2,1)).squeeze()
        x_s2 = self.avg(x_s2.permute(0,2,1)).squeeze()
        x_s3 = self.avg(x_s3.permute(0,2,1)).squeeze()
        x_s4 = self.avg(x_s4.permute(0,2,1)).squeeze()
        
        # Proj        
        x_contra = x_contra @ self.proj_cls # 768 -> 512
        x_s1 = x_s1 @ self.proj_s1 # 768 -> 512
        x_s2 = x_s2 @ self.proj_s2 # 768 -> 512
        x_s3 = x_s3 @ self.proj_s3 # 768 -> 512
        x_s4 = x_s4 @ self.proj_s4 # 768 -> 512
        
        if patches == False:
            return x_contra
        else:
            return x_contra, x_s1, x_s2, x_s3, x_s4

class rCLIP(nn.Module):
    def __init__(self, clip=None, model=None, device=None, mlp=None, cg2label=None):
        super().__init__()

        embed_dim=512
        # vision
        image_resolution=224
        vision_layers=12
        vision_width=768
        vision_patch_size=32

        # text
        context_length=77
        vocab_size=49408
        transformer_width=512
        transformer_heads=8
        transformer_layers=12
        self.context_length = context_length
        
        self.device = device
        
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()            
        )
        
        # --------------- Text Transformers ---------------
#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers - 1,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )
#         self.transformer_last = Transformer(
#             width=transformer_width,
#             layers=1,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )
        
#         # tokens
#         self.token_1_embedding = nn.Parameter(scale * torch.randn(width))
#         self.token_2_embedding = nn.Parameter(scale * torch.randn(width))
#         self.token_3_embedding = nn.Parameter(scale * torch.randn(width))
#         self.token_4_embedding = nn.Parameter(scale * torch.randn(width))
#         self.token_5_embedding = nn.Parameter(scale * torch.randn(width))
        
#         # tools
#         self.avg = torch.nn.AdaptiveAvgPool1d(1)
        
#         self.ce = torch.nn.CrossEntropyLoss()
#         self.softmax = torch.nn.Softmax(dim=2)

#         self.proj_k = nn.Parameter(scale * torch.randn(width, width))
#         self.proj_q = nn.Parameter(scale * torch.randn(width, width))
#         self.proj_v = nn.Parameter(scale * torch.randn(width, width))
        
#         self.ln_post_cls = LayerNorm(width)
#         self.ln_mid = LayerNorm(width)
#         self.ln_selection = LayerNorm(width)
        
#         self.proj_cls =  nn.Parameter(scale * torch.randn(width, output_dim))
#         self.proj_cls_mid =  nn.Parameter(scale * torch.randn(width, output_dim))       
        
        # --------------- Text Transformers ---------------
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=2)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, patches=False, visualization=False):
        return self.visual.forward(image.type(self.dtype), patches, visualization=visualization)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype) #(100, 77, 768)
        
        # add selection tokens 
          
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence), EOT is the last one (assigned 49407)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x
    
    def inner_forward(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text        
        
        
    def forward(self, image_input, text_inputs, tags, train=False):
        (t1,t2,t3,t4) = tags
        
        text_features = self.encode_text(text_inputs) # text_features: (100, 512)
        t1 = self.encode_text(t1)
        t2 = self.encode_text(t2)
        t3 = self.encode_text(t3)
        t4 = self.encode_text(t4)
        
        # obtain image features
        x_cls, x_s1, x_s2, x_s3, x_s4 = self.encode_image(image_input, patches=True)
                
        # similarity
        image_features = x_cls
        text_features = text_features

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        
        # normalized features
        image_features =  logit_scale.sqrt() * image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = logit_scale.sqrt() * text_features / text_features.norm(dim=-1, keepdim=True)
        s1_features = logit_scale.sqrt() * x_s1 / x_s1.norm(dim=-1, keepdim=True)
        s2_features = logit_scale.sqrt() * x_s2 / x_s2.norm(dim=-1, keepdim=True)
        s3_features = logit_scale.sqrt() * x_s3 / x_s3.norm(dim=-1, keepdim=True)
        s4_features = logit_scale.sqrt() * x_s4 / x_s4.norm(dim=-1, keepdim=True)
        
        t1_f = logit_scale.sqrt() * t1 / t1.norm(dim=-1, keepdim=True)
        t2_f = logit_scale.sqrt() * t2 / t2.norm(dim=-1, keepdim=True)
        t3_f = logit_scale.sqrt() * t3 / t3.norm(dim=-1, keepdim=True)
        t4_f = logit_scale.sqrt() * t4 / t4.norm(dim=-1, keepdim=True)
        
        if train == False:
            return image_features, text_features
        else: 
            return image_features, text_features, s1_features, s2_features, s3_features, s4_features, t1_f, t2_f, t3_f, t4_f
    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()