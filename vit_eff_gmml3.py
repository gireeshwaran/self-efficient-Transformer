import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_
from itertools import repeat
from utils import get_2d_sincos_pos_embed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attn
        
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., n_splits=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()
        
        self.n_splits = n_splits
        
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.num_patches = num_patches
  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False) #fixed sin cos 
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth-2)]  # stochastic depth decay rule
        self.blocks_local_attn = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-2)])

        self.norm_local = norm_layer(embed_dim)
        
        self.blocks_global_attn = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer)
            for i in range(2)])
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Sequential(*[nn.Linear(2*embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_classes)]) if num_classes > 0 else nn.Identity()

        #Decoder related stuff!!-------------------------------------------------
        self.decoder_embed_dim = embed_dim
        self.decoder_num_heads = num_heads
        self.decoder_depth = 6
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.pos_embed_dec = nn.Parameter(torch.zeros(1, num_patches, self.decoder_embed_dim), requires_grad=False) # fixed sin cos
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        self.decoder_shared = nn.ModuleList([
            Block(dim=self.decoder_embed_dim, num_heads=self.decoder_num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(2)])
        self.norm_shared_decoder = norm_layer(self.decoder_embed_dim)
        
        self.decoder_blocks = nn.ModuleList([
            Block(dim=self.decoder_embed_dim, num_heads=self.decoder_num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(self.decoder_depth-2)])
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
   
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.pos_embed_dec.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed_dec.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def random_masking(self, x, n_splits):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        B, N, D = x.shape  # batch, length, dim        
        noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        x_masked = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        masks = torch.empty(0).to(x[0].device)
        for i in range(n_splits):
            len_keep = N//n_splits
            mask = torch.ones([B, N], device=x.device)
            mask[:, i*len_keep : (i+1)*len_keep] = 0
        
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            masks = torch.cat((masks, mask.unsqueeze(0)))
            
        return x_masked, masks, ids_restore
        
        
    def forward_encoder(self, images):

        x = self.patch_embed(images)  # patch linear embedding
        x = x + self.pos_embed
        
        B, N, D = x.size()
        
        # shuffle patches
        x, mask, ids_restore = self.random_masking(x, self.n_splits)

        #first (depth-2) blocks (local attn)
        x_cls = torch.zeros_like(x)
        x_cls_2 = torch.zeros_like(x)
        recons_images = torch.empty(0).to(x[0].device)
        total_loss = 0.
        for i in range(self.n_splits):
            len_keep = N//self.n_splits
            x_sub = x[:, i*len_keep : (i+1)*len_keep]
            for blk in self.blocks_local_attn:
                x_sub = blk(x_sub)
                
            x_sub = self.norm_local(x_sub)

            #decoder
            x_ = self.decoder_embed(x_sub)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x_.shape[0], ids_restore.shape[1] - x_.shape[1], 1)
            
            
            if i == 0:
                x_ = torch.cat([x_, mask_tokens], dim=1) 
            elif i==(self.n_splits-1):
                x_ = torch.cat([mask_tokens, x_], dim=1) 
            else:
                x_ = torch.cat([mask_tokens[:, 0:i*len_keep], x_, mask_tokens[:, i*len_keep:]], dim=1) 
                
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))  # unshuffle
    
            # add pos embed
            x_ = x_ + self.pos_embed_dec
    
            # apply Transformer blocks
            for b, blk in enumerate(self.decoder_shared):
                x_ = blk(x_)
            
            
            mask_idx = 1 - mask[i].unsqueeze(-1).repeat(1, 1, x_.shape[2])
            
            with torch.no_grad():
                x_cls_2[mask_idx==1] = x_[mask_idx==1]
            x_cls[mask_idx==1] = self.norm_shared_decoder(x_)[mask_idx==1]
            
            for b, blk in enumerate(self.decoder_blocks):
                x_ = blk(x_)
            x_ = self.decoder_norm(x_)
    
            # predictor projection
            x_ = self.decoder_pred(x_)
    
            # remove cls token
            total_loss += self.forward_loss(images, x_, mask[i])
            
            recons_images = torch.cat((recons_images, self.unpatchify(x_).unsqueeze(0)), dim=0)
            
        total_loss /= self.n_splits
        
        with torch.no_grad():
            x_cls_recons = self.decoder_norm(x_cls_2)
    
            # predictor projection
            x_cls_recons = self.decoder_pred(x_cls_recons)
            x_cls_recons = self.unpatchify(x_cls_recons)
    
            
        return x_cls, ids_restore, total_loss, recons_images, mask, x_cls_recons
            
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
        

    def forward(self, x , recHead = False):
        x_sub, ids_restore, rec_loss, recons_images, mask, x_cls_recons = self.forward_encoder(x)
    

        B, N, D = x_sub.size()

        # Add class token 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_sub = torch.cat((cls_tokens, x_sub), dim=1)

        #Last 2 (Global attn)
        for blk in (self.blocks_global_attn):
            x_sub = blk(x_sub)

        #classify
        x_sub = self.norm(x_sub) 
        
        return self.head(torch.cat( (x_sub[:, 0], torch.mean(x_sub[:, 1:], dim=1)), dim=-1)), rec_loss, recons_images, mask, x_cls_recons

    def patchify(self, imgs):
      """
      imgs: (N, 3, H, W)
      x: (N, L, patch_size**2 *3)
      """
      p = self.patch_size
      assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

      h = w = imgs.shape[2] // p
      x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
      x = torch.einsum('nchpwq->nhwpqc', x)
      x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
      return x
      
    def unpatchify(self, x):
      """
      x: (N, L, patch_size**2 *3)
      imgs: (N, 3, H, W)
      """
      p = self.patch_size
      h = w = int(x.shape[1]**.5)
      assert h * w == x.shape[1]
      
      x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
      x = torch.einsum('nhwpqc->nchpwq', x)
      imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
      return imgs



class RECHead(nn.Module):
    def __init__(self, in_dim, in_chans=3, patch_size=16):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, kernel_size=(patch_size, patch_size), 
                                                stride=(patch_size, patch_size))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        
        x_rec = x.transpose(1, 2)
        out_sz = tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))
                
                
        return x_rec

def vit_tiny(img_size, patch_size, drop_rate, drop_path_rate, num_classes, n_splits):
    
    model = VisionTransformer(
            img_size = [img_size], patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, n_splits=n_splits,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            num_classes = num_classes, drop_rate=drop_rate, drop_path_rate=drop_path_rate)

    return model

def vit_small(img_size, patch_size, drop_rate, drop_path_rate, num_classes, n_splits):
    
    model = VisionTransformer(
            img_size = [img_size], patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, n_splits=n_splits,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            num_classes = num_classes, drop_rate=drop_rate, drop_path_rate=drop_path_rate)

    return model