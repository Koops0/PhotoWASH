# UNet paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the UNet model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25

import math
import jax
import jax.numpy as jnp
from flax import nnx, linen

#Global RNGX
RNG = nnx.Rngs(0)

#This is a function from a previous paper about diffusion models.
#Paper: Denoising Diffusion Probabilistic Models: From Fairseq
#This creates sinusoidal embeddings for the timesteps, but in JAX. Not torch.
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        # Padding
        emb = jnp.concatenate([emb, jnp.zeros((timesteps.shape[0], 1), device=timesteps.device)], axis=1)
    return emb

def nonlinearity(x):
    return x*jax.nn.sigmoid(x)

def normalize(in_c):
    return linen.GroupNorm(
        num_groups=32,
        epsilon=1e-6,
        axis=-1,
        use_bias=False,
    )(in_c)

class Upsample(nnx.Module):
    def __init__(self, in_c, with_conv):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(3,3),
            rngs=RNG,
        )


    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, shape=(B, H*2, W*2, C), method='bilinear')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nnx.Module):
    def __init__(self, in_c, with_conv):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(3,3),
            strides=2,
            padding='VALID',
            rngs=RNG,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        
        if self.with_conv:
            #Use custom padding
            pad = (0, 1, 0, 1)
            x = jnp.pad(x, pad_width=pad, mode='constant', constant_values=0)
            x = self.conv(x)
        else:
            #Use average pooling in 2 dimensions
            x = jax.image.resize(x, shape=(B, H//2, W//2, C), method='average')
        return x

class ResBlock(nnx.Module):
    def __init__(self, *, in_c, out_c=None, conv_shortcut=False, dropout, temb_c=512):
        super(ResBlock, self).__init__()

        #Init
        self.in_c = in_c
        out_c = in_c if out_c is None else out_c
        self.out_c = out_c
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_c)
        self.conv1 = nnx.Conv(
            in_features=in_c,
            out_features=out_c,
            kernel_size=(3,3),
            rngs=RNG,
        )
        self.temb_proj = nnx.Linear(
            in_features=temb_c,
            out_features=out_c,
            rngs=RNG,
        )
        self.norm2 = normalize(out_c)
        self.dropout = nnx.Dropout(dropout)
        self.conv2 = nnx.Conv(
            in_features=out_c,
            out_features=out_c,
            kernel_size=(3,3),
            rngs=RNG,
        )

        #if in_c != out_c:
        if self.in_c != out_c:
            if self.use_conv_shortcut:
                self.shortcut = nnx.Conv(
                    in_features=in_c,
                    out_features=out_c,
                    kernel_size=(3,3),
                    rngs=RNG,
                )
            else:
                self.shortcut = nnx.Conv(
                    in_features=in_c,
                    out_features=out_c,
                    kernel_size=(1,1),
                    padding='VALID',
                    rngs=RNG,
                )
    
    def __call__(self, x, temb):
        #Layer 1
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        #Apply temb
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        #Layer 2
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        #Apply shortcut
        x = self.shortcut(x)

        #Result is x + h
        return h + x
    
# This is the attention block, which is used in the UNet model.
class AttnBlock(nnx.Module):
    def __init__(self, in_c):
        super(AttnBlock, self).__init__()
        self.in_c = in_c

        self.norm1 = normalize(in_c)
        self.q = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(1,1),
            padding='VALID',
            rngs=RNG,
        )
        self.k = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(1,1),
            padding='VALID',
            rngs=RNG,
        )
        self.v = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(1,1),
            padding='VALID',
            rngs=RNG,
        )
        self.proj_out = nnx.Conv(
            in_features=in_c,
            out_features=in_c,
            kernel_size=(1,1),
            padding='VALID',
            rngs=RNG,
        )
    
    def __call__(self, x):
        h_ = x
        h = self.norm1(h_)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Now, we can compute the attention, via B(the batch size), 
        # H (the height), W (the width), and C (the channels)
        B, H, W, C = h.shape
        q = jnp.reshape(q, (B, C, H*W))
        q = jnp.transpose(q, (0, 2, 1))
        k = jnp.reshape(k, (B, C, H*W))
        w_ = jax.lax.batch_matmul(q, k)
        w_ = w_ * (int(C)**(-0.5))
        w_ = nnx.softmax(w_, axis=2)

        # values
        v = jnp.reshape(v, (B, C, H*W))
        w_ = jnp.transpose(w_, (0, 2, 1))
        h_ = jax.lax.batch_matmul(v, w_)
        h_ = jnp.reshape(h_, (B, C, H, W))
        h_ = self.proj_out(h_)

        return h_ + x
    
# This is the main UNet model, which is used in the diffusion model.
class DiffusionUNet(nnx.Module):
    def __init__(self, config):
        super(DiffusionUNet, self).__init__()
        self.config = config

        #Get the config (Taken from paper)
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        #Timestep embedding with JAX
        self.temb = nnx.Module()
        self.temb.dense = [
            nnx.Linear(
                in_features=ch,
                out_features=self.temb_ch,
                rngs=RNG,
            ),
            nnx.Linear(
                in_features=self.temb_ch,
                out_features=self.temb_ch,
                rngs=RNG,
            )
        ]

        #Downsampling
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.ch,
            kernel_size=(3,3),
            rngs=RNG,
        )

        in_ch_mult = (1,)+ch_mult
        self.down = []
        block_in = None

        # For loop that creates the downsampling blocks (pyramid)
        for i_l in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch*in_ch_mult[i_l]
            block_out = ch*ch_mult[i_l]

            #Another loop that appends resnet blocks
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(
                    in_c=block_in,
                    out_c=block_out,
                    dropout=dropout,
                    temb_c=self.temb_ch,
                ))
                block_in = block_out

                # On the 3rd level (==2), we append the attention block
                if i_l == 2:
                    attn.append(AttnBlock(block_out))
            
            # Now, we append the downsampling block
            down = nnx.Module()
            down.block = block
            down.attn = attn

            if i_l != self.num_resolutions - 1:
                down.downsample = Downsample(block_out,resamp_with_conv)
            self.down.append(down)
        
        # Middle block
        self.mid = nnx.Module()
        self.mid.block1 = ResBlock(
            in_c=block_in,
            out_c=block_in,
            dropout=dropout,
            temb_c=self.temb_ch,
        )
        self.mid.attn = AttnBlock(block_in)
        self.mid.block2 = ResBlock(
            in_c=block_in,
            out_c=block_in,
            dropout=dropout,
            temb_c=self.temb_ch,
        )

        # Upsampling
        self.up = []

        # For loop that creates the downsampling blocks (pyramid)
        for i_l in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch*ch_mult[i_l]
            skip_in = ch*ch_mult[i_l]

            #Another loop that appends resnet blocks
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_l]
                block.append(ResBlock(
                    in_c=block_in+skip_in,
                    out_c=block_out,
                    dropout=dropout,
                    temb_c=self.temb_ch,
                ))
                block_in = block_out

                # On the 3rd level (==2), we append the attention block
                if i_l == 2:
                    attn.append(AttnBlock(block_out))
            
            # Now, we append the upsampling block
            up = nnx.Module()
            up.block = block
            up.attn = attn

            if i_l != 0:
                up.upsample = Upsample(block_out, resamp_with_conv)
            self.up.insert(0, up)

        # Final conv
        self.norm_out = normalize(block_out)
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=out_ch,
            kernel_size=(3,3),
            rngs=RNG,
        )

    # From the paper code. 
    def __call__(self, x, t):
        # Get the timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block1(h, temb)
        h = self.mid.attn(h)
        h = self.mid.block2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    jnp.concatenate([h, hs.pop()], axis=-1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

