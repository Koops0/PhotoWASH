# FGM paper code by Minglong Xue, Jinhong He, Shivakumara Palaiahnakote, Mingliang Zhou
# This is a JAX implementation of the FGM model, which is used in the diffusion model.
# Translated by Kershan A.
# 4/6/25

import jax
import jax.numpy as jnp
from flax import nnx

#Global RNGX
RNG = nnx.Rngs(0)

#This is a dense layer of the fgm nn.
#JAX vs PyTorch is different, forward is __call__ in JAX
class Dense(nnx.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super(Dense, self).__init__()
        # Initialize the convolutional layer
        self.conv = nnx.Conv(
            in_features=in_c,
            out_features=out_c,
            kernel_size=(kernel_size, kernel_size),
            rngs=RNG,
        )
    
    def __call__(self, x):
        # Apply the convolutional layer + ReLU activation
        conv_out = jax.nn.relu(self.conv(x))
        return jnp.concatenate([x, conv_out], axis=-1)

#This is a residual block of the fgm nn.
class ResidualBlock(nnx.Module):
    def __init__(self, in_c, growth_rate, layers, kernel_size=5):
        super(ResidualBlock, self).__init__()

        #Dense layers
        self.layers = [
            Dense(in_c, growth_rate, kernel_size) for i in range(layers)
        ]

        self.lff = nnx.Conv(
            in_features = in_c + growth_rate * layers,
            out_features=growth_rate,
            kernel_size=(1, 1),
            rngs=RNG,
        )

    def __call__(self, x, lrl=True):
        #if local residual learning is enabled, add the input to the output
        if lrl:
            features = self.layers(x) 
            x = x + self.lff(features)
        else:
            x = self.layers(x)
        
        return x

#This is the main fgm model
class FGM(nnx.Module):
    def __init__(self, in_c, out_c, kernel_size=5):
        super(FGM, self).__init__()
        self.conv1 = nnx.Conv(
            in_features=in_c,
            out_features=out_c,
            kernel_size=(kernel_size, kernel_size),
            rngs=RNG
        )

        self.res_block1 = nnx.Sequential([
            ResidualBlock(64,64,3),
            nnx.relu
        ])

        self.res_block2 = nnx.Sequential([
            ResidualBlock(64,64,3),
            nnx.relu
        ])

        self.conv_block = nnx.Sequential([
            nnx.Conv(
                in_features=64,
                out_features=64,
                kernel_size=(5,5),
                rngs=RNG
            ),
            nnx.relu
        ])

        self.conv2 = nnx.Conv(
            in_features=out_c,
            out_features=in_c,
            kernel_size=(5,5),
            rngs=RNG
        )
    
    def __call__(self, x):
        y = x 
        o1 = self.conv1(x)
        o2 = self.res_block1(o1)
        o3 = self.res_block2(o2)
        o4 = o1 + o2 + o3
        o5 = self.conv2(o4)
        o6 = self.conv2(o5)

        return y - o6