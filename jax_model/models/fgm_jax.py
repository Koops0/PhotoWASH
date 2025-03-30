import jax
import jax.numpy as jnp
from flax import nnx

#This is a dense layer of the fgm nn.
#JAX vs PyTorch is different, forward is __call__ in JAX
class Dense(nnx.Module):
    def __init__(self, out_c, kernel_size=5):
        super(Dense, self).__init__()
        # Initialize the convolutional layer
        self.conv = nnx.Conv(
            features=out_c,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='SAME',
            kernel_init=nnx.initializers.xavier_uniform(),
            use_bias=True,
        )
    
    def __call__(self, x):
        # Apply the convolutional layer + ReLU activation
        conv_out = jax.nn.relu(self.conv(x))
        return jnp.concatenate([x, conv_out], axis=-1)

#This is a residual block of the fgm nn.
class ResidualBlock(nnx.Module):
    def __init__(self, in_c, growth_rate, layers, kernel_size=5):
        super(ResidualBlock, self).__init__()

        #First conv layer. You don't need in_channels because of lazy load
        self.conv1 = nnx.Conv(
            features=64,
            kernel_size=(kernel_size, kernel_size),
            strides=(1, 1),
            padding='SAME',
            use_bias=True,
        )

        #Dense layers
        self.layers = [
            Dense(growth_rate, kernel_size) for i in range(layers)
        ]

        self.lff = nnx.Conv(
            features=growth_rate,
            kernel_size=(1, 1),
            use_bias=True,
        )

    def __call__(self, x, lrl=True):
        #if local residual learning is enabled, add the input to the output
        if lrl:
            x = x + self.lff(self.layers(x))
        else:
            x = self.layers(x)
        
        return x

#This is the main fgm model
class FGM(nnx.Module):
    def init