import math
import jax
import jax.numpy as jnp
from flax import nnx

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