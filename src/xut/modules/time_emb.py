import jax
import jax.numpy as jnp
from flax import nnx

class TimestepEmbedding(nnx.Module):
    def __init__(
        self,
        dim: int,
        max_period: float = 10000.0,
        time_factor: float = 1000.0,
        *,
        rngs: nnx.rnglib.Rngs,
    ):
        self.dim = dim
        self.max_period = max_period
        self.time_factor = time_factor
        self.freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(0, self.dim // 2, dtype=jnp.float32) / (self.dim // 2))[None, :]
        self.proj = nnx.Linear(self.dim, self.dim, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        if t.ndim > 1:
            t = jnp.squeeze(t, -1)
        t = self.time_factor * t
        args = t[:, None] * self.freqs  # (B, dim//2)
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

        # padding
        if self.dim % 2 == 1:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros((embedding.shape[0], 1), dtype=embedding.dtype)], axis=-1
            )

        x = self.proj(embedding)
        x = jax.nn.mish(x)
        return x