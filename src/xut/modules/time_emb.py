import jax
import jax.numpy as jnp
from flax import linen as nn

class TimestepEmbedding(nn.Module):
    dim: int
    max_period: float = 10000.0
    time_factor: float = 1000.0

    def setup(self):
        self.freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(0, self.dim // 2, dtype=jnp.float32) / (self.dim // 2))[None, :]
        self.proj = nn.Dense(self.dim)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        NOTE! (B,) | (B, 1) -> (B, dim)
        """
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

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    B, D = 4, 16
    t = jnp.linspace(0, 1, B)

    model = TimestepEmbedding(dim=D)
    variables = model.init(key, t)
    out = model.apply(variables, t)
    print(out.shape)
    assert out.shape == (4, 16)
