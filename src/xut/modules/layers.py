import jax
import jax.numpy as jnp
from flax import linen as nn

class SwiGLU(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    bias: bool = True
    _pack_weights: bool = True

    def setup(self):
        hidden_features = self.hidden_features or self.in_features
        out_features = self.out_features or self.in_features
        if self._pack_weights:
            self.w12 = nn.Dense(2 * hidden_features, use_bias=self.bias)
        else:
            self.w1 = nn.Dense(hidden_features, use_bias=self.bias)
            self.w2 = nn.Dense(hidden_features, use_bias=self.bias)
        self.w3 = nn.Dense(out_features, use_bias=self.bias)

    def __call__(self, x):
        if self._pack_weights:
            w12 = self.w12(x)
            x1, x2 = jnp.split(w12, 2, axis=-1)
        else:
            x1 = self.w1(x)
            x2 = self.w2(x)
        return self.w3(nn.silu(x1) * x2)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 16, 128))
    model = SwiGLU(128, 256, 128)
    variables = model.init(key, x)
    output = model.apply(variables, x)
    print(output)