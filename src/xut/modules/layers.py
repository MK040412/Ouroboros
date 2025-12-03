import jax
import jax.numpy as jnp
from flax import nnx

class SwiGLU(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        *,
        use_bias: bool = True,
        rngs: nnx.rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.w12 = nnx.Linear(self.in_features, 2*self.hidden_features, use_bias=use_bias, rngs=rngs)
        self.w3 = nnx.Linear(self.hidden_features, self.out_features, use_bias=use_bias, rngs=rngs)
    
    def __call__(self, x):
        w12 = self.w12(x)
        x1, x2 = jnp.split(w12, 2, axis=-1)
        return self.w3(nnx.silu(x1) * x2)

if __name__ == '__main__':
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 16, 128))
    model = SwiGLU(128, 256, 128, rngs=nnx.Rngs(params=42))
    output = model(x)
    print(output)