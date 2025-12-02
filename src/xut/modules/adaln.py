import jax
import jax.numpy as jnp
from flax import linen as nn

from .norm import RMSNorm

class AdaLN(nn.Module):
    dim: int
    y_dim: int
    gate: bool = True
    norm_layer: nn.Module = RMSNorm
    shared: bool = False

    def setup(self):
        self.norm = self.norm_layer()
        if self.shared:
            self.adaln = None
        else:
            self.adaln = nn.Dense(
                self.dim * (2 + int(self.gate)),
                use_bias=True,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )

    def __call__(self, x, y, shared_adaln=None):
        if shared_adaln is None:
            split = jnp.split(self.adaln(y), 2 + int(self.gate), axis=-1)
            scale, shift = split[:2]
            gate = split[2:] if self.gate else [jnp.zeros_like(scale)]
        else:
            scale, shift = shared_adaln[:2]
            gate = shared_adaln[2:] if self.gate else [jnp.zeros_like(scale)]
        normed_x = self.norm(x)
        if x.ndim == 3:
            if scale.ndim == 2:
                scale = scale[:, None, :]
            if shift.ndim == 2:
                shift = shift[:, None, :]
        result = normed_x * (scale + 1.0) + shift
        return result, (gate[0] + 1.0) if self.gate else jnp.ones_like(scale)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    batch_size, dim, y_dim = 2, 16, 32
    x = jax.random.normal(key, (batch_size, dim))
    y = jax.random.normal(key, (batch_size, y_dim))

    # shared=False
    model = AdaLN(dim=dim, y_dim=y_dim, gate=True, shared=False)
    variables = model.init(key, x, y)
    output, gate = model.apply(variables, x, y)
    print("Output:", output.shape, " Gate:", gate.shape)

    # shared=True
    scale = jnp.zeros_like(output)
    shift = jnp.zeros_like(output)
    gate = jnp.zeros_like(output)

    shared_model = AdaLN(dim=dim, y_dim=y_dim, gate=True, shared=True)
    shared_vars = shared_model.init(key, x, y, (scale, shift, gate))
    output2, gate2 = shared_model.apply(shared_vars, x, y, (scale, shift, gate))
    print("Output:", output2.shape, " Gate:", gate2.shape)