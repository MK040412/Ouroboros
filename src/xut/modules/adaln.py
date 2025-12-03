import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx import RMSNorm

class AdaLN(nnx.Module):
    def __init__(
        self,
        dim: int,
        y_dim: int,
        *,
        gate: bool = True,
        norm_layer: nnx.Module = RMSNorm,
        shared: bool = False,
        rngs: rnglib.Rngs,
    ):
        self.norm = norm_layer(num_features=dim, rngs=rngs)
        self.gate = gate
        self.shared = shared
        if shared:
            self.adaln = None
        else:
            self.adaln = nnx.Linear(
                y_dim,
                dim * (2 + int(self.gate)),
                use_bias = True,
                bias_init = nnx.initializers.zeros_init(),
                kernel_init = nnx.initializers.zeros_init(),
                rngs = rngs,
            )

    def __call__(self, x, y, shared_adaln=None):
        if shared_adaln is None:
            if self.adaln is None:
                raise ValueError("shared_adaln should not be None")
            out = self.adaln(y)
            scale, shift, *gate = jnp.split(out, 2 + int(self.gate), axis=-1)
        else:
            scale, shift, *gate = shared_adaln

        if x.ndim == 3:
            if scale.ndim == 2:
                scale = scale[:, None, :]
            if shift.ndim == 2:
                shift = shift[:, None, :]
            if gate and gate[0].ndim == 2:
                gate[0] = gate[0][:, None, :]

        normed = self.norm(x)
        result = normed * (scale + 1.0) + shift
        if self.gate:
            g = gate[0] + 1.0
        else:
            g = jnp.ones_like(scale)
        return result, g

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    batch_size, dim, y_dim = 2, 16, 32
    x = jax.random.normal(key, (batch_size, dim))
    y = jax.random.normal(key, (batch_size, y_dim))

    model = AdaLN(dim, y_dim, gate=True, shared=False, rngs=nnx.Rngs(params=42))
    output, gate = model(x, y)
    print(f'{output.shape = }, {gate.shape = }')

    shared_adaln = nnx.Linear(
        y_dim,
        dim * 3,
        use_bias = True,
        bias_init = nnx.initializers.zeros_init(),
        kernel_init = nnx.initializers.zeros_init(),
        rngs = nnx.Rngs(params=43),
    )

    shared = shared_adaln(y)
    scale, shift, *gate = jnp.split(shared, 3, axis=-1)

    model = AdaLN(dim, y_dim, gate=True, shared=True, rngs=nnx.Rngs(params=42))
    output, gate = model(x, y, shared_adaln=(scale, shift, *gate))
    print(f'{output.shape = }, {gate.shape = }')
