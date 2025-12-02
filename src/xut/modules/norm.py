import jax
from flax import linen as nn

class RMSNorm(nn.Module):
    eps: float = 1e-6
    offset: float = 0.0

    @nn.compact
    def __call__(self, x):
        norm = nn.RMSNorm(epsilon=self.eps, use_scale=False)
        return norm(x) * (1.0 + self.offset)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    hidden_size = 16
    hidden_states = jax.random.normal(key, hidden_size)
    norm = RMSNorm()
    variables = norm.init(key, hidden_states)
    output = norm.apply(variables, hidden_states)
    print(output)