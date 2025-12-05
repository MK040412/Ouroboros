# JAX monitoring patch (필수 - orbax 호환성)
import jax
if not hasattr(jax, 'monitoring'):
    class _DummyMonitoring:
        @staticmethod
        def record_scalar(*args, **kwargs):
            pass
    jax.monitoring = _DummyMonitoring()
elif not hasattr(jax.monitoring, 'record_scalar'):
    jax.monitoring.record_scalar = lambda *args, **kwargs: None

from gemma import gm
import numpy as np

text = "a computer"  # 테스트용 텍스트

model = gm.nn.Gemma3_270M()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_PT)
tokenizer = gm.text.Gemma3Tokenizer()

# Tokenize
max_length = 128
tokens = tokenizer.encode(text, add_bos=True)
if len(tokens) > max_length:
    tokens = tokens[:max_length]
else:
    tokens = tokens + [0] * (max_length - len(tokens))

tokens_array = np.array([tokens], dtype=np.int32)

# Forward pass
out = model.apply(
    {'params': params},
    tokens=tokens_array,
    return_last_only=False,
    return_hidden_states=True,
)
last_hidden = out.hidden_states[-1]

print("Text:", text)
print("Last hidden states shape:", last_hidden.shape)
print("Last hidden range:", last_hidden.min(), last_hidden.max())
print(last_hidden)


# Example of multi-turn conversation
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=True,
)

prompt = """I will Talk about black tea in 3 sentences. """
out0 = sampler.chat(prompt)
print(out0)