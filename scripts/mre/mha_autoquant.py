# MRE for MHA Autoquantization
import torch
import torchao

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mha = torch.nn.MultiheadAttention(128, 4)
    def forward(self, x):
        x, _ = self.mha(x, x, x)
        return x
    
model = Model()
model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
inputs = (torch.randn(10, 32, 128),)
out = model(*inputs)
print(out)

"""
ERR: subclass doesn't implement <function multi_head_attention_forward at 0x7a74c74148b0>
Traceback (most recent call last):
  File "$HOME/.conda/envs/cu124/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "$HOME/.conda/envs/cu124/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "$WORKSPACE/scripts/mre/mha_autoquant.py", line 16, in <module>
    out = model(*inputs)
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1844, in _call_impl
    return inner()
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1769, in inner
    args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torchao/quantization/autoquant.py", line 720, in autoquant_prehook
    real_model.forward(*args, **kwargs)
  File "$WORKSPACE/scripts/mre/mha_autoquant.py", line 10, in forward
    x, _ = self.mha(x, x, x)
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "$HOME/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/activation.py", line 1368, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
TypeError: cannot unpack non-iterable NoneType object
"""