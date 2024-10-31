# https://github.com/pytorch/pytorch/issues/136642 
import torch
import torch.nn as nn
import typing


class NetMRE(nn.Module):
    def forward(self, x: typing.List[torch.Tensor]):
        shapes: typing.List[typing.Tuple[int, int]] = []
        for xi in x:
            shapes.append(xi.shape[2:])

        shapes = torch.as_tensor(shapes, dtype=torch.long)
        return [
            xx[
                0,
                : (shapes[ix, 0] // 2) : (shapes[ix, 1] // 4),
                : (shapes[ix, 1] // 2) : (shapes[ix, 1] // 4),
            ]
            for ix, xx in enumerate(x)
        ]


x = [
    torch.rand(1, 3, 64, 64),
    torch.rand(1, 3, 32, 32),
]

torch._subclasses.fake_tensor.CONSTANT_NUMEL_LIMIT = 8
exported_program: torch.export.ExportedProgram = torch.export.export(NetMRE(), (x,))
print(exported_program)

# Traceback (most recent call last):
#   File "/home/dgcnz/development/amsterdam/edge/scripts/mre/pt2e_const_index.py", line 29, in <module>
#     exported_program: torch.export.ExportedProgram = torch.export.export(NetMRE(), (x,))
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/__init__.py", line 366, in export
#     return _export(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 1014, in wrapper
#     raise e
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 987, in wrapper
#     ep = fn(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/exported_program.py", line 116, in wrapper
#     return fn(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 1964, in _export
#     export_artifact = export_func(  # type: ignore[operator]
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 1235, in _strict_export
#     return _strict_export_lower_to_aten_ir(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 1263, in _strict_export_lower_to_aten_ir
#     gm_torch_level = _export_to_torch_ir(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py", line 565, in _export_to_torch_ir
#     gm_torch_level, _ = torch._dynamo.export(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 1462, in inner
#     result_traced = opt_f(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py", line 487, in _fn
#     return fn(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 1364, in __call__
#     return self._torchdynamo_orig_callable(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 544, in __call__
#     return _compile(
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 964, in _compile
#     guarded_code = compile_inner(code, one_graph, hooks, transform)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 695, in compile_inner
#     return _compile_inner(code, one_graph, hooks, transform)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_utils_internal.py", line 87, in wrapper_function
#     return function(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 728, in _compile_inner
#     out_code = transform_code_object(code, transform)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py", line 1337, in transform_code_object
#     transformations(instructions, code_options)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 229, in _fn
#     return fn(*args, **kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py", line 657, in transform
#     tracer.run()
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 2888, in run
#     super().run()
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1095, in run
#     while self.step():
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1007, in step
#     self.dispatch_table[inst.opcode](self, inst)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 615, in wrapper
#     return inner_fn(self, inst)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1714, in CALL_FUNCTION
#     self.call_function(fn, args, {})
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 942, in call_function
#     self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/functions.py", line 111, in call_function
#     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 948, in inline_user_function_return
#     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 3103, in inline_call
#     return cls.inline_call_(parent, func, args, kwargs)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 3231, in inline_call_
#     tracer.run()
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1095, in run
#     while self.step():
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1007, in step
#     self.dispatch_table[inst.opcode](self, inst)
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py", line 1931, in BUILD_SLICE
#     self.push(SliceVariable(items))
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/lists.py", line 792, in __init__
#     unimplemented("Dynamic slicing on data-dependent value is not supported")
#   File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/exc.py", line 304, in unimplemented
#     raise Unsupported(msg, case_name=case_name)
# torch._dynamo.exc.Unsupported: Dynamic slicing on data-dependent value is not supported
# 
# from user code:
#    File "/home/dgcnz/development/amsterdam/edge/scripts/mre/pt2e_const_index.py", line 13, in forward
#     return [
#   File "/home/dgcnz/development/amsterdam/edge/scripts/mre/pt2e_const_index.py", line 16, in <listcomp>
#     : (shapes[ix, 0] // 2) : (shapes[ix, 1] // 4),
# 
# Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information