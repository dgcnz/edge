import torch
import torch.nn as nn
import typing


class NetMRE(nn.Module):
    def forward(self, x: typing.List[torch.Tensor]):
        spatial_shapes: typing.List[typing.Tuple[int, int]] = []
        for xi in x:
            spatial_shapes.append(xi.shape[2:])

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
        reference_points_list = []
        for H, W in spatial_shapes:
            lin = torch.linspace(0.5, H.item() - 0.5, H.item())
            reference_points_list.append(lin)
        return reference_points_list


x = [
    torch.rand(1, 3, 64, 64),
    torch.rand(1, 3, 32, 32),
]

torch._subclasses.fake_tensor.CONSTANT_NUMEL_LIMIT = 8
exported_program: torch.export.ExportedProgram = torch.export.export(NetMRE(), (x,))
print(exported_program)

# E1025 19:03:46.483248 10043 site-packages/torch/export/_trace.py:1000] See unsupported_operator in exportdb for unsupported case.                 https://pytorch.org/docs/main/generated/exportdb/index.html#unsupported-operator
# {
# 	"name": "Unsupported",
# 	"message": "torch.* op returned non-Tensor int call_method item
# 
# from user code:
#    File \"/tmp/ipykernel_10043/362225648.py\", line 4, in forward
#     reference_points = self.get_reference_points(spatial_shapes)
#   File \"/tmp/ipykernel_10043/362225648.py\", line 13, in get_reference_points
#     H.item()
# 
# Set TORCH_LOGS=\"+dynamo\" and TORCHDYNAMO_VERBOSE=1 for more information
# ",
# 	"stack": "---------------------------------------------------------------------------
# Unsupported                               Traceback (most recent call last)
# Cell In[50], line 3
#       1 torch._subclasses.fake_tensor.CONSTANT_NUMEL_LIMIT = 8
#       2 torch._dynamo.config.automatic_dynamic_shapes =False
# ----> 3 ep: torch.export.ExportedProgram = torch.export.export(NetC3(), (x,))
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/__init__.py:366, in export(mod, args, kwargs, dynamic_shapes, strict, preserve_module_call_signature)
#     360 if isinstance(mod, torch.jit.ScriptModule):
#     361     raise ValueError(
#     362         \"Exporting a ScriptModule is not supported. \"
#     363         \"Maybe try converting your ScriptModule to an ExportedProgram \"
#     364         \"using `TS2EPConverter(mod, args, kwargs).convert()` instead.\"
#     365     )
# --> 366 return _export(
#     367     mod,
#     368     args,
#     369     kwargs,
#     370     dynamic_shapes,
#     371     strict=strict,
#     372     preserve_module_call_signature=preserve_module_call_signature,
#     373     pre_dispatch=True,
#     374 )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:1014, in _log_export_wrapper.<locals>.wrapper(*args, **kwargs)
#    1007     else:
#    1008         log_export_usage(
#    1009             event=\"export.error.unclassified\",
#    1010             type=error_type,
#    1011             message=str(e),
#    1012             flags=_EXPORT_FLAGS,
#    1013         )
# -> 1014     raise e
#    1015 finally:
#    1016     _EXPORT_FLAGS = None
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:987, in _log_export_wrapper.<locals>.wrapper(*args, **kwargs)
#     985 try:
#     986     start = time.time()
# --> 987     ep = fn(*args, **kwargs)
#     988     end = time.time()
#     989     log_export_usage(
#     990         event=\"export.time\",
#     991         metrics=end - start,
#     992         flags=_EXPORT_FLAGS,
#     993         **get_ep_stats(ep),
#     994     )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/exported_program.py:116, in _disable_prexisiting_fake_mode.<locals>.wrapper(*args, **kwargs)
#     113 @functools.wraps(fn)
#     114 def wrapper(*args, **kwargs):
#     115     with unset_fake_temporarily():
# --> 116         return fn(*args, **kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:1964, in _export(mod, args, kwargs, dynamic_shapes, strict, preserve_module_call_signature, pre_dispatch, allow_complex_guards_as_runtime_asserts, _is_torch_jit_trace)
#    1961 # Call the appropriate export function based on the strictness of tracing.
#    1962 export_func = _strict_export if strict else _non_strict_export
# -> 1964 export_artifact = export_func(  # type: ignore[operator]
#    1965     mod,
#    1966     args,
#    1967     kwargs,
#    1968     dynamic_shapes,
#    1969     preserve_module_call_signature,
#    1970     pre_dispatch,
#    1971     original_state_dict,
#    1972     original_in_spec,
#    1973     allow_complex_guards_as_runtime_asserts,
#    1974     _is_torch_jit_trace,
#    1975 )
#    1976 export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
#    1978 forward_arg_names = (
#    1979     _get_forward_arg_names(mod, args, kwargs) if not _is_torch_jit_trace else None
#    1980 )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:1235, in _strict_export(mod, args, kwargs, dynamic_shapes, preserve_module_call_signature, pre_dispatch, original_state_dict, orig_in_spec, allow_complex_guards_as_runtime_asserts, _is_torch_jit_trace)
#    1222 def _strict_export(
#    1223     mod: torch.nn.Module,
#    1224     args: Tuple[Any, ...],
#    (...)
#    1232     _is_torch_jit_trace: bool,
#    1233 ) -> ExportArtifact:
#    1234     lower_to_aten = functools.partial(_export_to_aten_ir, pre_dispatch=pre_dispatch)
# -> 1235     return _strict_export_lower_to_aten_ir(
#    1236         mod=mod,
#    1237         args=args,
#    1238         kwargs=kwargs,
#    1239         dynamic_shapes=dynamic_shapes,
#    1240         preserve_module_call_signature=preserve_module_call_signature,
#    1241         pre_dispatch=pre_dispatch,
#    1242         original_state_dict=original_state_dict,
#    1243         orig_in_spec=orig_in_spec,
#    1244         allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
#    1245         _is_torch_jit_trace=_is_torch_jit_trace,
#    1246         lower_to_aten_callback=lower_to_aten,
#    1247     )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:1263, in _strict_export_lower_to_aten_ir(mod, args, kwargs, dynamic_shapes, preserve_module_call_signature, pre_dispatch, original_state_dict, orig_in_spec, allow_complex_guards_as_runtime_asserts, _is_torch_jit_trace, lower_to_aten_callback)
#    1250 def _strict_export_lower_to_aten_ir(
#    1251     mod: torch.nn.Module,
#    1252     args: Tuple[Any, ...],
#    (...)
#    1261     lower_to_aten_callback: Callable,
#    1262 ) -> ExportArtifact:
# -> 1263     gm_torch_level = _export_to_torch_ir(
#    1264         mod,
#    1265         args,
#    1266         kwargs,
#    1267         dynamic_shapes,
#    1268         preserve_module_call_signature=preserve_module_call_signature,
#    1269         restore_fqn=False,  # don't need to restore because we will do it later
#    1270         allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
#    1271         _log_export_usage=False,
#    1272     )
#    1274     # We detect the fake_mode by looking at gm_torch_level's placeholders, this is the fake_mode created in dynamo.
#    1275     (
#    1276         fake_args,
#    1277         fake_kwargs,
#    1278         dynamo_fake_mode,
#    1279     ) = _extract_fake_inputs(gm_torch_level, args, kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_trace.py:565, in _export_to_torch_ir(f, args, kwargs, dynamic_shapes, preserve_module_call_signature, disable_constraint_solver, allow_complex_guards_as_runtime_asserts, restore_fqn, _log_export_usage, same_signature)
#     561     module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
#     562     with _wrap_submodules(
#     563         f, preserve_module_call_signature, module_call_specs
#     564     ), _ignore_backend_decomps():
# --> 565         gm_torch_level, _ = torch._dynamo.export(
#     566             f,
#     567             dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
#     568             assume_static_by_default=True,
#     569             tracing_mode=\"symbolic\",
#     570             disable_constraint_solver=disable_constraint_solver,
#     571             # currently the following 2 flags are tied together for export purposes,
#     572             # but untangle for sake of dynamo export api
#     573             prefer_deferred_runtime_asserts_over_guards=True,
#     574             allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
#     575             _log_export_usage=_log_export_usage,
#     576             same_signature=same_signature,
#     577         )(
#     578             *args,
#     579             **kwargs,
#     580         )
#     581 except (ConstraintViolationError, ValueRangeError) as e:
#     582     raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:1462, in export.<locals>.inner(*args, **kwargs)
#    1460 # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideeffects and reject.
#    1461 try:
# -> 1462     result_traced = opt_f(*args, **kwargs)
#    1463 except ConstraintViolationError as e:
#    1464     constraint_violation_error = e
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1735 else:
# -> 1736     return self._call_impl(*args, **kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
#    1742 # If we don't have any hooks, we want to skip the rest of the logic in
#    1743 # this function, and just call forward.
#    1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1745         or _global_backward_pre_hooks or _global_backward_hooks
#    1746         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1747     return forward_call(*args, **kwargs)
#    1749 result = None
#    1750 called_always_called_hooks = set()
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py:487, in _TorchDynamoContext.__call__.<locals>._fn(*args, **kwargs)
#     482 saved_dynamic_layer_stack_depth = (
#     483     torch._C._functorch.get_dynamic_layer_stack_depth()
#     484 )
#     486 try:
# --> 487     return fn(*args, **kwargs)
#     488 finally:
#     489     # Restore the dynamic layer stack depth if necessary.
#     490     torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
#     491         saved_dynamic_layer_stack_depth
#     492     )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
#    1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
#    1735 else:
# -> 1736     return self._call_impl(*args, **kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
#    1742 # If we don't have any hooks, we want to skip the rest of the logic in
#    1743 # this function, and just call forward.
#    1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
#    1745         or _global_backward_pre_hooks or _global_backward_hooks
#    1746         or _global_forward_hooks or _global_forward_pre_hooks):
# -> 1747     return forward_call(*args, **kwargs)
#    1749 result = None
#    1750 called_always_called_hooks = set()
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:1364, in CatchErrorsWrapper.__call__(self, frame, cache_entry, frame_state)
#    1358             return hijacked_callback(
#    1359                 frame, cache_entry, self.hooks, frame_state
#    1360             )
#    1362 with compile_lock, _disable_current_modes():
#    1363     # skip=1: skip this frame
# -> 1364     return self._torchdynamo_orig_callable(
#    1365         frame, cache_entry, self.hooks, frame_state, skip=1
#    1366     )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:544, in ConvertFrameAssert.__call__(self, frame, cache_entry, hooks, frame_state, skip)
#     541     info = f\"{code.co_name} {code.co_filename}:{code.co_firstlineno}\"
#     542     dynamo_tls.traced_frame_infos.append(info)
# --> 544 return _compile(
#     545     frame.f_code,
#     546     frame.f_globals,
#     547     frame.f_locals,
#     548     frame.f_builtins,
#     549     self._torchdynamo_orig_callable,
#     550     self._one_graph,
#     551     self._export,
#     552     self._export_constraints,
#     553     hooks,
#     554     cache_entry,
#     555     cache_size,
#     556     frame,
#     557     frame_state=frame_state,
#     558     compile_id=compile_id,
#     559     skip=skip + 1,
#     560 )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:964, in _compile(code, globals, locals, builtins, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
#     962 guarded_code = None
#     963 try:
# --> 964     guarded_code = compile_inner(code, one_graph, hooks, transform)
#     965     return guarded_code
#     966 except Exception as e:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:695, in _compile.<locals>.compile_inner(code, one_graph, hooks, transform)
#     693 with dynamo_timed(\"_compile.compile_inner\", phase_name=\"entire_frame_compile\"):
#     694     with CompileTimeInstructionCounter.record():
# --> 695         return _compile_inner(code, one_graph, hooks, transform)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_utils_internal.py:87, in compile_time_strobelight_meta.<locals>.compile_time_strobelight_meta_inner.<locals>.wrapper_function(*args, **kwargs)
#      84     kwargs[\"skip\"] = kwargs[\"skip\"] + 1
#      86 if not StrobelightCompileTimeProfiler.enabled:
# ---> 87     return function(*args, **kwargs)
#      89 return StrobelightCompileTimeProfiler.profile_compile_time(
#      90     function, phase_name, *args, **kwargs
#      91 )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:728, in _compile.<locals>._compile_inner(code, one_graph, hooks, transform)
#     726 CompileContext.get().attempt = attempt
#     727 try:
# --> 728     out_code = transform_code_object(code, transform)
#     729     break
#     730 except exc.RestartAnalysis as e:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/bytecode_transformation.py:1337, in transform_code_object(code, transformations, safe)
#    1334 instructions = cleaned_instructions(code, safe)
#    1335 propagate_line_nums(instructions)
# -> 1337 transformations(instructions, code_options)
#    1338 return clean_and_assemble_instructions(instructions, keys, code_options)[1]
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:229, in preserve_global_state.<locals>._fn(*args, **kwargs)
#     227 exit_stack.enter_context(torch_function_mode_stack_state_mgr)
#     228 try:
# --> 229     return fn(*args, **kwargs)
#     230 finally:
#     231     cleanup.close()
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/convert_frame.py:657, in _compile.<locals>.transform(instructions, code_options)
#     655 try:
#     656     with tracing(tracer.output.tracing_context), tracer.set_current_tx():
# --> 657         tracer.run()
#     658 except exc.UnspecializeRestartAnalysis:
#     659     speculation_log.clear()
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:2888, in InstructionTranslator.run(self)
#    2887 def run(self):
# -> 2888     super().run()
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1095, in InstructionTranslatorBase.run(self)
#    1093 try:
#    1094     self.output.push_tx(self)
# -> 1095     while self.step():
#    1096         pass
#    1097 except BackendCompilerFailed:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1007, in InstructionTranslatorBase.step(self)
#    1004 self.update_block_stack(inst)
#    1006 try:
# -> 1007     self.dispatch_table[inst.opcode](self, inst)
#    1008     return not self.output.should_exit
#    1009 except exc.ObservedException as e:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:615, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
#     613     return handle_graph_break(self, inst, speculation.reason)
#     614 try:
# --> 615     return inner_fn(self, inst)
#     616 except Unsupported as excp:
#     617     if self.generic_context_manager_depth > 0:
#     618         # We don't support graph break under GenericContextWrappingVariable,
#     619         # If there is, we roll back to the checkpoint and fall back.
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1714, in InstructionTranslatorBase.CALL_FUNCTION(self, inst)
#    1712 args = self.popn(inst.argval)
#    1713 fn = self.pop()
# -> 1714 self.call_function(fn, args, {})
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:942, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
#     940 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
#     941     raise AssertionError(f\"Attempt to trace forbidden callable {inner_fn}\")
# --> 942 self.push(fn.call_function(self, args, kwargs))
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/functions.py:339, in UserFunctionVariable.call_function(self, tx, args, kwargs)
#     337         with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
#     338             return super().call_function(tx, args, kwargs)
# --> 339 return super().call_function(tx, args, kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
#     105 def call_function(
#     106     self,
#     107     tx: \"InstructionTranslator\",
#     108     args: \"List[VariableTracker]\",
#     109     kwargs: \"Dict[str, VariableTracker]\",
#     110 ) -> \"VariableTracker\":
# --> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:948, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
#     944 def inline_user_function_return(self, fn, args, kwargs):
#     945     \"\"\"
#     946     A call to some user defined function by inlining it.
#     947     \"\"\"
# --> 948     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:3103, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
#    3100 @classmethod
#    3101 def inline_call(cls, parent, func, args, kwargs):
#    3102     with patch.dict(counters, {\"unimplemented\": counters[\"inline_call\"]}):
# -> 3103         return cls.inline_call_(parent, func, args, kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:3231, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
#    3229 try:
#    3230     with strict_ctx:
# -> 3231         tracer.run()
#    3232 except exc.ObservedException as e:
#    3233     msg = f\"Observed exception DURING INLING {code} : {e}\"
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1095, in InstructionTranslatorBase.run(self)
#    1093 try:
#    1094     self.output.push_tx(self)
# -> 1095     while self.step():
#    1096         pass
#    1097 except BackendCompilerFailed:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1007, in InstructionTranslatorBase.step(self)
#    1004 self.update_block_stack(inst)
#    1006 try:
# -> 1007     self.dispatch_table[inst.opcode](self, inst)
#    1008     return not self.output.should_exit
#    1009 except exc.ObservedException as e:
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:615, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
#     613     return handle_graph_break(self, inst, speculation.reason)
#     614 try:
# --> 615     return inner_fn(self, inst)
#     616 except Unsupported as excp:
#     617     if self.generic_context_manager_depth > 0:
#     618         # We don't support graph break under GenericContextWrappingVariable,
#     619         # If there is, we roll back to the checkpoint and fall back.
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:1714, in InstructionTranslatorBase.CALL_FUNCTION(self, inst)
#    1712 args = self.popn(inst.argval)
#    1713 fn = self.pop()
# -> 1714 self.call_function(fn, args, {})
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/symbolic_convert.py:942, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
#     940 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
#     941     raise AssertionError(f\"Attempt to trace forbidden callable {inner_fn}\")
# --> 942 self.push(fn.call_function(self, args, kwargs))
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/misc.py:1023, in GetAttrVariable.call_function(self, tx, args, kwargs)
#    1017 def call_function(
#    1018     self,
#    1019     tx: \"InstructionTranslator\",
#    1020     args: \"List[VariableTracker]\",
#    1021     kwargs: \"Dict[str, VariableTracker]\",
#    1022 ) -> \"VariableTracker\":
# -> 1023     return self.obj.call_method(tx, self.name, args, kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/tensor.py:563, in TensorVariable.call_method(self, tx, name, args, kwargs)
#     559         unimplemented(f\"unhandled args for {name}: {e}\")
#     561 from .builder import wrap_fx_proxy
# --> 563 return wrap_fx_proxy(
#     564     tx,
#     565     tx.output.create_proxy(
#     566         \"call_method\",
#     567         name,
#     568         *proxy_args_kwargs([self, *args], kwargs),
#     569     ),
#     570 )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/builder.py:2057, in wrap_fx_proxy(tx, proxy, example_value, subclass_type, **options)
#    2049 kwargs = {
#    2050     \"tx\": tx,
#    2051     \"proxy\": proxy,
#    (...)
#    2054     **options,
#    2055 }
#    2056 if subclass_type is None:
# -> 2057     return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
#    2058 else:
#    2059     result = wrap_fx_proxy_cls(target_cls=TensorWithTFOverrideVariable, **kwargs)
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/variables/builder.py:2354, in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, subclass_type, **options)
#    2352     return ConstantVariable.create(example_value, **options)
#    2353 else:
# -> 2354     unimplemented(
#    2355         \"torch.* op returned non-Tensor \"
#    2356         + f\"{typestr(example_value)} {proxy.node.op} {proxy.node.target}\",
#    2357         case_name=\"unsupported_operator\",
#    2358     )
# 
# File ~/.conda/envs/cu124/lib/python3.10/site-packages/torch/_dynamo/exc.py:304, in unimplemented(msg, from_exc, case_name)
#     302 if from_exc is not _NOTHING:
#     303     raise Unsupported(msg, case_name=case_name) from from_exc
# --> 304 raise Unsupported(msg, case_name=case_name)
# 
# Unsupported: torch.* op returned non-Tensor int call_method item
# 
# from user code:
#    File \"/tmp/ipykernel_10043/362225648.py\", line 4, in forward
#     reference_points = self.get_reference_points(spatial_shapes)
#   File \"/tmp/ipykernel_10043/362225648.py\", line 13, in get_reference_points
#     H.item()
# 
# Set TORCH_LOGS=\"+dynamo\" and TORCHDYNAMO_VERBOSE=1 for more information
# "
# }