# https://github.com/pytorch/TensorRT/issues/3269

import torch
import torch_tensorrt
from torchvision.ops.boxes import _batched_nms_coordinate_trick


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Parameter(torch.randn(4))
        self.r = torch.nn.Parameter(torch.randn(10))

    def forward(self, x: torch.Tensor):
        boxes = torch.outer(x, self.c)
        scores = self.r * x
        idx = torch.arange(boxes.shape[0], device=boxes.device)
        keep = _batched_nms_coordinate_trick(boxes, scores, idx, 0.0)
        torch._check(keep.shape[0] <= 10)
        boxes = boxes[keep]
        return boxes


with torch.no_grad():
    x = torch.randn(10).cuda()
    inputs = (x,)
    m = Model().eval().cuda()
    m(*inputs)
    ep = torch.export.export(m, inputs)
    trt_gm = torch_tensorrt.dynamo.compile(ep, inputs, debug=True)
    print(trt_gm.graph)


"""
DEBUG:torch_tensorrt.dynamo.lowering.passes.remove_detach:Removed 1 detach nodes:
graph():
    %p_c : [num_users=1] = placeholder[target=p_c]
    %p_r : [num_users=1] = placeholder[target=p_r]
    %c_lifted_tensor_0 : [num_users=1] = placeholder[target=c_lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %outer : [num_users=3] = call_function[target=torch.ops.aten.outer.default](args = (%x, %p_c), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_r, %x), kwargs = {})
    %arange : [num_users=1] = call_function[target=torch.ops.aten.arange.default](args = (10,), kwargs = {device: cuda:0, pin_memory: False})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%outer,), kwargs = {})
    %_to_copy : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%arange,), kwargs = {dtype: torch.float32, device: cuda:0})
    %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%c_lifted_tensor_0,), kwargs = {})
    %_to_copy_1 : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%lift_fresh_copy,), kwargs = {dtype: torch.float32, device: cuda:0})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_to_copy_1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_to_copy, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%outer, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=3] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %ge_1 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 2), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_1, Runtime assertion failed for expression u0 >= 2 on node 'ge_1'), kwargs = {})
    %le_1 : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int_1, 10), kwargs = {})
    %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le_1, Runtime assertion failed for expression u0 <= 10 on node 'le_1'), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%outer, [%nms]), kwargs = {})
    return (index,)
WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_unlift.py:63: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)

WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/fx/graph.py:1794: UserWarning: Node lifted_tensor_0 target lifted_tensor_0 lifted_tensor_0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(

WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torchvision/_meta_registrations.py:173: FutureWarning: `create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead
  num_to_keep = ctx.create_unbacked_symint()

WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torchvision/_meta_registrations.py:173: FutureWarning: `create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead
  num_to_keep = ctx.create_unbacked_symint()

WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/export/_unlift.py:63: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)

WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch/fx/graph.py:1794: UserWarning: Node lifted_tensor_0 target lifted_tensor_0 lifted_tensor_0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(

DEBUG:torch_tensorrt.dynamo._compiler:Input graph: graph():
    %c : [num_users=1] = get_attr[target=c]
    %r : [num_users=1] = get_attr[target=r]
    %lifted_tensor_0 : [num_users=1] = get_attr[target=lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%x, [10, 1]), kwargs = {})
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %c), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %arange : [num_users=1] = call_function[target=torch.ops.aten.arange.start_step](args = (0, 10), kwargs = {layout: torch.strided, device: cuda:0, pin_memory: False})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_to_copy : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%arange,), kwargs = {dtype: torch.float32, device: cuda:0})
    %_to_copy_1 : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%lifted_tensor_0,), kwargs = {dtype: torch.float32, device: cuda:0})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_to_copy_1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_to_copy, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=3] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %ge_1 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 2), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_1, Runtime assertion failed for expression u0 >= 2 on node 'ge_1'), kwargs = {})
    %le_1 : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int_1, 10), kwargs = {})
    %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le_1, Runtime assertion failed for expression u0 <= 10 on node 'le_1'), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return (index,)
DEBUG:torch_tensorrt.dynamo.lowering.passes.constant_folding:Graph after constant folding:
graph():
    %c : [num_users=1] = get_attr[target=c]
    %r : [num_users=1] = get_attr[target=r]
    %x : [num_users=2] = placeholder[target=x]
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%x, [10, 1]), kwargs = {})
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %c), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_frozen_param0 : [num_users=1] = get_attr[target=_frozen_param0]
    %_frozen_param1 : [num_users=1] = get_attr[target=_frozen_param1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_frozen_param1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_frozen_param0, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=3] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %ge_1 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 2), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_1, Runtime assertion failed for expression u0 >= 2 on node 'ge_1'), kwargs = {})
    %le_1 : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int_1, 10), kwargs = {})
    %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le_1, Runtime assertion failed for expression u0 <= 10 on node 'le_1'), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return (index,)
DEBUG:torch_tensorrt.dynamo.lowering.passes.view_to_reshape:Graph after replacing view with reshape:
graph():
    %c : [num_users=1] = get_attr[target=c]
    %r : [num_users=1] = get_attr[target=r]
    %x : [num_users=2] = placeholder[target=x]
    %reshape_default : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%x, [10, 1]), kwargs = {})
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reshape_default, %c), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_frozen_param0 : [num_users=1] = get_attr[target=_frozen_param0]
    %_frozen_param1 : [num_users=1] = get_attr[target=_frozen_param1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_frozen_param1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_frozen_param0, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=3] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %ge_1 : [num_users=1] = call_function[target=operator.ge](args = (%sym_size_int_1, 2), kwargs = {})
    %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%ge_1, Runtime assertion failed for expression u0 >= 2 on node 'ge_1'), kwargs = {})
    %le_1 : [num_users=1] = call_function[target=operator.le](args = (%sym_size_int_1, 10), kwargs = {})
    %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%le_1, Runtime assertion failed for expression u0 <= 10 on node 'le_1'), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return (index,)
DEBUG:torch_tensorrt.dynamo.lowering.passes.remove_assert_scalar:Removed 2 assert_scalar nodes:
graph():
    %c : [num_users=1] = get_attr[target=c]
    %r : [num_users=1] = get_attr[target=r]
    %x : [num_users=2] = placeholder[target=x]
    %reshape_default : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%x, [10, 1]), kwargs = {})
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reshape_default, %c), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_frozen_param0 : [num_users=1] = get_attr[target=_frozen_param0]
    %_frozen_param1 : [num_users=1] = get_attr[target=_frozen_param1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_frozen_param1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_frozen_param0, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return (index,)
DEBUG:torch_tensorrt.dynamo.lowering.passes.accumulate_fp32_matmul:Skipping FP32 accumulation for matmul layers as use_fp32_acc is not enabled in the compilation settings
DEBUG:torch_tensorrt.dynamo._compiler:Lowered Input graph: graph():
    %c : [num_users=1] = get_attr[target=c]
    %r : [num_users=1] = get_attr[target=r]
    %x : [num_users=2] = placeholder[target=x]
    %reshape_default : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%x, [10, 1]), kwargs = {})
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reshape_default, %c), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_frozen_param0 : [num_users=1] = get_attr[target=_frozen_param0]
    %_frozen_param1 : [num_users=1] = get_attr[target=_frozen_param1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_frozen_param1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_frozen_param0, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return (index,)
DEBUG:torch_tensorrt.dynamo.partitioning._global_partitioner:
Supported Nodes:
- torch.ops.aten.reshape.default + Operator Count: 1
- torch.ops.aten.mul.Tensor + Operator Count: 3
- torch.ops.aten.max.default + Operator Count: 1
- torch.ops.aten.add.Tensor + Operator Count: 2
- torch.ops.aten.slice.Tensor + Operator Count: 1
- torch.ops.aten.unsqueeze.default + Operator Count: 1
- torch.ops.aten.sym_size.int + Operator Count: 1

DEBUG:torch_tensorrt.dynamo.partitioning._global_partitioner:
Unsupported or Excluded Nodes:
- torch.ops.torchvision.nms.default + Operator Count: 1
- torch.ops.aten.index.Tensor + Operator Count: 1

DEBUG:torch_tensorrt.dynamo._compiler:Detected support for 10 operators out of 13 in subgraph.
INFO:torch_tensorrt.dynamo._compiler:Partitioning the graph via the fast partitioner
DEBUG:torch_tensorrt.dynamo.partitioning._adjacency_partitioner:Eliminating acc subgraph because it's smaller than the threshold: 1 < 5
DEBUG:torch_tensorrt.dynamo.partitioning._adjacency_partitioner:
Number of TensorRT-Accelerated Engines Generated: 1
DEBUG:torch_tensorrt.dynamo.partitioning._adjacency_partitioner:
Supported Nodes:
- torch.ops.aten.reshape.default + Operator Count: 1
- torch.ops.aten.mul.Tensor + Operator Count: 3
- torch.ops.aten.max.default + Operator Count: 1
- torch.ops.aten.add.Tensor + Operator Count: 2
- torch.ops.aten.slice.Tensor + Operator Count: 1
- torch.ops.aten.unsqueeze.default + Operator Count: 1
- torch.ops.aten.sym_size.int + Operator Count: 1

DEBUG:torch_tensorrt.dynamo.partitioning._adjacency_partitioner:
Unsupported or Excluded Nodes:
- torch.ops.torchvision.nms.default + Operator Count: 1
- torch.ops.aten.index.Tensor + Operator Count: 1

DEBUG:torch_tensorrt.dynamo._compiler:Updated metadata for node: _run_on_acc_0 with its corresponding submodule outputs
DEBUG:torch_tensorrt.dynamo._compiler:Converting submodule: _run_on_acc_0
 Input shapes: [(10,)]
 graph():
    %x : [num_users=2] = placeholder[target=x]
    %reshape_default : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%x, [10, 1]), kwargs = {})
    %c : [num_users=1] = get_attr[target=c]
    %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reshape_default, %c), kwargs = {})
    %r : [num_users=1] = get_attr[target=r]
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%r, %x), kwargs = {})
    %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.default](args = (%mul,), kwargs = {})
    %_frozen_param1 : [num_users=1] = get_attr[target=_frozen_param1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%max_1, %_frozen_param1), kwargs = {})
    %_frozen_param0 : [num_users=1] = get_attr[target=_frozen_param0]
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_frozen_param0, %add), kwargs = {})
    %slice_1 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_2, 0, 0, 9223372036854775807), kwargs = {})
    %unsqueeze : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%slice_1, 1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %unsqueeze), kwargs = {})
    return (add_1, mul_1, mul)
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node x (kind: x, args: ())
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Adding input to in-progress INetwork: x [shape=[10], dtype=DataType.FLOAT]
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node x [x] (Inputs: () | Outputs: (x: (10,)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /reshape_default (kind: aten.reshape.default, args: ('x <Node>', ['10 <int>', '1 <int>']))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /reshape_default [aten.reshape.default] (Inputs: (x: (10,)@torch.float32, [10, 1]) | Outputs: (reshape_default: (10, 1)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node c (kind: c, args: ())
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node c [c] (Inputs: () | Outputs: (c: (4,)@float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /mul (kind: aten.mul.Tensor, args: ('reshape_default <Node>', 'c <Node>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /mul [aten.mul.Tensor] (Inputs: (reshape_default: (10, 1)@torch.float32, c: (4,)@float32) | Outputs: (mul: (10, 4)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node r (kind: r, args: ())
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node r [r] (Inputs: () | Outputs: (r: (10,)@float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /mul_1 (kind: aten.mul.Tensor, args: ('r <Node>', 'x <Node>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /mul_1 [aten.mul.Tensor] (Inputs: (r: (10,)@float32, x: (10,)@torch.float32) | Outputs: (mul_1: (10,)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /max_1 (kind: aten.max.default, args: ('mul <Node>',))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /max_1 [aten.max.default] (Inputs: (mul: (10, 4)@torch.float32) | Outputs: (max_1: ()@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node _frozen_param1 (kind: _frozen_param1, args: ())
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node _frozen_param1 [_frozen_param1] (Inputs: () | Outputs: (_frozen_param1: ()@float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /add (kind: aten.add.Tensor, args: ('max_1 <Node>', '_frozen_param1 <Node>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /add [aten.add.Tensor] (Inputs: (max_1: ()@torch.float32, _frozen_param1: ()@float32) | Outputs: (add: ()@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node _frozen_param0 (kind: _frozen_param0, args: ())
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node _frozen_param0 [_frozen_param0] (Inputs: () | Outputs: (_frozen_param0: (10,)@float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /mul_2 (kind: aten.mul.Tensor, args: ('_frozen_param0 <Node>', 'add <Node>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /mul_2 [aten.mul.Tensor] (Inputs: (_frozen_param0: (10,)@float32, add: ()@torch.float32) | Outputs: (mul_2: (10,)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /slice_1 (kind: aten.slice.Tensor, args: ('mul_2 <Node>', '0 <int>', '0 <int>', '9223372036854775807 <int>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /slice_1 [aten.slice.Tensor] (Inputs: (mul_2: (10,)@torch.float32, 0, 0, 9223372036854775807) | Outputs: (slice_1: (10,)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /unsqueeze (kind: aten.unsqueeze.default, args: ('slice_1 <Node>', '1 <int>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /unsqueeze [aten.unsqueeze.default] (Inputs: (slice_1: (10,)@torch.float32, 1) | Outputs: (unsqueeze: (10, 1)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node /add_1 (kind: aten.add.Tensor, args: ('mul <Node>', 'unsqueeze <Node>'))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node /add_1 [aten.add.Tensor] (Inputs: (mul: (10, 4)@torch.float32, unsqueeze: (10, 1)@torch.float32) | Outputs: (add_1: (10, 4)@torch.float32))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converting node output (kind: output, args: (('add_1 <Node>', 'mul_1 <Node>', 'mul <Node>'),))
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Marking output output0 [shape=(10, 4), dtype=DataType.FLOAT]
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Marking output output1 [shape=(10,), dtype=DataType.FLOAT]
DEBUG:torch_tensorrt.dynamo.conversion._TRTInterpreter:Marking output output2 [shape=(10, 4), dtype=DataType.FLOAT]
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Converted node output [output] (Inputs: ((add_1, mul_1, mul)) | Outputs: (output: ))
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:TRT INetwork construction elapsed time: 0:00:00.006279
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Not found cached TRT engines. Start building engine.
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:Build TRT engine elapsed time: 0:00:00.821156
INFO:torch_tensorrt.dynamo.conversion._TRTInterpreter:TRT Engine uses: 72812 bytes of Memory

DEBUG: [Torch-TensorRT] - Deserializing Device Info: 0%8%9%0%NVIDIA GeForce RTX 4060 Ti
DEBUG: [Torch-TensorRT] - Deserialized Device Info: Device(ID: 0, Name: NVIDIA GeForce RTX 4060 Ti, SM Capability: 8.9, Type: GPU)
DEBUG: [Torch-TensorRT] - Target Device: Device(ID: 0, Name: NVIDIA GeForce RTX 4060 Ti, SM Capability: 8.9, Type: GPU)
DEBUG: [Torch-TensorRT] - Setting Device(ID: 0, Name: NVIDIA GeForce RTX 4060 Ti, SM Capability: 8.9, Type: GPU) as active device
INFO: [Torch-TensorRT] - Loaded engine size: 0 MiB
DEBUG: [Torch-TensorRT] - Deserialization required 1352 microseconds.
DEBUG: [Torch-TensorRT] - Total per-runner device persistent memory is 0
DEBUG: [Torch-TensorRT] - Total per-runner host persistent memory is 1664
DEBUG: [Torch-TensorRT] - Allocated device scratch memory of size 1024
DEBUG: [Torch-TensorRT] - - Runner scratch: 1024 bytes
INFO: [Torch-TensorRT] - [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
DEBUG: [Torch-TensorRT] - CUDA lazy loading is enabled.
DEBUG: [Torch-TensorRT] - Input binding name: x has TensorRT binding index: 0, Torch binding index: 0
DEBUG: [Torch-TensorRT] - Output binding name: output0 has TensorRT binding index: 1, Torch binding index: 1
DEBUG: [Torch-TensorRT] - Output binding name: output1 has TensorRT binding index: 2, Torch binding index: 2
DEBUG: [Torch-TensorRT] - Output binding name: output2 has TensorRT binding index: 3, Torch binding index: 3
DEBUG: [Torch-TensorRT] - Torch-TensorRT TensorRT Engine:
  Name: _run_on_acc_0_engine
  Inputs: [
    id: 0
      name: x
      shape: [10]
      dtype: Float
  ]
  Outputs: [
    id: 0
      name: output0
      shape: [10, 4]
      dtype: Float
    id: 1
      name: output1
      shape: [10]
      dtype: Float
    id: 2
      name: output2
      shape: [10, 4]
      dtype: Float
  ]
  Device: Device(ID: 0, Name: NVIDIA GeForce RTX 4060 Ti, SM Capability: 8.9, Type: GPU)
  Hardware Compatibility: Disabled
  Target Platform: linux_x86_64

DEBUG:torch_tensorrt.dynamo._compiler:Submodule in PyTorch: _run_on_gpu_1
 graph():
    %add_1 : [num_users=1] = placeholder[target=add_1]
    %mul_1 : [num_users=1] = placeholder[target=mul_1]
    %nms : [num_users=2] = call_function[target=torch.ops.torchvision.nms.default](args = (%add_1, %mul_1, 0.0), kwargs = {})
    %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%nms, 0), kwargs = {})
    %sym_constrain_range_for_size_default : [num_users=0] = call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](args = (%sym_size_int_1,), kwargs = {})
    %mul : [num_users=1] = placeholder[target=mul]
    %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%mul, [%nms]), kwargs = {})
    return index
Traceback (most recent call last):
  File "/home/dgcnz/development/amsterdam/edge/scripts/mre/trt_graphbreak_symint.py", line 28, in <module>
    trt_gm = torch_tensorrt.dynamo.compile(ep, inputs, debug=True)
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/_compiler.py", line 318, in compile
    trt_gm = compile_module(
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/_compiler.py", line 544, in compile_module
    parse_graph_io(gm, dryrun_tracker)
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/utils.py", line 428, in parse_graph_io
    output_shapes = get_graph_io_attrs(output_nodes, "shape")
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/utils.py", line 407, in get_graph_io_attrs
    graph_io_attrs.append(attr_fn(metadata))
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/utils.py", line 374, in unwrap_tensor_shape
    tensor_shape.extend(unwrap_tensor_shape(dimension))
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/utils.py", line 370, in unwrap_tensor_shape
    min_max_opt = extract_var_range_info(tensor)
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/torch_tensorrt/dynamo/utils.py", line 345, in extract_var_range_info
    min_val, max_val, opt_val = int(var_range.lower), int(var_range.upper), int(var_val)
  File "/home/dgcnz/.conda/envs/cu124/lib/python3.10/site-packages/sympy/core/expr.py", line 307, in __int__
    raise TypeError("Cannot convert symbols to int")
TypeError: Cannot convert symbols to int
WARNING:py.warnings:/home/dgcnz/.conda/envs/cu124/lib/python3.10/tempfile.py:833: ResourceWarning: Implicitly cleaning up <TemporaryDirectory '/tmp/tmpgywq6drr'>
  _warnings.warn(warn_message, ResourceWarning)
"""
