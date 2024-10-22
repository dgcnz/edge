# Compilation

One of PyTorch main strengths is its flexibility and ease of use. The user can write a model with almost no restrictions as long as each operator is differentiable and PyTorch will take care of the rest. On each forward pass, it will evaluate the operators on-the-fly and dynamically construct the computation graph, which is then used to compute the gradients during the backward pass. This is called Eager execution mode and it is the default behavior of PyTorch.

This mode comes in handy when the computation graph is not static, for example when the model has if-statements or loops that depend on the input data or when the input has dynamic shapes (imagine training a model with multiple resolutions). However, this flexibility comes at a cost, because we can't optimize a model if we don't know what operations, which shapes, types or even order of operations will be executed until runtime. This is where compilers come in.

## Compilers 101

A compiler is a program that translates instructions written in one representation (source) into another representation (target). Nowadays, compilers usually are separated in a frontend and a backend. The frontend is responsible for parsing the source code and generating an intermediate representation (IR) that is independent of the source language. The backend is responsible for translating the IR into the target language. This separation allows for reusability of the frontend with different backends, and vice versa as we can see in {numref}`Figure {number} <retargetable>`.

:::{figure-md} retargetable
<img src="retargetable.png" alt="Retargetable Compilers">

Frontends produce an intermediate representation (IR) common to all backends. Backends take the IR and generate code for a specific target. {cite}`aosabook`
:::

So far, we've talked about compilation as a process that happens before the program is executed, also known as ahead-of-time (AOT) compilation. However, there are other ways to execute a program. Some languages, like Python, are *interpreted*, where programs are executed line by line by the runtime interpreter. Furthermore, some runtimes might use just-in-time (JIT) compilation, where parts of the program are compiled while it is being executed. This allows for optimizations that can only be done at runtime, like specializing code for specific inputs.


## Machine Learning Compilers

Machine learning compilers take a model written in some framework (e.g. PyTorch), translate it into a program that can be executed in some runtime (e.g. TensorRT, CoreML, PyTorch TorchInductor) which then ends up optimized for some specialized hardware (e.g. GPUs, TPUs, Apple Silicon). 
 
PyTorch has had a few different compiler solutions over the years, the most popular being TorchScript. Recently, however, PyTorch has been endorsing its new compiler frontend TorchDynamo and its accompanying API ecosystem `torch.export` and `torch.compile`. 




