# The Rationale of Amanda

Amanda is a computation graph instrumentation framework for neural networks.
To explain our motivation to propose Amanda, let's briefly introduce some background and terminologies first.

## What is instrumentation?

Instrumentation is a technique to insert extra code into an existed program, usually for:

- **Program analysis** like performance profiling, code tracing, code coverage and debugging
- **Simulation/Emulation** for processor or cache

There are two instrumentation approaches:

- **Source code instrumentation** that instruments source code
- **Binary instrumentation** that instruments binary executable, which can be further classified into:
    - **Static instrumentation** that instruments the binary executable before execution and generates a persistent modified binary executable
    - **Dynamic instrumentation** that instruments the binary executable during execution without making any persistent modifications to the binary executable

## What is instrumentation for neural networks?

Using instrumentation, we can also insert code into an existed neural network.
Instrumentation for neural network has some similar use cases with traditional program instrumentation:

- **Program Analysis** like performance profiling, dataflow analysis and debugging
- **Simulation/Emulation** for different execution backends (e.g. CPU/GPU/TPU)

However, instrumentation for neural network also serves for a series of distinct scenarios including:

- **(Computation) Graph partition** for distributed training or secure computation (protects critical model components)
- **Neural Architecture Search(NAS)** that adjusts a model's architecture during training of a controller
- **Quantization** to replace some operations in a neural network to its quantized version
- **Pruning** for weight/activation masking

Other than different scenarios, neural networks also have different abstraction levels for instrumentation, thanks to the complicated compilation pipelines in Deep Learning(DL) frameworks nowadays.
Simply speaking, a neural network will be represented as a computation graph in a DL framework (except some dynamic frameworks without JIT), and then compiled to binary for different execution backends.
According to the pipeline, there are three instrumentation approaches for neural networks:

- **Source code instrumentation** that instruments source code of the neural network. In this case, we can reuse instrumentation tools for the host languages.
- **(Computation) graph instrumentation** that instruments the computation graph. In this case, we can only rely on the framework itself to provide the instrumentation tools usually, which are often either absent or limited. For example, TensorFlow has Grappler for graph rewriting, which is only used internally and lack of documentation.
- **Binary instrumentation** that instruments the execution engine. Since the compiled binary is unavailable to users in most cases, the only option is dynamic instrumentation for the execution engine instead of the binary directly, which is verbose and error prone. Some execution engines provide integrated instrumentation tools like profiler, which is, however, barely extendable.

## Why do we bet on graph instrumentation?

Because the computation graph is the currency in the DL world.
On the one hand, the source code of a neural network is only available during training but unavailable during serving; on the other hand, the compiled binary and the serving engine are unavailable during training.
The serialized computation graph, as an well-defined exchange format, acts as a bridge between the training stage and the serving stage.
An instrumentation tool for computation graph can serve scenarios in either stages.

## Wait. Are we reinventing the wheel?

There are many existed works that provides some sort of graph instrumentation tools for neural networks:

- **ONNX**
- **MMdnn**
- **MLIR**

## Why does the "Grand Unified IR" approach not work?

- source instrumentation
- superset approach


IR:

- an structure specification
- an operation specification

loose vs strict

## Our approach: small compromises, huge gains

a bridge

## Other contributions
