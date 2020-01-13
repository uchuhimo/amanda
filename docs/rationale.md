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
The serialized computation graph, which is an well-defined exchange format, acts as a bridge between the training stage and the serving stage.
Thus an instrumentation tool designed around the computation graph can serve scenarios in either stages.

In spite of all these advantages, Graph instrumentation is not a perfect alternative for either source code instrumentation or binary instrumentation. Compared to source code, computation graph is more low-level and harder to instrument; compared to binary instrumentation, graph instrumentation cannot access the runtime state of the execution engine, making it unsuitable for operations' performance profiling (since we don't know how operations are scheduled during graph instrumentation).

## Three scenarios in graph instrumentation

The complexity of graph instrumentation largely depends on how many ecosystems (or frameworks, we use them interchangeably) is involved.
According to the complexity, we classify graph instrumentation into three cases:

- **Instrument in a single ecosystem (no-mapping case)**. This is the simplest case. For example, we use it when debugging a model on a single framework.
- **Instrument in one ecosystem, apply to another ecosystem (one-to-one case)**. In this case, we write instrumentation code for an ecosystem, and convert the modified graph to another ecosystem's graph format. For example, during post-training quantization, we use operations' quantized version to replace them in the training framework, and then convert the quantized graph to the serving framework's format.
- **Instrument once, apply to multiple ecosystems (one-to-many case)**. To achieve the goal of applying to multiple ecosystems, we need to build a common operation set across all the supported ecosystems, and verify that no operation in the graph is out of this common set. For example, if we are developing a cross-framework distributed training framework and want to write the graph partition code only once, we have to deal with this level of complexity.

## The current approaches: What's missing?

There are many existed works that provides some sort of graph instrumentation tools for neural networks:

- **Grappler** is TensorFlow's internal graph rewriting tool, which is used to develop compilation passes in TensorFlow's complier.
- **MMdnn** is an model conversion framework that supports many frameworks, each of which has its own graph format. It avoids the engineering complexity of implementing a separate converter for each pair of two frameworks by introducing a common IR.
- **ONNX** is an open computation graph exchange format supported by many DL frameworks. It proposes a common IR that other DL frameworks can convert their training graph to. Some instrumentation tools are provided out of box, including graph validation, graph optimization and shape inference.

All of them are not designed specially for graph instrumentation, but they indeed provide an infrastructure to build a full-featured graph instrumentation framework upon.

To support all the three scenarios mentioned above, many features are requested in a graph instrumentation framework:

| Features | Scenario | Grappler | MMdnn | ONNX |
| --- | --- | --- | --- | --- |
| Full-featured instrumentation APIs | all cases | Y | N | N |
| Represent all operations in a single framework | no-mapping case | Y | N | N |
| A common representation across multiple frameworks | one-to-many case | N | Y | Y |

## Our approach
