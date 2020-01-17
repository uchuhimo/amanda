# The Rationale of Amanda

Amanda is a computation graph instrumentation framework for neural networks.
To explain our motivation to propose Amanda, let's briefly introduce some background and terminologies first.

## What is instrumentation?

Instrumentation is a technique to insert extra code into an existed program[1], usually for:

- **Program analysis** like performance profiling, code tracing, code coverage and debugging
- **Simulation/Emulation** for processor or cache

There are two instrumentation approaches:

- **Source code instrumentation** that instruments source code
- **Binary instrumentation** that instruments binary executable, which can be further classified into:
    - **Static instrumentation** that instruments the binary executable before execution and generates a persistent modified binary executable
    - **Dynamic instrumentation** that instruments the binary executable during execution without making any persistent modifications to the binary executable

## What is instrumentation for neural networks?

Using instrumentation, we can also insert code into an existed neural network, usually for:

- **Program Analysis** like performance profiling, dataflow analysis and debugging
- **(Computation) Graph partition** for distributed training or secure computation (protects critical model components)
- **Neural Architecture Search(NAS)** that adjusts a model's architecture during searching
- **Quantization** to replace some operations in a neural network to its quantized version
- **Pruning** for weight/activation masking

Let's briefly introduce some examples to better understand the application scenarios of instrumentation for neural network:

- **Debugging**: In this case, we want to insert code into a neural network to dump all the intermediate results of every operation.
- **Quantization**: In this case, we want to replace every Conv operations with its quantized version in a trained model, and then convert the model to another model format for serving.
- **Effective path extraction[2]**: In this case, we want to extract a critical subset of neurons, synapses, and weights in the neural network that contributes most to the predicted class's output, which is called effective path. To extract effective path, we need to dump all the intermediate results during the inference, and then reconstruct the paths by back-propagation. During reconstruction, we apply the corresponding extraction op of each op to the output neurons in effective path the get the critical input neurons, synapses and weights in the current layer. Since different frameworks share the same extraction strategy, we want to use the same instrumentation code for multiple frameworks.

The examples above show three typical scenarios that the complexity is proportional to the number of involved frameworks (or ecosystems, we use them interchangeably):

- **Instrument in a single ecosystem (no-mapping case)**. A simple example is debugging a model on a single framework.
- **Instrument in one ecosystem, apply to another ecosystem (one-to-one case)**. In this case, we write instrumentation code for an ecosystem, and convert the modified graph to another ecosystem's graph format. The corresponding example is quantization.
- **Instrument once, apply to multiple ecosystems (one-to-many case)**. To achieve the goal of applying to multiple ecosystems, we need to build a common operation set across all the supported ecosystems, and verify that no operation in the graph is out of this common set. Graph partition in our examples has to deal with this level of complexity.

There are three instrumentation approaches:

- **Source code instrumentation** that instruments source code of the neural network.
- **(Computation) graph instrumentation** that instruments the computation graph. The computation graph is the intermediate representation of the neural network. Some DL frameworks expose them to be directly manipulated by users.
- **Binary instrumentation** that instruments the compiled binary or the execution engine.

## The current approaches: What's missing?

Although not designed specially for instrumentation, there are many existed works that provides some sort of instrumentation tools for neural networks:

- For no-mapping case:
    - Source code instrumentation
        - **PyTorch Module Hooks** are PyTorch's built-in tools to modify the forward and backward passes. We can add code before / after every op executes.
        - **MXNet Block Hooks** are MXNet's built-in tools to modify the graph. It is similar to PyTorch module hooks except that it only works in the forward pass.
    - Graph instrumentation
        - **TensorFlow Session Hooks** are TensorFlow's built-in tools to modify graph during the lifecycle of the session. Using session hooks, we can add op to the graph before finalized graph and observe the output value of arbitrary ops.
        - **Grappler** is TensorFlow's internal graph rewriting tool, which is used to develop compilation passes in TensorFlow's compiler.
    - Binary instrumentation
        - **TFLite Profiler** is TFLite's built-in profiler. It can provide runtime information for a serving model, including execution time and GPU memory usage. There are few binary instrumentation tools for neural network, TFLite Profiler is not an instrumentation tool but has a similar design with a binary instrumentation tool.
- For one-to-many case:
    - **MMdnn** is an model conversion framework that supports many frameworks, each of which has its own graph format. It avoids the engineering complexity of implementing a separate converter for each pair of two frameworks by introducing a common IR.
    - **ONNX** is an open computation graph exchange format supported by many DL frameworks. It proposes a common IR that other DL frameworks can convert their training graph to. Some instrumentation tools are provided out of box, including graph validation, graph optimization and shape inference.

To support all the three scenarios mentioned above, many features are requested in a instrumentation framework:

| Features | TensorFlow Session Hooks | PyTorch Module Hooks | MXNet Block Hooks | Grappler | MMdnn | ONNX | TFLite Profiler |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Available during training | Y | Y | Y | Y | N | N | N |
| Available during serving | N | N | N | N | Y | Y | Y |
| Public instrumentation APIs | Y | Y | Y | N | N | N | N |
| Insert ops | Y | Y | Y | Y | Y | Y | N |
| Remove ops | N | N | N | Y | Y | Y | N |
| Modify backward graph | Y | Y* | N | Y | N | N | N |
| Support all ops in a single framework | Y | N** | N*** | Y | N | N | Y |
| Support multiple frameworks | N | N | N | N | Y | Y | N |

*: only support to insert ops

**: only for module APIs, ops created from low-level functional APIs are not supported.

***: only for Gluon/Block APIs, ops created from Symbol/Module APIs are not supported.

It is obvious that none of the tools above can fully satisfy the requirements of the three scenarios. It is necessary to design a new instrumentation tool that can support all the required features. But what does it look like?

## Our Design Rationale

We follow several design principles to meet all the requirements above, which also distinguish our approach from the existed approaches:

- A standalone tool

    Instead of mending the existed tools in each framework, it would be better to support all these frameworks in one place. Being a standalone tool also eases the support for new tools in the future.

- A graph instrumentation framework

    We choose graph instrumentation among three possible instrumentation approaches because graph instrumentation is a general abstraction that is sufficient for all three scenarios. The serialized computation graph, which is an well-defined exchange format, acts as a bridge between the training stage and the serving stage. Thus an instrumentation tool designed around the computation graph can serve scenarios in either stages.

    In spite of all these advantages, Graph instrumentation is not a perfect alternative for either source code instrumentation or binary instrumentation. Compared to source code, computation graph is more low-level and harder to instrument; compared to binary instrumentation, graph instrumentation cannot access the runtime state of the execution engine, making it unsuitable for operations' performance profiling (since we don't know how operations are scheduled during graph instrumentation).

- A minimal abstraction

    We provide the minimum abstraction for graph instrumentation by removing some unnecessary strong assumption introduced by existed frameworks:

    - Closed operation set assumption: Existed tools like ONNX assume that the operation set is a closed set. This assumption introduces an unacceptable engineering complexity that we need to create a superset of all operations in all supported frameworks. In Amanda, we employ an open operation set instead.
    - Tensor layout assumption: Existed tools assume that tensors are in a specific layout. For example, only dense tensor or sparse tensor in DOK format is allowed in ONNX; only dense tensor is allowed in MMdnn. In Amanda, any tensor layout is allowed as long as the shape is correct.
    - Required attribute assumption: Existed tools assume that some attributes are required. For example, name attribute is required in TensorFlow/MMdnn, but op in PyTorch Graph is anonymous. In Amanda, any attribute of operation/graph is optional.

[1]: Pin: building customized program analysis tools with dynamic instrumentation
[2]: Adversarial Defense Through Network Profiling Based Path Extraction

## Wait. Are we reinventing the wheel?

There are many existed works that provides some sort of graph instrumentation tools for neural networks:

- **ONNX(common graph format)** is an open computation graph exchange format supported by many DL frameworks. It proposes a common IR that other DL frameworks can convert their training graph to. Some instrumentation tools are provided out of box, including graph validation, graph optimization and shape inference.
- **MLIR(compiler infrastructure)** is a compiler IR that aimed to serve as a common infrastructure for different DL compilers. It enables instrumentation for its IR by providing a DAG rewriter infrastructure.
- **MMdnn(domain specific tool)** is an model conversion framework that supports many frameworks, each of which has its own graph format. It avoids the engineering complexity of implementing a separate converter for each pair of two frameworks by introducing a common IR.

All of them are not designed specially for graph instrumentation, but they indeed provide an infrastructure (mainly the proposed common IR) to build a full-featured graph instrumentation framework upon.

Can we directly reuse their infrastructures without propose ours?
To answer this question, let's induce the design principle behind their IRs.
In general, an graph IR consists of three components:

- a definition of the computation graph structure
- definitions of a type system
- definitions of built-in operations

The design principle of all of them is that their IR is strict in all these three components.
It means that all frameworks are force to convert to the IR's graph structure, type system, and operations to access their infrastructure.
Let's call this design principle a "Grand Unified IR" (GUIR) approach for short.

## Why does the "Grand Unified IR" approach not work?

Theoretically, it is a feasible approach, provided every operation in every framework can be defined in the IR spec.
In practice, however, the GUIR approach doesn't work.
Let's take ONNX as an example.
ONNX is aimed to be an standard exchange format, and many companies invest a huge amount of time and money to extend its supported operations.
However, there are only about 137 operators in ONNX v1.5, while TensorFlow r1.13 has more than 500.
There are two reasons for the lack of operators:

- **The engineering complexity is unacceptable.** To support a common IR other than its own IR in every framework means that every operations should be defined twice. One in its own IR, another in the common IR. And every framework should do so. Thus every framework tends to support only a common subset among all frameworks in the common IR.
