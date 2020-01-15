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

Using instrumentation, we can also insert code into an existed neural network, usually for:

- **Program Analysis** like performance profiling, dataflow analysis and debugging
- **(Computation) Graph partition** for distributed training or secure computation (protects critical model components)
- **Neural Architecture Search(NAS)** that adjusts a model's architecture during searching
- **Quantization** to replace some operations in a neural network to its quantized version
- **Pruning** for weight/activation masking

There are three instrumentation approaches:

- **Source code instrumentation** that instruments source code of the neural network.
- **(Computation) graph instrumentation** that instruments the computation graph. The computation graph is the intermediate representation of the neural network. Some DL frameworks expose them to be directly manipulated by users.
- **Binary instrumentation** that instruments the compiled binary or the execution engine.

## Three typical scenarios in the instrumentation for neural network

Let's briefly introduce some examples to better understand the application scenarios of instrumentation for neural network:

- **Debugging**: In this case, we want to insert code into a neural network to dump all the intermediate results of every operation.
- **Quantization**: In this case, we want to replace every Conv operations with its quantized version in a trained model, and then convert the model to another model format for serving.
- **(Computation) Graph partition**: In this case, we want to partition a large model into several small sub-graphs that can be fit into a single GPU's memory. Since different frameworks share the same partition strategy, we want to use the same instrumentation code for multiple frameworks.

The examples above show three typical scenarios that the complexity is proportional to the number of involved frameworks (or ecosystems, we use them interchangeably):

- **Instrument in a single ecosystem (no-mapping case)**. A simple example is debugging a model on a single framework.
- **Instrument in one ecosystem, apply to another ecosystem (one-to-one case)**. In this case, we write instrumentation code for an ecosystem, and convert the modified graph to another ecosystem's graph format. The corresponding example is quantization.
- **Instrument once, apply to multiple ecosystems (one-to-many case)**. To achieve the goal of applying to multiple ecosystems, we need to build a common operation set across all the supported ecosystems, and verify that no operation in the graph is out of this common set. Graph partition in our examples has to deal with this level of complexity.

## The current approaches: What's missing?

Although not designed specially for instrumentation, there are many existed works that provides some sort of instrumentation tools for neural networks:

- **TensorFlow Session Hooks** are TensorFlow's built-in tools to modify graph during the lifecycle of the session. Using session hooks, we can add op to the graph before finalized graph and observe the output value of arbitrary ops.
- **PyTorch Module Hooks** are PyTorch's built-in tools to modify the forward and backward passes. We can add code before / after every op executes.
- **MXNet Block Hooks** are MXNet's built-in tools to modify the graph. It is similar to PyTorch module hooks except that it only works in the forward pass.
- **Grappler** is TensorFlow's internal graph rewriting tool, which is used to develop compilation passes in TensorFlow's compiler.
- **MMdnn** is an model conversion framework that supports many frameworks, each of which has its own graph format. It avoids the engineering complexity of implementing a separate converter for each pair of two frameworks by introducing a common IR.
- **ONNX** is an open computation graph exchange format supported by many DL frameworks. It proposes a common IR that other DL frameworks can convert their training graph to. Some instrumentation tools are provided out of box, including graph validation, graph optimization and shape inference.
- **TFLite Profiler** is TFLite's built-in profiler. It can provide runtime information for a serving model, including execution time and GPU memory usage.

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

    We choose graph instrumentation among three possible instrumentation approaches because the computation graph is the currency in the DL world.
    On the one hand, the source code of a neural network is only available during training but unavailable during serving; on the other hand, the compiled binary and the serving engine are unavailable during training.
    The serialized computation graph, which is an well-defined exchange format, acts as a bridge between the training stage and the serving stage.
    Thus an instrumentation tool designed around the computation graph can serve scenarios in either stages.

    In spite of all these advantages, Graph instrumentation is not a perfect alternative for either source code instrumentation or binary instrumentation. Compared to source code, computation graph is more low-level and harder to instrument; compared to binary instrumentation, graph instrumentation cannot access the runtime state of the execution engine, making it unsuitable for operations' performance profiling (since we don't know how operations are scheduled during graph instrumentation).

- A low learning curve

    We provide the minimum abstraction for each scenario to lower users' learning curve.
    Existed instrumentation tools use a common IR to support multiple frameworks, which is strict in all three components of an IR:

    - Use the same computation graph structure
    - Use the same type system
    - Use the same set of built-in operations

    The common IR approach is a heavy abstraction for our scenarios:

    - In no-mapping case, we only need to consider a single framework, thus don't need a common IR at all.
    - In one-to-one case, we only need to consider the conversion between two frameworks, a common IR is not necessary.
    - In one-to-many case, a common IR is only necessary only when we need to unify the semantics of all the operations, which is a rare case.

    Besides the heavy abstraction, the engineering complexity to build a common IR is also unacceptable. To support a common IR other than its own IR in every framework means that every operations should be defined twice. One in its own IR, another in the common IR. And every framework should do so. Thus every framework tends to support only a common subset among all frameworks in the common IR.

    Instead, we only define a common computation graph structure that is general enough to represent any frameworks' graphs.
    When a mapping from one framework to another framework is required, we use a table mapping mechanism to bridge them.
    User can extend this table to meet their own requirements, and share their extension with the community to help each other.
