# The Design of Amanda

Three design principles distinguish our approach from the existed approaches:

- A graph abstraction that is general enough to represent graphs for all frameworks
- A namespace that provides an accurate dictionary of vocabularies for each version of each framework
- A mapping mechanism that employs the flexible namespace design to automate the conversion between different namespaces

Let's introduce them in details in the following sections.

## Graph Abstraction

We use a common graph abstraction for graphs in different frameworks.
In our graph abstraction, every neural network can be represented by a graph containing multiple ops that connected by edges between them.
We will introduce the involved concepts one by one.

### Op

The basic entity in our graph abstraction is the op.
**An op is a computation that consumes some values and produces some new values.**
An op has several input ports and output ports.
**An input port is a port that consumes a value; an output port is a port that produces a value.**
For example, a computation `(m, n) = f(x, y)` that consumes `x` and `y` and produces `m` and `n` can be represented as an op as followed:

```yaml
op:
  name: f
  input_ports:
    - name: x
    - name: y
  output_ports:
    - name: m
    - name: n
```

An op can contain other ops, which makes it suitable to represent a subgraph. We will discuss this case after we introduce the graph concept.

### Edge

**An edge is a channel that sends a value from an output port to an input port.**
For example, if the output port `m` in `(m, n) = f(x, y)` is consumed by a computation `k = g(z)`, the edge `m->z` between `m` and `z` can be represented as:

```yaml
edge:
  output_port: { op: f, port: m }
  input_port: { op: g, port: z }
```

A special kind of edges is the control edges.
**A control edge is an edge that sends an empty value called a control signal.**
It guarantees that its output port's op is executed before its input port's op.
**A control input port is a port that consumes one or more control signals; a control output port is a port that produces a control signal.**
Every op has a pair of pre-defined control input port and control output port.
We can use a special name `^control` to address them.
For example, a control edge between `f`'s control output port and `m`'s control input port can be represented as:

```yaml
edge:
  output_port: { op: f, port: ^control }
  input_port: { op: g, port: ^control }
```

### Graph

**A graph is a computation that consists of a series of ops and edges.**
For example, op `(m, n) = f(x, y)`, op `k = g(z)` and edge `m->z` compose a computation as follows:

```python
(m, n) = f(x, y)
k = g(z=m)
```

This computation can be represented as a graph:

```yaml
graph:
  ops:
    - name: f
      input_ports:
        - name: x
        - name: y
      output_ports:
        - name: m
        - name: n
    - name: g
      input_ports:
        - name: z
      output_ports:
        - name: k
  edges:
    - output_port: { op: f, port: m }
      input_port: { op: g, port: z }
```

A graph can optionally contain input ports and output ports like an op.

**An op can contain a graph.**
In this case, the op contains a series of other ops and edges.
We also call such kind of ops as subgraphs.
For example, a computation `hk = h(hx, hy)` is implemented as a graph:

```python
hk = h(hx, hy):
  (m, n) = f(x=hx, y=hy)
  k = g(z=m)
  hk = k
```

Then we can define `k = h(x, y)` as an op as follows:

```yaml
op:
  name: h
  input_ports:
    - name: hx
    - name: hy
  output_ports:
    - name: hk
  ops:
    - name: f
      input_ports:
        - name: x
        - name: y
      output_ports:
        - name: m
        - name: n
    - name: g
      input_ports:
        - name: z
      output_ports:
        - name: k
  edges:
    - output_port: { op: f, port: m }
      input_port: { op: g, port: z }
    - output_port: { op: h, port: hx }
      input_port: { op: f, port: x }
    - output_port: { op: h, port: hy }
      input_port: { op: f, port: y }
    - output_port: { op: g, port: k }
      input_port: { op: h, port: hk }
```

Noting that the op's input ports are used as output ports inside the internal graph, while its output ports are used as input ports inside it.
The reason is that while its input ports consume values from outside, in the inside view they produces values to be consumed by inner input ports. So do its output ports.

### Attribute

**An attribute is a key-value pair that characterizes an entity including an op, an input port, an output port, an edge and a graph.**
For example, op `k = g(z)` has an attribute `name`, whose value is `g`. it can be represented as follows:

```yaml
op:
  attrs:
    name: g
```

Some kind of entities have some builtin attributes.
The op has builtin attributes `name` and `type`; both the input port and the output port has a builtin attribute `name`; the graph has a builtin attribute `namespace`.
All builtin attributes are optional by default.
The functions of `type` and `namespace` will be introduced in the next section.

### Full Specification

The full specification of our graph abstraction is as follows:

```yaml
graph:
  ops: [ op ]
  edges: [ edge ]
  namespace: string  # optional
  input_ports:  # optional
    - name: string
      attrs: { string : any }
  output_ports:  # optional
    - name: string
      attrs: { string : any }
  attrs: { string : any }  # optional
op:
  type: string
  name: string
  input_ports:
    - name: string
      attrs: { string : any }  # optional
  output_ports:
    - name: string
      attrs: { string : any }  # optional
  ops: [ op ]  # optional
  edges: [ edge ]  # optional
  attrs: { string : any }  # optional
edge:
  output_port:
    op: name
    port: name
  input_port:
    op: name
    port: name
```

The `name` in the specification of the `edge` means the op/port's name.

### A realistic example

Let's use a single layer neural network as an example to illustrate how to represent a graph of a mainstream framework in our graph abstraction.

The following TensorFlow code is a single layer neural network:

```python
input = tf.placeholder(dtype=tf.float32, shape=(1, 28 * 28))
fc = tf.layers.Dense(units=10, activation=tf.nn.relu, use_bias=False)
logits = fc(input)
```

It represents a graph like this:

```
               fc
        -----------------
input → | matmul → relu | → logits
        |    ↑          |
        | weight        |
        -----------------
```

The underlying TensorFlow graph will be represented as follows in our graph abstraction:

```yaml
graph:
  namespace: amanda/tensorflow/1.13.1
  attrs:
    versions: { producer: 27 }
  ops:
    - type: Placeholder
      name: Placeholder
      attrs:
        shape: [1, 784]
        dtype: float32
      output_ports:
        - name: output
          attrs:
            type_attr: dtype
    - type: VariableV2
      name: dense/kernel
      attrs:
        shape: [784, 10]
        dtype: float32
        container: ""
        shared_name: ""
      output_ports:
        - name: ref
          attrs:
            type_attr: dtype
            is_ref: true
    - type: MatMul
      name: dense/MatMul
      attrs:
        T: float32
        transpose_a: false
        transpose_b: false
      input_ports:
        - name: a
          attrs:
            type_attr: T
        - name: b
          attrs:
            type_attr: T
      output_ports:
        - name: product
          attrs:
            type_attr: T
    - type: Relu
      name: dense/Relu
      attrs:
        T: float32
      input_ports:
        - name: features
          attrs:
            type_attr: T
      output_ports:
        - name: activations
          attrs:
            type_attr: T
  edges:
    - output_port: { op: Placeholder, port: output }
      input_port: { op: dense/MatMul, port: a }
    - output_port: { op: dense/kernel, port: ref }
      input_port: { op: dense/MatMul, port: b }
    - output_port: { op: dense/MatMul, port: product }
      input_port: { op: dense/Relu, port: features }
```

The `fc` layer can also be represented as a subgraph in our graph abstraction:

```yaml
op:
  type: Dense
  name: fc
  attrs:
    units: 10
    activation: tf.nn.relu
    use_bias: false
  input_ports:
    - name: input
  output_ports:
    - name: logits
  ops:
    - type: VariableV2
      name: dense/kernel
      attrs:
        shape: [784, 10]
        dtype: float32
        container: ""
        shared_name: ""
      output_ports:
        - name: ref
          attrs:
            type_attr: dtype
            is_ref: true
    - type: MatMul
      name: dense/MatMul
      attrs:
        T: float32
        transpose_a: false
        transpose_b: false
      input_ports:
        - name: a
          attrs:
            type_attr: T
        - name: b
          attrs:
            type_attr: T
      output_ports:
        - name: product
          attrs:
            type_attr: T
    - type: Relu
      name: dense/Relu
      attrs:
        T: float32
      input_ports:
        - name: features
          attrs:
            type_attr: T
      output_ports:
        - name: activations
          attrs:
            type_attr: T
  edges:
    - output_port: { op: fc, port: input }
      input_port: { op: dense/MatMul, port: a }
    - output_port: { op: dense/kernel, port: ref }
      input_port: { op: dense/MatMul, port: b }
    - output_port: { op: dense/MatMul, port: product }
      input_port: { op: dense/Relu, port: features }
    - output_port: { op: dense/Relu, port: activations }
      input_port: { op: fc, port: logits }
```

The following PyTorch code is an equivalent single layer neural network:

```python
input = torch.randn(1, 28 * 28)
model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 100, bias=False),
    torch.nn.ReLU(),
)
traced_model = torch.jit.trace(model, (input,))
logits = traced_model(input)
```

After `torch.jit.trace`, the model is translated into the following TorchScript graph:

```python
graph(%input.1 : Float(1, 784), %10 : Float(100, 784)):
  %7 : Float(784, 100) = aten::t(%10)
  %input : Float(1, 100) = aten::matmul(%input.1, %7)
  %9 : Float(1, 100) = aten::relu(%input)
  return (%9)
```

Since `%10` is the weight parameter in the FC layer, this PyTorch model can be represented as a single-input, single-output graph:

```yaml
graph:
  namespace: amanda/pytorch/1.4.0
  name: graph
  input_ports:
    - name: input.1
      attrs:
        type: Float(1, 784)
  output_ports:
    - name: _0
  ops:
    - type: prim::Param
      name: _0
      output_ports:
        - name: "10"
          attrs:
            type: Float(100, 784)
    - type: aten::t
      name: _1
      input_ports:
        - name: _0
      output_ports:
        - name: "7"
          attrs:
            type: Float(784, 100)
    - type: aten::matmul
      name: _2
      input_ports:
        - name: _0
        - name: _1
      output_ports:
        - name: input
          attrs:
            type: Float(1, 100)
    - type: aten::relu
      name: _3
      input_ports:
        - name: _0
      output_ports:
        - name: "9"
          attrs:
            type: Float(1, 100)
  edges:
    - output_port: { op: graph, port: input.1 }
      input_ports: { op: _2, port: _0 }
    - output_port: { op: _0, port: "10" }
      input_ports: { op: _1, port: _0 }
    - output_port: { op: _1, port: "7" }
      input_ports: { op: _2, port: _1 }
    - output_port: { op: _2, port: input }
      input_ports: { op: _3, port: _0 }
    - output_port: { op: _3, port: "9" }
      input_ports: { op: graph, port: _0 }
```

Noting that all the anonymous ops and ports are automatically assigned with a unique name, which enables them to be referred in edges.
Variable like `%10` in the TorchScript will get names without the prefixed `%` like `10`, since `%` is a marker for variable, not a part of its name.

## Namespace

Our graph abstraction can represent many kinds of graphs:

- graphs from different frameworks, e.g., TensorFlow, PyTorch, ONNX
- graphs from different versions of the same framework, e.g., TensorFlow v1.x and TensorFlow v2.x
- graphs defined on demand for some adhoc usages

Namespace is used to distinguish between all these different graphs.
**Namespace is a name to represent a specific kind of graph, containing a dictionary to describe a valid collection of ops called namespace schema.**
Our namespace is hierarchical, unlike the flattened namespace in many frameworks (e.g., ONNX/MMdnn/MLIR).
For example, namespace `amanda/tensorflow/1.13.1` is a namespace inside the namespace `amanda/tensorflow`.
Generally speaking, every framework has its own namespace, with a namespace for each version inside the framework namespace.

A graph with a `namespace` attribute can be validated by the namespace schema.
To describe a valid op of a specific type, we provide an op schema in the namespace schema to provide information about the name, type and default value of attributes on ops, input ports and output ports.
The full specification of the namespace schema is as follows:

```yaml
namespace:
  name: string
  type_system: string
  op_schemas:
    - type: string
      input_ports:  # optional
        - attrs: { string : attr_type }
      output_ports:  # optional
        - attrs: { string : attr_type }
      ops: [ op ]  # optional
      edges: [ edge ]  # optional
      attrs: { string : attr_type }  # optional
attr_type: type | optional_attr_type | constant_attr_type | attr_types
optional_attr_type:
  type: type
  default: any
constant_attr_type:
  type: type
  constant: any
attr_types:
  one_of: [ attr_type ]
  default: any  # optional
```

We use a pluggable type system in the namespace schema.
It means that we don't make any assumptions about the available types for attributes and users can choose a pre-defined type system or customize their own type system.
A series of type systems are pre-defined:

- `python`: Python
- `cpp`: C++
- `protobuf`: Protocol buffers (users should specify the URLs for schema files)
- `flatbuffers`: FlatBuffers (users should specify the URLs for schema files)
- `torchscript`: TorchScript

For example, a minimal namespace schema for the TensorFlow graph mentioned in the previous section is as follows:

```yaml
namespace:
  name: amanda/tensorflow/1.13.1
  type_system: python
  op_schemas:
    - type: Placeholder
      attrs:
        name: str
        shape: tensorflow.TensorShape
        dtype: tensorflow.DType
      output_ports:
        - attrs:
            name:
              type: str
              default: output
            type_attr:
              type: str
              constant: dtype
    - type: VariableV2
      attrs:
        name: str
        shape: tensorflow.TensorShape
        dtype: tensorflow.DType
        container:
          type: str
          default: ""
        shared_name:
          type: str
          default: ""
      output_ports:
        - attrs:
            name:
              type: str
              default: ref
            type_attr:
              type: str
              constant: dtype
            is_ref:
              type: bool
              constant: true
    - type: MatMul
      attrs:
        name: str
        T:
          one_of:
            - type: tensorflow.DType
              constant: bfloat16
            - type: tensorflow.DType
              constant: half
            - type: tensorflow.DType
              constant: float
            - type: tensorflow.DType
              constant: double
            - type: tensorflow.DType
              constant: int32
            - type: tensorflow.DType
              constant: int64
            - type: tensorflow.DType
              constant: complex64
            - type: tensorflow.DType
              constant: complex64
        transpose_a:
          type: bool
          default: false
        transpose_b:
          type: bool
          default: false
      input_ports:
        - attrs:
            name:
              type: str
              default: a
            type_attr:
              type: str
              constant: T
        - attrs:
            name:
              type: str
              default: b
            type_attr:
              type: str
              constant: T
      output_ports:
        - attrs:
            name:
              type: str
              default: product
            type_attr:
              type: str
              constant: T
    - type: Relu
      attrs:
        name: str
        T:
          one_of:
            - type: tensorflow.DType
              constant: bfloat16
            - type: tensorflow.DType
              constant: half
            - type: tensorflow.DType
              constant: float
            - type: tensorflow.DType
              constant: double
            - type: tensorflow.DType
              constant: qint8
            - type: tensorflow.DType
              constant: uint8
            - type: tensorflow.DType
              constant: int8
            - type: tensorflow.DType
              constant: uint16
            - type: tensorflow.DType
              constant: int16
            - type: tensorflow.DType
              constant: uint32
            - type: tensorflow.DType
              constant: int32
            - type: tensorflow.DType
              constant: uint64
            - type: tensorflow.DType
              constant: int64
      input_ports:
        - attrs:
            name:
              type: str
              default: features
            type_attr:
              type: str
              constant: T
      output_ports:
        - attrs:
            name:
              type: str
              default: activations
            type_attr:
              type: str
              constant: T
```

This namespace schema provides the op schemas for all four types of ops in the TensorFlow graph.

A minimal namespace schema for the PyTorch graph mentioned in the previous section is as follows:

```yaml
namespace:
  name: amanda/pytorch/1.4.0
  type_system: torchscript
  op_schemas:
    - type: prim::Param
      input_ports:
        - attrs:
            name: str
      output_ports:
        - attrs:
            name: str
            type: type
    - type: aten::t
      input_ports:
        - attrs:
            name: str
      output_ports:
        - attrs:
            name: str
            type: type
    - type: aten::matmul
      input_ports:
        - attrs:
            name: str
        - attrs:
            name: str
      output_ports:
        - attrs:
            name: str
            type: type
    - type: aten::relu
      input_ports:
        - attrs:
            name: str
      output_ports:
        - attrs:
            name: str
            type: type
```

## Mapping Mechanism

When converting between different kinds of graphs, manual manipulation is inefficient and error prone.
We can provide a mapping mechanism to automatically convert graphs between two namespaces.
**A mapping table is a series of mapping rules for conversion from one namespace to another.**
**A mapping rule matches and transform an op.**
A mapping rule contains the following parts:

- a matcher to match an op (the `src` part)
- a mapper to transform the matched op (the `dst` part)
- an optional list of tags to describe in which situations it will be applied

For example, a mapping table containing a rule to convert the matmul op in the PyTorch graph to the TensorFlow graph in the previous section is as follows:

```yaml
table:
  src: amanda/pytorch/1.4.0
  dst: amanda/tensorflow/1.13.1
  rules:
    - rule_name: convert_matmul
      src:
        type: aten::matmul
        output_ports:
          - attrs:
              type: Float
      dst:
        type: MatMul
        name: dense/MatMul
        attrs:
          T: float32
```

The mapping table maps from namespace `amanda/pytorch/1.4.0` to `amanda/tensorflow/1.13.1`.
The rule `convert_matmul` uses a matcher in `src` to match an `aten::matmul` op with `Float` output type, and uses a mapper in `dst` to transform it into a `MatMul` op with name `dense/MatMul` and attribute `T: float32`.

### Subgraph matching

We can also convert the transpose op and the matmul op in the PyTorch graph to the matmul op in the TensorFlow graph using subgraph matching as follows:

```yaml
table:
  src: amanda/pytorch/1.4.0
  dst: amanda/tensorflow/1.13.1
  rules:
    - rule_name: convert_matmul
      src:
        ops:
          - type: aten::t
            name:
              ref: t_name
            output_ports:
              - name:
                  ref: t_output_name
          - type: aten::matmul
            name:
              ref: matmul_name
            input_ports:
              - name: _0
            output_ports:
              - attrs:
                  type: Float
        edges:
          - output_port: { op: "{t_name}", port: "{t_output_name}" }
            input_ports: { op: "{matmul_name}", port: _0 }
      dst:
        type: MatMul
        name: dense/MatMul
        attrs:
          T: float32
          transpose_a: true
```

### Pushdown

The matcher-mapper pair can be pushed down into the op structure to:

- Narrow the match/transform scope
- Reduce repeated matching structure between the matcher and the mapper

You can declare the matcher-mapper pair at any level of the op structure.
Here is an equivalent form of the `convert_matmul` rule:


```yaml
table:
  src: amanda/pytorch/1.4.0
  dst: amanda/tensorflow/1.13.1
  rules:
    - rule_name: convert_matmul
      type:
        src: aten::matmul
        dst: MatMul
      output_ports:
          - attrs:
              src:
                type: Float
      name:
        dst: dense/MatMul
      attrs:
        dst:
          T: float32
```





