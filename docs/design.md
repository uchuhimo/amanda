# The Design of Amanda

Three design principles distinguish our approach from the existed approaches:

- A graph abstraction that is general enough to represent graphs for all frameworks
- A namespace that provides an accurate dictionary of vocabularies for each version of each framework
- A mapping mechanism that employs the flexible namespace design to automate the conversion between different namespaces

Let's introduce them in details in the following sections.

## Graph Abstraction

We use a common graph abstraction for graphs in different frameworks.
We will introduce the involved concepts one by one.

### Op

One of the basic entities in our graph abstraction is the op.
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
We can use a special index `-1` to address them.
For example, a control edge between `f`'s control output port and `m`'s control input port can be represented as:

```yaml
edge:
  output_port: { op: f, port: -1 }
  input_port: { op: g, port: -1 }
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

### Subgraph

**A subgraph is a computation that acts as an op as well as a graph.**
An op provides information about its external interfaces (input ports and output ports); a graph provides information about its internal topology.
A subgraph provides both.
For example, a computation `hk = h(hx, hy)` is implemented as a graph:

```python
hk = h(hx, hy):
  (m, n) = f(x=hx, y=hy)
  k = g(z=m)
  hk = k
```

Then we can define `k = h(x, y)` as a subgraph as follows:

```yaml
subgraph:
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

Noting that the subgraph's input ports are used as output ports inside the subgraph, while its output ports are used as input ports inside it.
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
  namespace: string
  ops: [ op | subgraph ]
  edges: [ edge ]
  attrs: { string : any }
op:
  type: string
  name: string
  input_ports: [ input_port ]
  output_ports: [ output_port ]
  attrs: { string : any }
subgraph:
  namespace: string
  type: string
  name: string
  input_ports: [ input_port ]
  output_ports: [ output_port ]
  ops: [ op | subgraph ]
  edges: [ edge ]
  attrs: { string : any }
input_port:
  name: string
  attrs: { string : any }
output_port:
  name: string
  attrs: { string : any }
edge:
  output_port:
    op: name | index
    port: name | index
  input_port:
    op: name | index
    port: name | index
```

The `name | index` in the specification of the `edge` means the op/port's name or its index in the `ops`/`input_ports`/`output_ports`.

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

The `fc` layer can be represented as a subgraph in our graph abstraction:

```yaml
subgraph:
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

Since `%10` is the weight parameter in the FC layer, this PyTorch model can be represented as a single-input, single-output subgraph:

```yaml
subgraph:
  namespace: amanda/pytorch/1.4.0
  name: graph
  input_ports:
    - name: input.1
      attrs:
        type: Float(1, 784)
  output_ports:
    - {}
  ops:
    - type: prim::Param
      output_ports:
        - name: "10"
          attrs:
            type: Float(100, 784)
    - type: aten::t
      input_ports:
        - {}
      output_ports:
        - name: "7"
          attrs:
            type: Float(784, 100)
    - type: aten::matmul
      input_ports:
        - {}
        - {}
      output_ports:
        - name: input
          attrs:
            type: Float(1, 100)
    - type: aten::relu
      input_ports:
        - {}
      output_ports:
        - name: "9"
          attrs:
            type: Float(1, 100)
  edges:
    - output_port: { op: graph, port: input.1 }
      input_ports: { op: 2, port: 0 }
    - output_port: { op: 0, port: "10" }
      input_ports: { op: 1, port: 0 }
    - output_port: { op: 1, port: "7" }
      input_ports: { op: 2, port: 1 }
    - output_port: { op: 2, port: input }
      input_ports: { op: 3, port: 0 }
    - output_port: { op: 3, port: "9" }
      input_ports: { op: graph, port: 0 }
```

## Namespace

Our graph abstraction can represent many kinds of graphs:

- graphs from different frameworks, e.g., TensorFlow, PyTorch, ONNX
- graphs from different versions of the same framework, e.g., TensorFlow v1.x and TensorFlow v2.x
- graphs defined on demand for some adhoc usages

Namespace is used to distinguish between all these different graphs.
**Namespace is a name to represent a specific kind of graph, containing a dictionary to describe a valid collection of ops called namespace schema.**
A graph with a `namespace` attribute can be validated by the namespace schema.
To describe a valid op of a specific type, we provide an op schema in the namespace schema to provide information about the name, type and default value of attributes on ops, input ports and output ports.
The full specification of the namespace schema is as follows:

```yaml
namespace:
  name: string
  op_schemas: [ op_schema | subgraph_schema ]
op_schema:
  type: string
  input_ports: [ input_port_schema ]
  output_ports: [ output_port_schema ]
  attrs: { string : type | attr_schema }
subgraph_schema:
  type: string
  input_ports: [ input_port_schema ]
  output_ports: [ output_port_schema ]
  ops: [ op | subgraph ]
  edges: [ edge ]
  attrs: { string : type | attr_schema }
input_port_schema:
  attrs: { string : type | attr_schema }
output_port_schema:
  attrs: { string : type | attr_schema }
attr_schema:
  type: type
  default: any
```

For example, a minimal namespace schema for the TensorFlow graph mentioned in the previous section is as follows:

```yaml
namespace:
  name: amanda/tensorflow/1.13.1
  op_schemas:
    - type: Placeholder
      attrs:
        name: str
        shape: Shape
        dtype: DType
      output_ports:
        - attrs:
            name:
              type: str
              default: output
            type_attr:
              type:
                constant: dtype
    - type: VariableV2
      attrs:
        name: str
        shape: Shape
        dtype: DType
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
              type:
                constant: dtype
            is_ref:
              type:
                constant: true
    - type: MatMul
      attrs:
        name: str
        T:
          type:
            enum: DType
            items: [bfloat16, half, float, double, int32, int64, complex64, complex128]
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
              type:
                constant: T
        - attrs:
            name:
              type: str
              default: b
            type_attr:
              type:
                constant: T
      output_ports:
        - attrs:
            name:
              type: str
              default: product
            type_attr:
              type:
                constant: T
    - type: Relu
      attrs:
        name: str
        T: float32
      input_ports:
        - attrs:
            name:
              type: str
              default: features
            type_attr:
              type:
                constant: T
      output_ports:
        - attrs:
            name:
              type: str
              default: activations
            type_attr:
              type:
                constant: T
```

This namespace schema provides the op schemas for all four types of ops in the TensorFlow graph.

## Mapping Mechanism

When converting between different kinds of graphs, manual manipulation is inefficient and error prone.
We can provide a mapping mechanism to automatically convert graphs between two namespaces.
**A mapping table is a series of mapping rules for conversion from one namespace to another.**
**A mapping rule matches and transform an op or a subgraph.**
A mapping rule contains the following parts:

- a matcher to match an op or a subgraph
- a mapper to transform the matched op or subgraph
- an optional list of tags to describe in which situations it will be applied

For example, a mapping table containing a rule to convert a matmul op from PyTorch to TensorFlow is as follows:



