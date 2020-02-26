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

Let's use a TensorFlow graph as an example to illustrate how to represent a graph of a mainstream framework in our graph abstraction.

The following TensorFlow code is a single layer DNN:

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

## Namespace

## Mapping Mechanism
