# The Design of Amanda

Three design principles distinguish our approach from the existed approaches:

- A graph abstraction that is general enough to represent graphs for all frameworks
- A namespace that provides an accurate dictionary of vocabularies for each version of each framework
- A mapping mechanism that employs the flexible namespace design to automate the conversion between different namespaces

Let's introduce them in details in the following sections.

## Graph Abstraction

We use a common graph abstraction for graphs in different frameworks.
We will introduce the involved concepts one by one.

One of the basic entities in our graph abstraction is the op.
An op is a computation that consumes some values and produces some new values.
An op has several input ports and output ports.
An input port is a port that consumes a value; an output port is a port that produces a value.
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

An edge is a channel that sends a value from an output port to an input port.
For example, if the output port `m` in `(m, n) = f(x, y)` is consumed by a computation `k = g(z)`, the edge `m->z` between `m` and `z` can be represented as:

```yaml
edge:
  output_port:
    op: f
    port: m
  input_port:
    op: g
    port: z
```

A special kind of edges is the control edges.
A control edge is an edge that sends an empty value called a control signal.
It guarantees that its output port's op is executed before its input port's op.
A control input port is a port that consumes one or more control signals; a control output port is a port that produces a control signal.
Every op has a pair of pre-defined control input port and control output port.
We can use a special index `-1` to address them.
For example, a control edge between `f`'s control output port and `m`'s control input port can be represented as:

```yaml
edge:
  output_port:
    op: f
    port: -1
  input_port:
    op: g
    port: -1
```

A graph is a computation that consists of a series of ops and edges.
For example, op `(m, n) = f(x, y)`, op `k = g(z)` and edge `m->z` compose a graph as follows:

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
    - output_port:
        op: f
        port: m
      input_port:
        op: g
        port: z
```

An attribute is a key-value pair that characterizes an entity including an op, an input port, an output port, an edge and a graph.
For example, op `k = g(z)` has an attribute `name`, whose value is `g`. it can be represented as follows:

```yaml
op:
  attrs:
    - key: name
      value: g
```

Some kind of entities have some builtin attributes.
The op has builtin attributes `name` and `type`; both the input port ans the output port has a builtin attribute `name`; the graph has a builtin attribute `namespace`.
The functions of `type` and `namespace` will be introduced in the next section.

Let's use a TensorFlow graph as an example to illustrate how to represent a graph in a mainstream framework in our graph abstraction.

The following TensorFlow code is a single layer DNN:

```python
input = tf.placeholder(dtype=tf.float32, shape=(1, 28 * 28))
fc = tf.layers.Dense(units=10, activation=tf.nn.relu, use_bias=False)
logits = fc(input)
```

It represents a graph like this:

```
input -> matmul -> relu
           ^
weight ----|
```

The underlying TensorFlow graph will be represented as follows in our graph abstraction:

```yaml
graph:
  namespace: amanda/tensorflow/1.13.1
  attrs:
    - key: versions
      value:
        producer: 27
  ops:
    - type: Placeholder
      name: Placeholder
      attrs:
        - key: shape
          value: [1, 784]
        - key: dtype
          value: float32
      output_ports:
        - name: output
          attrs:
            - key: type_attr
              value: dtype
    - type: VariableV2
      name: dense/kernel
      attrs:
        - key: shape
          value: [784, 10]
        - key: dtype
          value: float32
        - key: container
          value: ""
        - key: shared_name
          value: ""
      output_ports:
        - name: ref
          attrs:
            - key: type_attr
              value: dtype
            - key: is_ref
              value: true
    - type: MatMul
      name: dense/MatMul
      attrs:
        - key: T
          value: float32
        - key: transpose_a
          value: false
        - key: transpose_b
          value: false
      input_ports:
        - name: a
          attrs:
            - key: type_attr
              value: T
        - name: b
          attrs:
            - key: type_attr
              value: T
      output_ports:
        - name: product
          attrs:
            - key: type_attr
              value: T
    - type: Relu
      name: dense/Relu
      attrs:
        - key: T
          value: float32
      input_ports:
        - name: features
          attrs:
            - key: type_attr
              value: T
      output_ports:
        - name: activations
          attrs:
            - key: type_attr
              value: T
  edges:
    - output_port:
        op: Placeholder
        port: output
      input_port:
        op: dense/MatMul
        port: a
    - output_port:
        op: dense/kernel
        port: ref
      input_port:
        op: dense/MatMul
        port: b
    - output_port:
        op: dense/MatMul
        port: product
      input_port:
        op: dense/Relu
        port: features
```
