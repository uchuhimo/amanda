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
f:
  input_ports: [x, y]
  output_ports: [m, n]
```

An op can contain other ops, which makes it suitable to represent a subgraph. We will discuss this case after we introduce the graph concept.

### Edge

**An edge is a channel that sends a value from an output port to an input port.**
For example, if the output port `m` in `(m, n) = f(x, y)` is consumed by a computation `k = g(z)`, the edge `m->z` between `m` and `z` can be represented as:

```yaml
f.m -> g.z
```

A special kind of edges is the control edges.
**A control edge is an edge that sends an empty value called a control signal.**
It guarantees that its output port's op is executed before its input port's op.
**A control input port is a port that consumes one or more control signals; a control output port is a port that produces a control signal.**
Every op has a pair of pre-defined control input port and control output port.
We can use a special name `^control` to address them.
For example, a control edge between `f`'s control output port and `m`'s control input port can be represented as:

```yaml
f.^control -> g.^control
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
    f:
      input_ports: [x, y]
      output_ports: [m, n]
    g:
      input_ports: [z]
      output_ports: [k]
  edges:
    - f.m -> g.z
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
h:
  input_ports: [hx, hy]
  output_ports: [hk]
  ops:
    f:
      input_ports: [x, y]
      output_ports: [m, n]
    g:
      input_ports: [z]
      output_ports: [k]
  edges:
    - h.hx -> f.x
    - h.hy -> f.y
    - f.m -> g.z
    - g.k -> h.hk
```

Noting that the op's input ports are used as output ports inside the internal graph, while its output ports are used as input ports inside it.
The reason is that while its input ports consume values from outside, in the inside view they produces values to be consumed by inner input ports. So do its output ports.

### Attribute

**An attribute is a key-value pair that characterizes an op or a graph.**
For example, if op `k = g(z)` is a Relu op, it will have an attribute `type`, whose value is `Relu`. it can be represented as follows:

```yaml
g:
  attrs:
    type: Relu
```

Some kind of entities have some builtin attributes.
The op has a builtin attribute `type`; the graph has a builtin attribute `namespace`.
The functions of `type` and `namespace` will be introduced in the section for namespace.

### Full Specification

The full specification of our graph abstraction is as follows:

```yaml
graph ::=
  ops:
    $name: $op
  edges:
    - $op_name.$output_port_name -> $op_name.$input_port_name
  namespace: string  # optional
  input_ports:  # optional
    - $name | $name: $dtype
  output_ports:  # optional
    - $name | $name: $dtype
  attrs:  # optional
    $name: $value
op ::=
  type: string
  input_ports:  # optional
    - $name | $name: $dtype
  output_ports:  # optional
    - $name | $name: $dtype
  ops:  # optional
    $name: $op
  edges: # optional
    - $op_name.$output_port_name -> $op_name.$input_port_name
  attrs:  # optional
    $name: $value
name ::= string
dtype ::= string
value ::= any
op_name ::= string
output_port_name ::= string
input_port_name ::= string
```

There are some assumptions for our graph abstraction:

- Every op has a unique name among all ops in the same graph.
- Every input port has a unique name among all input ports in the same op. So does every output port.
- For anonymous ops and ports in some kinds of graphs (e.g. graphs from PyTorch), we will automatically assign a unique name for them based on ops' type or ports' order.

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
    Placeholder:
      type: Placeholder
      attrs:
        shape: [1, 784]
        dtype: float32
      output_ports:
        - output: float32
    dense/kernel:
      type: VariableV2
      attrs:
        shape: [784, 10]
        dtype: float32
        container: ""
        shared_name: ""
      output_ports:
        - ref: float32_ref
    dense/MatMul:
      type: MatMul
      attrs:
        T: float32
        transpose_a: false
        transpose_b: false
      input_ports:
        - a: float32
        - b: float32
      output_ports:
        - product: float32
    dense/Relu:
      type: Relu
      attrs:
        T: float32
      input_ports:
        - features: float32
      output_ports:
        - activations: float32
  edges:
    - Placeholder.output -> dense/MatMul.a
    - dense/kernel.ref -> dense/MatMul.b
    - dense/MatMul.product -> dense/Relu.features
```

The `fc` layer can also be represented as a subgraph in our graph abstraction:

```yaml
fc:
  type: Dense
  attrs:
    units: 10
    activation: tf.nn.relu
    use_bias: false
  input_ports: [input]
  output_ports: [logits]
  ops:
    dense/kernel:
      type: VariableV2
      attrs:
        shape: [784, 10]
        dtype: float32
        container: ""
        shared_name: ""
      output_ports:
        - ref: float32_ref
    dense/MatMul:
      type: MatMul
      attrs:
        T: float32
        transpose_a: false
        transpose_b: false
      input_ports:
        - a: float32
        - b: float32
      output_ports:
        - product: float32
    dense/Relu:
      type: Relu
      attrs:
        T: float32
      input_ports:
        - features: float32
      output_ports:
        - activations: float32
  edges:
    - fc.input -> dense/MatMul.a
    - dense/kernel.ref -> dense/MatMul.b
    - dense/MatMul.product -> dense/Relu.features
    - dense/Relu.features -> fc.logits
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
  input_ports:
    - in0: Float(1, 784)
  output_ports:
    - out0
  ops:
    param:
      type: prim::Param
      output_ports:
        out0: Float(100, 784)
    t:
      type: aten::t
      input_ports:
        - in0
      output_ports:
        - out0: Float(784, 100)
    matmul:
      type: aten::matmul
      input_ports:
        - in0
        - in1
      output_ports:
        - out0: Float(1, 100)
    relu:
      type: aten::relu
      input_ports:
        - in0
      output_ports:
        - out0: Float(1, 100)
  edges:
    - graph.in0 -> matmul.in0
    - param.out0 -> t.in0
    - t.out0 -> matmul.in1
    - matmul.out0 -> relu.in0
    - relu.out0 -> graph.out0
```

Noting that all the anonymous ops and ports are automatically assigned with a unique name, which enables them to be referred in edges.

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
namespace_schema ::=
  namespace: string
  default_type_system: string  # optional
  op_schemas:
    - type: string
      input_ports:  # optional
        - $name | $name: $type | $name: $type = $default_dtype
      output_ports:  # optional
        - $name | $name: $type | $name: $type = $default_dtype
      ops:  # optional
        $name: $op
      edges:  # optional
        - $op_name.$output_port_name -> $op_name.$input_port_name
      attrs:  # optional
        $name: $type | $type = $default_value
op ::=
  type: string
  input_ports:  # optional
    - $name | $name: $dtype
  output_ports:  # optional
    - $name | $name: $dtype
  ops:  # optional
    $name: $op
  edges: # optional
    - $op_name.$output_port_name -> $op_name.$input_port_name
  attrs:  # optional
    $name: $value
name ::= string
type ::= $type_name | $type_system.$type_name
dtype ::= string
default_dtype ::= string
default_value ::= any
op_name ::= string
output_port_name ::= string
input_port_name ::= string
```

There are some assumptions for our namespace schema:

- Same kind of ops have the same number of input ports and output ports.
- Same kind of ops have the same list of attributes. Optional attributes are not supported. You can use an attribute with null as default value instead.
- Same kind of ops contain the same list of ops and edges if they are subgraphs.

We use a pluggable type system in the namespace schema.
It means that we don't make any assumptions about the available types for attributes and users can choose a pre-defined type system or customize their own type system.
Some of the pre-defined type systems are listed below:

- `python`: Python's `int`, `str`, `List[float]`, `Union[int, str]`, etc.
- `cpp`: C++'s `int`, `std::string`, `std::vector<float>`, etc.
- `tf`: TensorFlow's `Tensor`, `TensorShape`, `DType`, etc.
- `torch`: PyTorch's `Tensor`, `Type`, etc.

We can use a type with its full qualified name like `python.int`, `tf.Tensor`, etc.
If we specify the default type system for a namespace schema using `default_type_system`, we can use a type without the type system prefix.
For example, if `default_type_system` is `python`, we can use `str` instead of `python.str`.

For example, a minimal namespace schema for the TensorFlow graph mentioned in the previous section is as follows:

```yaml
namespace_schema:
  namespace: amanda/tensorflow/1.13.1
  default_type_system: python
  op_schemas:
    - type: Placeholder
      attrs:
        shape: tf.TensorShape
        dtype: tf.DType
      output_ports:
        - output: tf.DType
    - type: VariableV2
      attrs:
        shape: tf.TensorShape
        dtype: tf.DType
        container: str = ""
        shared_name: str = ""
      output_ports:
        - ref: tf.DType
    - type: MatMul
      attrs:
        T: tf.DType
        transpose_a: bool = false
        transpose_b: bool = false
      input_ports:
        - a: tf.DType
        - b: tf.DType
      output_ports:
        - product: tf.DType
    - type: Relu
      attrs:
        T: tf.DType
      input_ports:
        - features: tf.DType
      output_ports:
        - activations: tf.DType
```

This namespace schema provides the op schemas for all four types of ops in the TensorFlow graph.

A minimal namespace schema for the PyTorch graph mentioned in the previous section is as follows:

```yaml
namespace_schema:
  namespace: amanda/pytorch/1.4.0
  default_type_system: python
  op_schemas:
    - type: prim::Param
      output_ports:
        - out0: torch.Type
    - type: aten::t
      input_ports:
        - in0
      output_ports:
        - out0: torch.Type
    - type: aten::matmul
      input_ports:
        - in0
        - in1
      output_ports:
        - out0: torch.Type
    - type: aten::relu
      input_ports:
        - in0
      output_ports:
        - out0: torch.Type
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

The full specification of a mapping table is as follows:

```yaml
table ::=
  src: string
  dst: string
  rules:
    $rule_name:
      apply_after: [$rule_name]
      src: $matcher
      dst: $mapper
  tags: [string]  # optional

matcher ::=
  ops:
    $op_name: $op_matcher
  edges:  # optional
    - $op_name.$port_name -> $op_name.$port_name
op_matcher ::=
  type: $value_matcher
  input_ports:  # optional
    - $port_name | $port_name: $value_matcher
  output_ports:  # optional
    - $port_name | $port_name: $value_matcher
  ops:  # optional
    $op_name: $op_matcher
  edges: # optional
    - $op_name.$port_name -> $op_name.$port_name
  attrs:  # optional
    $name: $value_matcher
value_matcher ::= $value | $expression

mapper ::=
  ops:
    $op_name: $op_mapper
  edges:  # optional
    - $op_name.$port_name -> $op_name.$port_name
op_mapper ::=
  type: $value_mapper
  input_ports:  # optional
    - $port_name | $port_name: $value_mapper
  output_ports:  # optional
    - $port_name | $port_name: $value_mapper
  ops:  # optional
    $op_name: $op_mapper
  edges: # optional
    - $op_name.$port_name -> $op_name.$port_name
  attrs:  # optional
    $name: $value_mapper
value_mapper ::= $value | $expression

expression ::= "${" + string + "}"
variable_name ::= "$" + string
rule_name ::= string
op_name ::= $variable_name | $name
port_name ::= $variable_name | $name
name ::= string
value ::= any
```

There are some assumptions for the mapping rules:

- Expression is a valid Python expression, which can access declared variables.
- Expression in a matcher returns a boolean value, which indicates whether it matches or not.
- Expression in a mapper returns a value for assignment.
- Variable name is a unique name among all variables in the same matcher/mapper.
- If the same variable/name is used in both a matcher and a mapper, the mapper will update the matched op/port.
- If a variable/name only appears in the matcher, the corresponding op/port will be removed.
- All edges in the matcher will be replaced by the edges in the mapper. An edge only appears in the matcher will be removed; an edge only appears in the mapper will be added.

For example, a mapping table containing a rule to convert the matmul op in the PyTorch graph to the TensorFlow graph in the previous section is as follows:

```yaml
table:
  src: amanda/pytorch/1.4.0
  dst: amanda/tensorflow/1.13.1
  rules:
    convert_matmul:
      src:
        ops:
          $matmul:
            type: aten::matmul
      dst:
        ops:
          dense/MatMul:
            type: MatMul
            attrs:
              T: |
                ${
                    dtype = op.output_ports["out0"].dtype.scalarType()
                    if dtype == "Float":
                      return "float32"
                    elif dtype == "Double":
                      return "float64"
                    ...
                }
```

The mapping table maps from namespace `amanda/pytorch/1.4.0` to `amanda/tensorflow/1.13.1`.
The rule `convert_matmul` uses a matcher in `src` to match an `aten::matmul` op, and uses a mapper in `dst` to transform it into a `MatMul` op with name `dense/MatMul`.

### Subgraph matching

We can also convert the transpose op and the matmul op in the PyTorch graph to the matmul op in the TensorFlow graph using subgraph matching as follows:

```yaml
table:
  src: amanda/pytorch/1.4.0
  dst: amanda/tensorflow/1.13.1
  rules:
    convert_matmul:
      src:
        ops:
          $t:
            type: aten::t
          $matmul:
            type: aten::matmul
        edges:
          - $t.out0 -> $matmul.in0
      dst:
        ops:
          dense/MatMul:
            type: MatMul
            attrs:
              transpose_a: true
              T: |
                ${
                    dtype = op.output_ports["out0"].dtype.scalarType()
                    if dtype == "Float":
                      return "float32"
                    elif dtype == "Double":
                      return "float64"
                    ...
                }
```
