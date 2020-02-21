## Namespace

Namespace is a vocabulary for a particular kind of graphs, which can be described as a series of schemas.
Different frameworks use different kinds of graphs; even in the same framework, different versions use different op sets.
We use a separate namespace for each kind of them.
To define each namespace explicitly, we employ a series of schemas encoded as YAML.

### YAML Syntax Convention

Here are the syntax conventions used in this article:

- To the left of `:` is a literal keyword used in definitions.
- To the right of `:` is a data type. The data type can be a primitive type like `string` or a reference to another schema defined elsewhere.
- The notation `[ datatype ]` indicates an array of the mentioned data type. For instance, `[ string ]` is an array of strings.
- The notation `{ datatype : datatype }` indicates a mapping of one data type to another. For instance, `{ string: string }` is a mapping of strings to strings.
- The symbol `|` indicates there are multiple data types available for the keyword.

### Namespace Schema

A namespace is composed of a common graph structure and a list of ops (i.e., a opset).
Its schema is as followed:

```yaml
namespace:
  id: string
  graph: graph
  opset: opset
```

We employ a graph schema to describe the common graph structure and a opset schema to describe the opset.

### Graph Schema

A common graph structure is described by common graph/op/edge/tensor attributes.
Its schema is as followed:

```yaml
graph:
  id: str
  attrs: { str: str }
  op:
    attrs: { str: str }
  edge:
    attrs: { str: str }
  tensor:
    attrs: { str: str }
```

### Opset Schema

An opset is a list of different types of ops.
Its schema is as followed:

```yaml
opset:
  id: str
  ops: [ op ]
```

An op is composed of its type, attributes, input ports and output ports (including the tensor in it).
The op schema is as followed:

```yaml
op:
  id: str
  type: str
  attrs: { str: str }
  input_ports:
    - tensor:
        attrs: { str: str }
   output_tensor:
     - tensor:
         attrs: { str: str }
```

The `tensor` in `input_ports` describe what kinds of tensors each input port would like to accept.

The op schema can be used to describe a subgraph.
In this case, an `inner` should be provided to describe the inner structure of the subgraph.
The schema for a subgraph (as a composite op) is as followed:

```yaml
op:
  id: str
  type: str
  attrs: { str: str }
  input_ports:
    - tensor:
        attrs: { str: str }
   output_tensor:
     - tensor:
         attrs: { str: str }
    inner:
      ops: [ op ]
      edges: [ edge ]
```

The edge schema in `inner` is as followed:

```yaml
edge:
  src_op: integer
  src_index: integer
  dst_op: integer
  dst_index: integer
  attrs: { str: str }
```

The `src_op` and `dst_op` is the index of the referred op in `inner`'s `ops`.

## Mapping Mechanism

To convert graph in different namespaces, we can define a series of rules for mapping between them.
Three are three kinds of rules:

- graph rule for mapping between graph schemas
- op rule for mapping between op schemas in opsets
- node rule for mapping between op instances (i.e., nodes)
