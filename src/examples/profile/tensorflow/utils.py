import tensorflow as tf


def from_attr_proto(attr_value: tf.AttrValue):
    field_name = attr_value.WhichOneof("value")
    if field_name == "s":
        return attr_value.s.decode()
    elif field_name == "b":
        return attr_value.b
    elif field_name == "i":
        return attr_value.i
    elif field_name == "f":
        return attr_value.f
    elif field_name == "type":
        return str(attr_value.type)
    elif field_name == "shape":
        return str(attr_value.shape)
    elif field_name == "tensor":
        return str(attr_value.tensor)
    elif field_name == "func":
        return str(attr_value.func)
    elif field_name == "placeholder":
        return str(attr_value.placeholder)
    elif field_name == "list":
        list_value = attr_value.list
        if len(list_value.s) != 0:
            return [value.decode() for value in list_value.s]
        elif len(list_value.b) != 0:
            return [value for value in list_value.b]
        elif len(list_value.i) != 0:
            return [value for value in list_value.i]
        elif len(list_value.f) != 0:
            return [value for value in list_value.f]
        elif len(list_value.type) != 0:
            return [str(value) for value in list_value.type]
        elif len(list_value.shape) != 0:
            return [str(value) for value in list_value.shape]
        elif len(list_value.tensor) != 0:
            return [str(value) for value in list_value.tensor]
        elif len(list_value.func) != 0:
            return [str(value) for value in list_value.func]
        else:
            return []
