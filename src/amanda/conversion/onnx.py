from amanda.namespace import Namespace, default_namespace, get_global_registry
from amanda.rule import NoopRule, RuleMapper

_namespace = default_namespace() / Namespace("onnx")


def onnx_namespace() -> Namespace:
    return _namespace


_onnx_to_default_mapper = RuleMapper(rules=[NoopRule()])
_default_to_onnx_mapper = RuleMapper(rules=[NoopRule()])


def onnx_to_default_mapper() -> RuleMapper:
    return _onnx_to_default_mapper


def default_to_onnx_mapper() -> RuleMapper:
    return _default_to_onnx_mapper


get_global_registry().add_mapper(
    onnx_namespace(), default_namespace(), onnx_to_default_mapper()
)
get_global_registry().add_mapper(
    default_namespace(), onnx_namespace(), default_to_onnx_mapper()
)
