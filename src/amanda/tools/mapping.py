import amanda


class Rule:
    def __init__(self, namespace, apply_fn):
        self.namespace = namespace
        self.apply_fn = apply_fn

    def apply(self, context):
        self.apply_fn(context)


class MappingTool(amanda.Tool):
    def __init__(self, rules):
        super().__init__(namespace="mapping")

        self.rules = []
        for namespace, apply_fn in rules:
            self.add_rule(namespace, apply_fn)
        self.add_inst_for_op(self.instrumentation)
        self.add_inst_for_op(self.instrumentation, require_outputs=True)
        self.add_inst_for_op(
            self.instrumentation,
            backward=True,
        )
        self.add_inst_for_op(
            self.instrumentation,
            backward=True,
            require_outputs=True,
        )

    def add_rule(self, namespace, apply_fn):
        self.rules.append(Rule(namespace, apply_fn))

    def instrumentation(self, context: amanda.OpContext):
        for rule in self.rules:
            if rule.namespace == context.namespace:
                rule.apply(context)
