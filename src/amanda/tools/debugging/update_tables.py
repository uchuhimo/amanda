# type: ignore
import amanda.tools.debugging.graph as amanda

rule = amanda.create_rule(
    {
        "ops": [
            {
                "output_ports": [
                    {
                        "tensor": {
                            "attrs": {
                                "src": [{"key": "type", "ref": "tensor_type"}],
                                "dst": [
                                    {"ref": "tensor_type"},
                                    {
                                        "key": "is_valid",
                                        "value": "{tensor_type.value.kind() == 'TensorType'}",  # noqa: E501
                                    },
                                ],
                            }
                        }
                    }
                ]
            }
        ]
    }
)
table = amanda.get_mapping_table("amanda/pytorch", "debugging")
table.insert_rule(rule)

rule1 = amanda.create_rule(
    {
        "ops": [
            {
                "type": "store_tensor_to_file",
                "output_ports": {
                    "src": [{"index": 0, "ref": "output_port"}],
                    "dst": [
                        {
                            "ref": "output_port",
                            "tensor": {
                                "attrs": [
                                    {
                                        "key": "type",
                                        "value": "{op.input_tensors[0].attrs['type']}",
                                    }
                                ]
                            },
                        }
                    ],
                },
            }
        ]
    }
)
rule2 = amanda.create_rule(
    {
        "ops": [
            {
                "type": {
                    "src": "store_tensor_to_file",
                    "dst": "amanda::store_tensor_to_file",
                }
            }
        ]
    }
)
table = amanda.get_mapping_table("debugging", "amanda/pytorch")
table.insert_rule(rule1)
table.insert_rule(rule2, index=1)

rule = amanda.create_rule(
    {
        "ops": [
            {
                "output_ports": [
                    {
                        "tensor": {
                            "attrs": {
                                "src": [{"key": "dtype", "ref": "dtype"}],
                                "dst": [
                                    {"ref": "dtype"},
                                    {
                                        "key": "is_valid",
                                        "value": "{dtype.value._is_ref_dtype}",
                                    },
                                ],
                            }
                        }
                    }
                ]
            }
        ]
    }
)
table = amanda.get_mapping_table("amanda/tensorflow", "debugging")
table.insert_rule(rule)

rule1 = amanda.create_rule(
    {
        "ops": {
            "src": [{"type": "store_tensor_to_file", "ref": "debug_op"}],
            "dst": [
                {
                    "ref": "debug_op",
                    "name": "{op.input_ops[0].name + op.output_ops[0].name}",
                    "attrs": [
                        {"key": "T", "value": "{op.input_tensors[0].attrs['dtype']}"}
                    ],
                }
            ],
        }
    }
)
rule2 = amanda.create_rule(
    {"ops": [{"type": {"src": "store_tensor_to_file", "dst": "StoreTensorToFile"}}]}
)
table = amanda.get_mapping_table("debugging", "amanda/tensorflow")
table.insert_rule(rule1)
table.insert_rule(rule2, index=1)

amanda.save_mapping_tables("tables.yaml")
