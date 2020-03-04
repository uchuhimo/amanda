# type: ignore
import amanda.tools.debugging.graph as amanda

if __name__ == "__main__":
    rule = amanda.create_rule({})
    table = amanda.get_mapping_table("amanda/pytorch", "debugging")
    table.insert_rule(rule)

    rule1 = amanda.create_rule({})
    rule2 = amanda.create_rule({})
    table = amanda.get_mapping_table("debugging", "amanda/pytorch")
    table.insert_rule(rule1)
    table.insert_rule(rule2, index=1)

    rule = amanda.create_rule({})
    table = amanda.get_mapping_table("amanda/tensorflow", "debugging")
    table.insert_rule(rule)

    rule1 = amanda.create_rule({})
    rule2 = amanda.create_rule({})
    table = amanda.get_mapping_table("debugging", "amanda/tensorflow")
    table.insert_rule(rule1)
    table.insert_rule(rule2, index=1)

    amanda.save_mapping_tables("tables.yaml")
