import tensorflow as tf

import amanda
from examples.common.tensorflow.model.resnet_50 import ResNet50
from examples.effective_path.tensorflow.effective_path_tool import EffectivePathTool


def graph_traverse(graph, entrypoint):
    def _dfs(op):
        print(op.raw_op.name)
        visited.append(op)
        for input_op in op.input_ops:
            if input_op in visited:
                continue
            _dfs(input_op)

    for op in graph.ops:
        if op.raw_op.name == entrypoint:
            visited = list()
            _dfs(op)
            break


def main():
    model = ResNet50()
    x = tf.random.uniform(shape=[2, 227, 227, 3])

    tool = EffectivePathTool()

    with amanda.tool.apply(tool):
        y = model(x)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            session.run(y)

    graph_traverse(tool.graph, y.op.name)


if __name__ == "__main__":
    main()
