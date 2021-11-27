#include <pybind11/pybind11.h>
#include <tensorflow/core/common_runtime/shape_refiner.h>
#include <tensorflow/core/graph/graph.h>

typedef struct {
  PyObject_HEAD void *ptr; // This is the pointer to the actual C++ obj.
  void *ty;
  int own;
  PyObject *next;
  PyObject *dict;
} SwigPyObject;

struct TF_Graph {
  tensorflow::mutex mu;
  tensorflow::Graph graph GUARDED_BY(mu);
  tensorflow::ShapeRefiner refiner GUARDED_BY(mu);
  std::unordered_map<tensorflow::string, tensorflow::Node *>
      name_map GUARDED_BY(mu);
};

struct TF_Operation {
  tensorflow::Node node;
};

void remove_op(const pybind11::object &graph, const pybind11::object &op) {
  TF_Graph *c_graph = (TF_Graph *)((SwigPyObject *)graph.ptr())->ptr;
  TF_Operation *c_op = (TF_Operation *)((SwigPyObject *)op.ptr())->ptr;
  tensorflow::mutex_lock l(c_graph->mu);
  c_graph->name_map.erase(c_op->node.name());
  c_graph->graph.RemoveNode(&c_op->node);
}

PYBIND11_MODULE(amanda_tf_pybind, m) {
  m.doc() = "extra binding for Amanda on TensorFlow";
  m.def("remove_op", &remove_op, "remove op from graph");
}
