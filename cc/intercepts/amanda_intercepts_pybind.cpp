#include <pybind11/pybind11.h>
#include <Python.h>

namespace py = pybind11;

static PyObject *intercept_handler(PyObject *self, PyObject *args, PyObject *kwargs)
{
    if (self == NULL)
    {
        printf("null self\n");
        Py_RETURN_NONE;
    }
    char *ptr;
    size_t func_addr = strtoul(PyModule_GetName(self), &ptr, 10);

    PyObject *module_dict = PyImport_GetModuleDict();
    PyObject *intercepts_module = PyDict_GetItemString(module_dict, "amanda.intercepts");
    if (PyErr_Occurred())
        return NULL;
    PyObject *intercept_handler = PyObject_GetAttrString(intercepts_module, "_intercept_handler");
    if (PyErr_Occurred())
    {
        return NULL;
    }
    Py_INCREF(intercept_handler);

    PyTupleObject *co_consts = (PyTupleObject *)(((PyCodeObject *)(((PyFunctionObject *)intercept_handler)->func_code))->co_consts);
    Py_ssize_t size_co_consts = PyTuple_Size((PyObject *)co_consts);

    PyTupleObject *new_co_consts = (PyTupleObject *)PyTuple_New(size_co_consts + 1);
    for (int i = 0; i < size_co_consts; i++)
    {
        PyTuple_SetItem(
            (PyObject *)new_co_consts,
            i,
            PyTuple_GetItem((PyObject *)co_consts, i));
    }
    PyObject *func_id = Py_BuildValue("K", func_addr);
    PyTuple_SetItem(
        (PyObject *)new_co_consts,
        size_co_consts,
        func_id);

    ((PyCodeObject *)(((PyFunctionObject *)intercept_handler)->func_code))->co_consts = (PyObject *)new_co_consts;
    PyObject *result = PyObject_Call(
        (PyObject *)intercept_handler,
        args,
        kwargs);
    ((PyCodeObject *)(((PyFunctionObject *)intercept_handler)->func_code))->co_consts = (PyObject *)co_consts;
    Py_DECREF(intercept_handler);

    return result;
}

py::object get_builtin_handler(size_t func_id)
{
    const char *func_name = ((PyCFunctionObject *)func_id)->m_ml->ml_name;
    const char *func_doc = ((PyCFunctionObject *)func_id)->m_ml->ml_doc;

    static PyMethodDef handler_def;
    handler_def.ml_name = func_name;
    handler_def.ml_meth = (PyCFunction)intercept_handler;
    handler_def.ml_flags = METH_VARARGS | METH_KEYWORDS;
    handler_def.ml_doc = func_doc;

    char func_id_str[32];
    sprintf(func_id_str, "%lu", func_id);
    PyObject *new_module = PyModule_New((const char *)func_id_str);
    PyObject *fn = PyCFunction_NewEx(
        &handler_def,
        (PyObject *)new_module,
        NULL);
    Py_DECREF(new_module);
    auto pybind_fn = py::reinterpret_steal<py::object>(fn);
    return pybind_fn;
}

py::object get_method_descriptor_handler(size_t func_id)
{
    const char *func_name = ((PyMethodDescrObject *)func_id)->d_method->ml_name;
    const char *func_doc = ((PyMethodDescrObject *)func_id)->d_method->ml_doc;

    static PyMethodDef handler_def;
    handler_def.ml_name = func_name;
    handler_def.ml_meth = (PyCFunction)intercept_handler;
    handler_def.ml_flags = METH_VARARGS | METH_KEYWORDS;
    handler_def.ml_doc = func_doc;

    PyObject *fn = PyDescr_NewMethod(
        ((PyMethodDescrObject *)func_id)->d_common.d_type,
        &handler_def);
    auto pybind_fn = py::reinterpret_steal<py::object>(fn);
    return pybind_fn;
}

PYBIND11_MODULE(amanda_intercepts_pybind, m)
{
    m.doc() = "extra binding for amanda";

    m.def("get_method_descriptor_handler", &get_method_descriptor_handler, "function to get method descriptor handler");
    m.def("get_builtin_handler", &get_builtin_handler, "function to get builtin handler");
}
