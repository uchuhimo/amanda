#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/extension.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

using torch::autograd::variable_list;

void init_THPVariableClass()
{
    auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensor"));
    THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
}

static PyObject* amanda_wrap_variables(const variable_list& c_variables)
{
    size_t num_vars = c_variables.size();
    THPObjectPtr tuple(PyTuple_New(num_vars));
    if (!tuple) throw python_error();
    for (size_t i = 0; i < num_vars; ++i) {
        THPObjectPtr var(THPVariable_Wrap(c_variables[i]));
        if (!var) throw python_error();
        PyTuple_SET_ITEM(tuple.get(), i, var.release());
    }
    return tuple.release();

}

static variable_list amanda_unwrap_variables(PyObject *py_variables)
{
    variable_list results(PyTuple_GET_SIZE(py_variables));
    for (size_t i = 0; i < results.size(); i++) {
        PyObject* item = PyTuple_GET_ITEM(py_variables, i);
        if (item == Py_None) {
            continue;
        } else if (THPVariable_Check(item)) {
            results[i] = ((THPVariable*)item)->cdata;
        } else {
            // this should never happen, but just in case...
            std::stringstream ss;
            ss << "expected variable but got " << Py_TYPE(item)->tp_name;
            throw std::runtime_error(ss.str());
        }
    }
    return results;
}

class AmandaPreHook : public torch::autograd::FunctionPreHook
{
public:
    AmandaPreHook(PyObject* fn): fn_(fn) {
        Py_INCREF(fn_);
    }

    ~AmandaPreHook()
    {
        pybind11::gil_scoped_acquire gil;
        Py_DECREF(fn_);
    }

    variable_list operator()(const variable_list& _inputs) override
    {
        pybind11::gil_scoped_acquire gil;
        // wrap cpp vector<torch::autograd::Variable> _inputs -> PyObject inputs
        THPObjectPtr inputs(amanda_wrap_variables(_inputs));
        // call python function from cpp as PyObject
        THPObjectPtr res(PyObject_CallFunctionObjArgs(fn_, inputs.get(), nullptr));
        // unwarp PyObject into vector<torch::autograd::Variable>
        return amanda_unwrap_variables(res.get());
    }

protected:
    PyObject* fn_;
};

int amanda_add_pre_hook(const pybind11::object &grad_fn, const pybind11::object & hook)
{
    PyObject *raw_grad_fn = grad_fn.ptr();
    PyObject *raw_hook = hook.ptr();
    torch::autograd::THPCppFunction *cast_grad_fn = (torch::autograd::THPCppFunction *)raw_grad_fn;
    auto final_grad_fn = cast_grad_fn->cdata;
    std::unique_ptr<torch::autograd::FunctionPreHook> pre_hook(new AmandaPreHook(raw_hook));
    final_grad_fn->add_pre_hook(std::move(pre_hook));
    return final_grad_fn->pre_hooks().size() - 1;
}

void amanda_remove_pre_hook(const pybind11::object &grad_fn, int pos)
{
    PyObject *raw_grad_fn = grad_fn.ptr();
    torch::autograd::THPCppFunction *cast_grad_fn = (torch::autograd::THPCppFunction *)raw_grad_fn;
    auto final_grad_fn = cast_grad_fn->cdata;
    final_grad_fn->pre_hooks().erase(final_grad_fn->pre_hooks().begin() + pos);
}
