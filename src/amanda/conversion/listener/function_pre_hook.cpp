#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <torch/torch.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

using torch::autograd::variable_list;

template<class T>
class amanda_THPPointer {
public:
    amanda_THPPointer(): ptr(nullptr) {};
    explicit amanda_THPPointer(T *ptr) noexcept : ptr(ptr) {};
    amanda_THPPointer(amanda_THPPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; };

    ~amanda_THPPointer() { free(); };
    T * get() { return ptr; }
    const T * get() const { return ptr; }
    T * release() { T *tmp = ptr; ptr = nullptr; return tmp; }
    operator T*() { return ptr; }
    amanda_THPPointer& operator =(T *new_ptr) noexcept { free(); ptr = new_ptr; return *this; }
    amanda_THPPointer& operator =(amanda_THPPointer &&p) noexcept { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
    T * operator ->() { return ptr; }
    explicit operator bool() const { return ptr != nullptr; }

private:
    void free() {if (ptr) Py_DECREF(ptr);}
    T *ptr = nullptr;
};

auto tensor_module = amanda_THPPointer<PyObject>(PyImport_ImportModule("torch.tensor"));
PyObject* THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");

PyObject* amanda_THPVariable_NewWithVar(PyTypeObject* type, torch::autograd::Variable var)
{
    PyObject* obj = type->tp_alloc(type, 0);
    if (obj) {
        auto v = (THPVariable*) obj;
        new (&v->cdata) torch::autograd::Variable(std::move(var));
        torch::autograd::impl::set_pyobj(v->cdata, obj);
    }
    return obj;
}

PyObject * amanda_THPVariable_Wrap(torch::autograd::Variable var)
{
    if (!var.defined()) {
        Py_RETURN_NONE;
    }

    if (auto obj = torch::autograd::impl::pyobj(var)) {
        Py_INCREF(obj);
        return obj;
    }

    return amanda_THPVariable_NewWithVar((PyTypeObject *)THPVariableClass, std::move(var));
}

PyObject* amanda_wrap_variables(const variable_list& c_variables)
{
    size_t num_vars = c_variables.size();
    amanda_THPPointer<PyObject> tuple(PyTuple_New(num_vars));
    if (!tuple) throw python_error();
    for (size_t i = 0; i < num_vars; ++i) {
        amanda_THPPointer<PyObject> var(amanda_THPVariable_Wrap(c_variables[i]));
        if (!var) throw python_error();
        PyTuple_SET_ITEM(tuple.get(), i, var.release());
    }
    return tuple.release();

}

inline bool amanda_THPVariable_Check(PyObject *obj)
{
    return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

variable_list amanda_unwrap_variables(PyObject* py_variables)  {
    variable_list results(PyTuple_GET_SIZE(py_variables));
    for (size_t i = 0; i < results.size(); i++) {
        PyObject* item = PyTuple_GET_ITEM(py_variables, i);
        if (item == Py_None) {
            continue;
        } else if (amanda_THPVariable_Check(item)) {
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
    AmandaPreHook(PyObject* fn): fn_(fn) {}

    variable_list operator()(const variable_list& _inputs) override
    {
        pybind11::gil_scoped_acquire gil;
        // wrap cpp vector<torch::autograd::Variable> _inputs -> PyObject inputs
        amanda_THPPointer<PyObject> inputs(amanda_wrap_variables(_inputs));
        // call python function from cpp as PyObject
        amanda_THPPointer<PyObject> res(PyObject_CallFunctionObjArgs(fn_, inputs.get(), nullptr));
        // unwarp PyObject into vector<torch::autograd::Variable>
        amanda_THPPointer<PyObject> outputs = std::move(res);
        return amanda_unwrap_variables(outputs.get());
    }

protected:
    PyObject* fn_;
};

// int add_pre_hook(const std::shared_ptr<torch::autograd::Node> grad_fn, PyObject* hook)
// {
//     std::unique_ptr<torch::autograd::FunctionPreHook> pre_hook(new AmandaPreHook(hook));
//     grad_fn->add_pre_hook(std::move(pre_hook));
//     return grad_fn->pre_hooks().size() - 1;
// }

// void remove_pre_hook(const std::shared_ptr<torch::autograd::Node> &grad_fn, int pos)
// {
//     grad_fn->pre_hooks().erase(grad_fn->pre_hooks().begin() + pos);
// }

// void py_add_pre_hook(const pybind11::object &grad_fn, const pybind11::object & hook)
// {
//     PyObject *raw_grad_fn = grad_fn.ptr();
//     PyObject *raw_hook = hook.ptr();
//     torch::autograd::THPCppFunction *cast_grad_fn = (torch::autograd::THPCppFunction *)raw_grad_fn;
//     auto final_grad_fn = cast_grad_fn->cdata;
//     int handle = add_pre_hook(final_grad_fn, raw_hook);
//     // int handle = add_pre_hook(final_grad_fn, dummy_pre_hook);
//     // std::cout << cast_grad_fn->cdata->pre_hooks().size() << std::endl;
// }

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


// PYBIND11_MODULE(amanda, m)
// {
//     m.doc() = "amanda binding for function pre hook";

//     //   pybind11::class_<HookRegisterer>(m, "HookRegisterer")
//     // .def(pybind11::init<const std::function<std::string(std::string)> &>());
//     m.def("add_pre_hook", &add_pre_hook, "function to add a hook");
//     m.def("remove_pre_hook", &remove_pre_hook, "function to remove hook by handle");
//     m.def("py_add_pre_hook", &py_add_pre_hook, "duumy test");
// }
