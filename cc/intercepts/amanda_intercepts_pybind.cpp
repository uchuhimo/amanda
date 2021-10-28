#include <Python.h>
#include <ffi.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

uintptr_t addr(py::object func) {
  return reinterpret_cast<uintptr_t>(func.ptr());
}

void builtin_handler_ffi(ffi_cif *cif, void *ret, void *args[],
                         void *py_handler) {
  PyObject *self = *(PyObject **)args[0];
  PyObject *handler_args = *(PyObject **)args[1];
  PyObject *kwargs = *(PyObject **)args[2];
  PyObject *result =
      PyObject_Call((PyObject *)py_handler, handler_args, kwargs);
  *((PyObject **)ret) = result;
}

void method_descriptor_handler_ffi(ffi_cif *cif, void *ret, void *args[],
                                   void *py_handler) {
  PyObject *self = *(PyObject **)args[0];
  PyObject *handler_args = *(PyObject **)args[1];
  PyObject *kwargs = *(PyObject **)args[2];
  Py_ssize_t size_args = PyTuple_Size(handler_args);
  PyObject *new_args = PyTuple_New(size_args + 1);
  for (int i = 0; i < size_args; i++) {
    PyObject *arg = PyTuple_GetItem(handler_args, i);
    Py_XINCREF(arg);
    PyTuple_SetItem(new_args, i + 1, arg);
  }
  Py_XINCREF(self);
  PyTuple_SetItem(new_args, 0, self);
  PyObject *result = PyObject_Call((PyObject *)py_handler, new_args, kwargs);
  Py_DECREF(new_args);
  *((PyObject **)ret) = result;
}

void getter_handler_ffi(ffi_cif *cif, void *ret, void *args[],
                        void *py_handler) {
  PyObject *self = *(PyObject **)args[0];
  PyObject *closure = *(PyObject **)args[1];
  PyObject *new_args = PyTuple_New(1);
  Py_XINCREF(self);
  PyTuple_SetItem(new_args, 0, self);
  PyObject *result = PyObject_Call((PyObject *)py_handler, new_args, nullptr);
  Py_DECREF(new_args);
  *((PyObject **)ret) = result;
}

void setter_handler_ffi(ffi_cif *cif, void *ret, void *args[],
                        void *py_handler) {
  PyObject *self = *(PyObject **)args[0];
  PyObject *value = *(PyObject **)args[1];
  PyObject *closure = *(PyObject **)args[2];
  PyObject *new_args = PyTuple_New(2);
  Py_XINCREF(self);
  PyTuple_SetItem(new_args, 0, self);
  Py_XINCREF(value);
  PyTuple_SetItem(new_args, 1, value);
  PyObject *result = PyObject_Call((PyObject *)py_handler, new_args, nullptr);
  Py_DECREF(new_args);
  if (result) {
    *((int *)ret) = 0;
  } else {
    *((int *)ret) = -1;
  }
}

PyCFunctionWithKeywords get_func_handler(void (*handler_ffi)(ffi_cif *, void *,
                                                             void **, void *),
                                         PyObject *py_handler) {
  PyCFunctionWithKeywords handler;
  ffi_closure *closure =
      (ffi_closure *)ffi_closure_alloc(sizeof(ffi_closure), (void **)&handler);
  if (closure) {
    ffi_type **args = new ffi_type *[3];
    args[0] = &ffi_type_pointer;
    args[1] = &ffi_type_pointer;
    args[2] = &ffi_type_pointer;

    ffi_cif *cif = new ffi_cif();
    if (ffi_prep_cif(cif, FFI_DEFAULT_ABI, 3, &ffi_type_pointer, args) ==
        FFI_OK) {
      if (ffi_prep_closure_loc(closure, cif, handler_ffi, py_handler,
                               (void *)handler) == FFI_OK) {
        return handler;
      }
    }
  }
  return nullptr;
}

getter get_getter_handler(void (*handler_ffi)(ffi_cif *, void *, void **,
                                              void *),
                          PyObject *py_handler) {
  getter handler;
  ffi_closure *closure =
      (ffi_closure *)ffi_closure_alloc(sizeof(ffi_closure), (void **)&handler);
  if (closure) {
    ffi_type **args = new ffi_type *[2];
    args[0] = &ffi_type_pointer;
    args[1] = &ffi_type_pointer;

    ffi_cif *cif = new ffi_cif();
    if (ffi_prep_cif(cif, FFI_DEFAULT_ABI, 2, &ffi_type_pointer, args) ==
        FFI_OK) {
      if (ffi_prep_closure_loc(closure, cif, handler_ffi, py_handler,
                               (void *)handler) == FFI_OK) {
        return handler;
      }
    }
  }
  return nullptr;
}

setter get_setter_handler(void (*handler_ffi)(ffi_cif *, void *, void **,
                                              void *),
                          PyObject *py_handler) {
  setter handler;
  ffi_closure *closure =
      (ffi_closure *)ffi_closure_alloc(sizeof(ffi_closure), (void **)&handler);
  if (closure) {
    ffi_type **args = new ffi_type *[3];
    args[0] = &ffi_type_pointer;
    args[1] = &ffi_type_pointer;
    args[2] = &ffi_type_pointer;

    ffi_cif *cif = new ffi_cif();
    if (ffi_prep_cif(cif, FFI_DEFAULT_ABI, 3, &ffi_type_sint, args) == FFI_OK) {
      if (ffi_prep_closure_loc(closure, cif, handler_ffi, py_handler,
                               (void *)handler) == FFI_OK) {
        return handler;
      }
    }
  }
  return nullptr;
}

py::object get_builtin_handler(uintptr_t func_id, py::object py_handler) {
  auto handler =
      get_func_handler(builtin_handler_ffi, py_handler.inc_ref().ptr());
  if (handler) {
    PyCFunctionObject *func = (PyCFunctionObject *)func_id;

    PyMethodDef *handler_def = new PyMethodDef();
    handler_def->ml_name = func->m_ml->ml_name;
    handler_def->ml_meth = (PyCFunction)handler;
    handler_def->ml_flags = METH_VARARGS | METH_KEYWORDS;
    handler_def->ml_doc = func->m_ml->ml_doc;

    PyObject *fn = PyCFunction_NewEx(handler_def, func->m_self, func->m_module);
    py::object pybind_fn = py::reinterpret_steal<py::object>(fn);
    return pybind_fn;
  }
  return py::none();
}

py::object get_method_descriptor_handler(uintptr_t func_id,
                                         py::object py_handler) {
  auto handler = get_func_handler(method_descriptor_handler_ffi,
                                  py_handler.inc_ref().ptr());
  if (handler) {
    PyMethodDescrObject *func = (PyMethodDescrObject *)func_id;

    PyMethodDef *handler_def = new PyMethodDef();
    handler_def->ml_name = func->d_method->ml_name;
    handler_def->ml_meth = (PyCFunction)handler;
    handler_def->ml_flags = METH_VARARGS | METH_KEYWORDS;
    handler_def->ml_doc = func->d_method->ml_doc;

    PyObject *fn = PyDescr_NewMethod(func->d_common.d_type, handler_def);
    py::object pybind_fn = py::reinterpret_steal<py::object>(fn);
    return pybind_fn;
  }
  return py::none();
}

py::object get_getset_descriptor_handler(uintptr_t func_id,
                                         py::object py_getter_handler,
                                         py::object py_setter_handler) {
  getter getter_handler =
      get_getter_handler(getter_handler_ffi, py_getter_handler.inc_ref().ptr());
  setter setter_handler =
      get_setter_handler(setter_handler_ffi, py_setter_handler.inc_ref().ptr());
  if (getter_handler && setter_handler) {
    PyGetSetDescrObject *func = (PyGetSetDescrObject *)func_id;

    PyGetSetDef *handler_def = new PyGetSetDef();
    handler_def->name = func->d_getset->name;
    handler_def->get = getter_handler;
    handler_def->set = setter_handler;
    handler_def->doc = func->d_getset->doc;
    handler_def->closure = func->d_getset->closure;

    PyObject *fn = PyDescr_NewGetSet(func->d_common.d_type, handler_def);
    py::object pybind_fn = py::reinterpret_steal<py::object>(fn);
    return pybind_fn;
  }
  return py::none();
}

PYBIND11_MODULE(amanda_intercepts_pybind, m) {
  m.doc() = "extra binding for amanda.intercepts";
  m.def("addr", &addr, "get function's address");
  m.def("get_builtin_handler", &get_builtin_handler, "get builtin handler");
  m.def("get_method_descriptor_handler", &get_method_descriptor_handler,
        "get method descriptor handler");
  m.def("get_getset_descriptor_handler", &get_getset_descriptor_handler,
        "get getset descriptor handler");
}
