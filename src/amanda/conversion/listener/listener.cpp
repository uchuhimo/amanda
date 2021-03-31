#include <torch/torch.h>

#include <iostream>
#include <vector>

struct HookOpRegistrationListener : public c10::OpRegistrationListener {

  std::function<std::string (std::string)> hook_function;

  HookOpRegistrationListener(const std::function<std::string(std::string)>  &py_hook_function) {
    hook_function = py_hook_function;
  }


  void onOperatorRegistered(const c10::OperatorHandle& op) override {
     std::string op_name = hook_function(op.schema().operator_name().name);
    //  std::cout << op.schema().operator_name() << '\t' << op_name << std::endl;
  }
  void onOperatorDeregistered(const c10::OperatorHandle& op) override {

  }
};


std::string test_py_hook_function(std::string name){
auto pos = name.find("::");
if (pos == std::string::npos) {
    return name;
} else {
    return name.substr(pos);
}    
}


struct HookRegisterer final {
  HookRegisterer(const std::function<std::string(std::string)>  &py_function) {
    std::unique_ptr<HookOpRegistrationListener> listener = std::make_unique<HookOpRegistrationListener>(py_function);
    c10::Dispatcher::singleton().addRegistrationListener(std::move(listener));         
  }
};

void test_hook_listener(){
    HookRegisterer test_listener(test_py_hook_function);
}

int main() {

    test_hook_listener();

    return 0;
}