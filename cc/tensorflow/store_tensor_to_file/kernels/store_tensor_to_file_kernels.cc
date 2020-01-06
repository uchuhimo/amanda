/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fstream>
#include <string>
#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

template <typename T>
class StoreTensorToFileOp : public OpKernel {
 public:
  explicit StoreTensorToFileOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("store_dir", &store_dir_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("file_name", &file_name_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    output_tensor->CopyFrom(input_tensor, input_tensor.shape());

    std::string filename = store_dir_ + "/" + file_name_;
    std::ofstream write;
    write.open(filename.c_str());
    write << input_tensor.DebugString();
    write.close();
  }

  private:
  std::string store_dir_;
  std::string file_name_;
};

// see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
#define REGISTER_KERNEL(type)                                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("StoreTensorToFile").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      StoreTensorToFileOp<type>)
TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
