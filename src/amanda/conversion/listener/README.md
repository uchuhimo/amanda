# code description
- `listener\` contains src for pybind listener.
  - build it first
```
cmake -DCMAKE_PREFIX_PATH=`python -c "import torch;print(torch.utils.cmake_prefix_path)"`\;`python -m pybind11 --cmakedir` -S src/amanda/conversion/listener -B src/amanda/conversion/listener/build
cmake --build src/amanda/conversion/listener/build
```
