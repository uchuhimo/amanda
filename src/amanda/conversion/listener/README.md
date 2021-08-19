# code description
- `listener\` contains src for pybind listener.
  - build it first
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c "import torch;print(torch.utils.cmake_prefix_path)"`\;`python -m pybind11 --cmakedir` ..
make listener
```
<!-- cmake -DPYBIND11_PYTHON_VERSION=3.7 -DPYTHON_LIBRARY=/home/yguan/anaconda3/envs/amanda/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=/home/yguan/anaconda3/envs/amanda/bin/python3.7m  -DCMAKE_PREFIX_PATH=`python -c "import torch;print(torch.utils.cmake_prefix_path)"`\;`python -m pybind11 --cmakedir` .. -->
