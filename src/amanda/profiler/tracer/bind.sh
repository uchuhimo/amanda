cmake -B build
cmake --build build

cp  build/tracer.cpython-37m-x86_64-linux-gnu.so .
python3 tracer_pybind_test.py