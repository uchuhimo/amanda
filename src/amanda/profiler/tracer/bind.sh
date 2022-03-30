cmake -B build
cmake --build build

cp tracer_pybind_test.py build/
python3 build/tracer_pybind_test.py