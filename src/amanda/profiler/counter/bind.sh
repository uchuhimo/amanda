cmake -B build
cmake --build build

cp counter_pybind_test.py build/
python3 build/counter_pybind_test.py