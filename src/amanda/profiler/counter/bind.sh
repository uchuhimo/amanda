cmake -B build
cmake --build build

cp build/counter.cpython-37m-x86_64-linux-gnu.so .
python3 counter_pybind_test.py