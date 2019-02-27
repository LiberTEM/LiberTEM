EASYJIT_PATH=/home/alex/source/easy-just-in-time/
CXX=clang++-5.0 
CXXFLAGS=-g -Wall -O3 -march=native -fPIC --std=c++14 \
  -Xclang -load -Xclang $(EASYJIT_PATH)/build/bin/EasyJitPass.so\
  -I$(EASYJIT_PATH)/include \
  -L$(EASYJIT_PATH)/build/bin -lEasyJitRuntime
  #-fsanitize=undefined\

PERF_STAT_FLAGS=-e task-clock,cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,page-faults,context-switches

all: bm2 maskjit.cpython-36m-x86_64-linux-gnu.so

bm2: bm2.o
	$(CXX) $(CXXFLAGS) -o bm2 bm2.o bm2_run.cpp

bm2.o: bm2.cpp
	$(CXX) $(CXXFLAGS) -c -o bm2.o bm2.cpp

run_bm2: bm2
	perf stat ${PERF_STAT_FLAGS} ./bm2

record: bm2
	perf record ${PERF_STAT_FLAGS} ./bm2

maskjit.cpython-36m-x86_64-linux-gnu.so: bm2.o bm.pyx
	rm -f maskjit.cpython-36m-x86_64-linux-gnu.so
	python setup.py build_ext

clean:
	rm -f bm2 bm2.o maskjit.cpython-36m-x86_64-linux-gnu.so bm.cpp

.PHONY: clean run_bm2 record
