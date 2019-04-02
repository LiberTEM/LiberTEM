CXX=g++-8
CC=gcc-8
CXXFLAGS=-g -Wall -O3 -march=native -fPIC
CFLAGS=-g -Wall -O3 -march=native -fPIC
PERF_STAT_FLAGS=-e task-clock,cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,page-faults,context-switches

all: bm1 bm2 test.cpython-36m-x86_64-linux-gnu.so

bm1: bm1.cpp

bm2: bm2.cpp

bm3.o: bm3.c

bm4.o: bm4.c

run_bm1: bm1
	perf stat ${PERF_STAT_FLAGS} ./bm1

record: bm1
	perf record ${PERF_STAT_FLAGS} ./bm1

test.cpython-36m-x86_64-linux-gnu.so: bm4.o
	rm -f test.cpython-36m-x86_64-linux-gnu.so
	python setup.py build_ext

clean:
	rm -f bm1 bm2 test.cpython-36m-x86_64-linux-gnu.so bm3.o

.PHONY: clean run_bm1
