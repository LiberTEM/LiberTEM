CXX=clang++
CC=clang
# ARCH=skylake-avx512  # for moellenstedt
ARCH=native
CXXFLAGS=-g -Wall -O3 -march=$(ARCH) -mtune=$(ARCH) -fPIC
CFLAGS=-g -Wall -O3 -march=$(ARCH) -mtune=$(ARCH) -fPIC
PERF_STAT_FLAGS=-e task-clock,cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,page-faults,context-switches

all: u12decode.o u12mod.cpython-36m-x86_64-linux-gnu.so driver

u12decode.o: u12decode.cpp

driver: driver.cpp u12decode.o
	$(CXX) u12decode.o driver.cpp -o driver $(CXXFLAGS)

u12mod.cpython-36m-x86_64-linux-gnu.so: u12decode.o u12mod.pyx
	rm -f u12mod.cpython-36m-x86_64-linux-gnu.so
	python setup.py build_ext

clean:
	rm -f u12mod.cpython-36m-x86_64-linux-gnu.so *.o driver

.PHONY: clean
