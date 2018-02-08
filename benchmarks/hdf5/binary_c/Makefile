CC=gcc
CFLAGS=-g -Wall -O4 -march=native
PERF_STAT_FLAGS=-e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,page-faults,context-switches

all: bench create_file

bench: bench.c

create_file: create_file.c

run_bench: bench
	perf stat ${PERF_STAT_FLAGS} ./bench

record_bench: bench
	perf record --call-graph lbr -e task-clock,cycles,instructions,cache-references,cache-misses ./bench

clean:
	rm -f bench create_file

.PHONY: clean run
