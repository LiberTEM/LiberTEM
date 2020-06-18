CC=cc
CFLAGS=-g -Wall -O4 -march=native
PERF_STAT_FLAGS=-e task-clock,cycles,instructions,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,page-faults,context-switches

all: bench

bench: bench.c

run_bench: bench
	perf stat ${PERF_STAT_FLAGS} ./bench > results-$(shell date -Iminutes).csv

clean:
	rm -f bench

.PHONY: clean run
