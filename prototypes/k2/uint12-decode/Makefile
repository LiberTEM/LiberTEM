all: unpack-12-6 unpack-12-4 unpack-12-4-1

unpack-12-4: unpack-12-4.cpp expected.cpp harness.cpp harness.h
	gcc -Wall -o unpack-12-4 expected.cpp unpack-12-4.cpp harness.cpp

unpack-12-4-1: unpack-12-4.cpp expected.cpp harness_ones.cpp harness.h
	gcc -Wall -o unpack-12-4-1 expected.cpp unpack-12-4.cpp harness_ones.cpp

unpack-12-6: unpack-12-6.cpp expected.cpp harness.cpp harness.h
	gcc -Wall -o unpack-12-6 expected.cpp unpack-12-6.cpp harness.cpp

clean:
	rm -f unpack-12-4 unpack-12-6 unpack-12-4-1
