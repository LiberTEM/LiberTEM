# Continuous benchmarks

This directory contains all continuous benchmarks, which can be run for pull requests and which will be run for each push into the master branch.

When adding benchmarks here, be mindful of execution time, and try to adhere to these rules:

- For keeping benchmarks comparable, we need good statistics, so only add benchmarks that have rounds below approx. 200ms, meaning we can run five rounds in a second. If possible, keep it well below that number.
- Be very careful when adding `pedantic` benchmarks - removing warmup or reducing the number of rounds will increase noise.
- The general run time will be approx. proportional to the number of benchmarks, as even very fast benchmarks are executed for a minimum of one second, so be mindful of the number of benchmarks and variants via parametrization.