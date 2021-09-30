[Bugfix] Fix memory leak
========================

* Don't submit dynamically generated callables directly to the distributed cluster,
  as they are cached in an unbounded cache (:issue:`894`, :pr:`1119`).
