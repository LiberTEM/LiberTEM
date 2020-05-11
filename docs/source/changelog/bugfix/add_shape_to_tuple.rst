[Bugfix] Addition between Shape and tuple
===========================================================

* Tuples can be added directly to Shape objects. Right
  addition adds to the signal dimensions of the Shape
  object while left addition adds to the navigation
  dimensions (:pr:`749`)
* Added tests to test_shape.py
