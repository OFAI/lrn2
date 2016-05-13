"""
The nn_bricks package contains different 'bricks' for defining NN layers.
There are different unit types, cost functions, regularizations, plot functions,
monitoring types and training methods.

From those bricks, NN layers can be defined by deriving from multiple bricks.
Those mix-in classes can be further combined to stack objects, which can again
be defined by deriving from NN bricks. See make.py for an example.
"""