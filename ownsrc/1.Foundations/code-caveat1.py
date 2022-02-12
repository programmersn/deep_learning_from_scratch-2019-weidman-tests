import numpy
from numpy.lib.index_tricks import AxisConcatenator

a = [1, 2, 3]
b = [4, 5, 6]
print("a+b:", a+b)
#print("a*b:", a*b)

a = numpy.array([1, 2, 3])
b = numpy.array([4, 5, 6])
print("a+b:", a+b)
print("a*b:", a*b)

a = numpy.array([[1, 2],
                 [3, 4]])
print("a.sum(axis=0)", a.sum(axis=0))
print("a.sum(axis=1)", a.sum(axis=1))

a = numpy.array([[1,2,3],
                 [4,5,6]])
b = numpy.array([10, 20, 30, 40])
print("a+b=", a+b)