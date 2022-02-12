import numpy

def square(x: numpy.ndarray) -> numpy.ndarray:
    '''
    Square function
    For x real, f(x) = x² 
    '''
    return numpy.power(x, 2)

def leaky_relu(x: numpy.ndarray) -> numpy.ndarray:
    '''
    Leaky Rectified Unit function
    f(x) = x if x > 0.01
    f(x) = 0.01x otherwise
    Equivalent to f(x) = max(0.01*x, x)
    ''' 
    return numpy.maximum(x*0.01, x)

a = numpy.array([-6, -5, -4, -3, -2, -1, -0.01, -0.009, 1,2,3,4,5,6])

print("square(a)=", square(a))
print("leaky_relu(a)=", leaky_relu(a))  