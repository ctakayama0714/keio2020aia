## import numpy
import numpy as np
##

## define two random matrices
a = np.random.randn(50, 200)
b = np.random.randn(200, 20)
##

##
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i,j] += a[i,k] * b[k,j]
    return c
##

## faster version with elementwise array operations
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        for j in range(bc):
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
##

## broadcasting
x = np.array(list(range(5)))
y = np.array(list(range(25))).reshape((5, 5)) + 1
z = y - x
x2 = np.broadcast_arrays(x, y)[0]

x3 = np.expand_dims(x, axis=0)
x4 = np.expand_dims(x, axis=1)

x3 = x[None,:]
x4 = x[:,None]

x5 = np.squeeze(x3)

y2 = y[2,:]
##

## faster matrix multiplication with a single loop and broadcasting
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        c[i] = (a[i,:,None] * b).sum()
    return c
##

## matrix multiplication using Einstein summation
matmul = lambda a, b: np.einsum('ik,kj->ij', a, b)
##

## fastest version using BLAS (Basic Linear Algebra Subprogram)
matmul = np.matmul
##

## recursion / list comprehension riddle
def f(l):
    if not l:
        return [[]]
    return [x for y in f(l[1:]) for x in [y[:i]+[l[0]]+y[i:] for i in range(len(y)+1)]]
##

# Question: what does f compute, given some input list l?


