>>> import numpy as np
>>> a = np.array([1, 2, 3])
>>> print type(a)
<type 'numpy.ndarray'>
>>> a.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object is not callable
>>> print a.shape
(3,)
>>> print a[0]
1
>>> print a[1]
2
>>> print a[3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
>>> print a[2]
3
>>> print a[0][0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: invalid index to scalar variable.
>>> a[0] = 5
>>> print a
[5 2 3]
>>> print a.shape[0]
3
>>> print a[a.shape[0]-1]
3
>>> b = np.array([[1, 2, 3], [4, 5, 6]])
>>> b
array([[1, 2, 3],
       [4, 5, 6]])
>>> b.shape
(2, 3)
>>> print b[0, 0]
1
>>> print b[0, 1]
2
>>> print b[0, 2]
3
>>> print b[2, 2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 2 is out of bounds for axis 0 with size 2
>>> print b[1, 2]
6
>>> c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> print c.shape
(3, 3)
>>> print c[c.shape[0]-1][c.shape[-1]-1]
9
>>> d = np.array([[[1,2,3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
>>> d.shape
(2, 2, 3)
>>> d[1][1][2]
12
>>> e = np.array([[[1,2,3]], [[10, 11, 12]]])
>>> e.shape
(2, 1, 3)
>>> f = np.array([[[1,2,3]], [[10, 11, 12, 13]]])
>>> f.shape
(2, 1)
>>> f[1][0]
[10, 11, 12, 13]
>>> f[1][0][3]
13
>>> f[0][0][3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> f[0][0][2]
3
>>> import numpy as np
>>> a = np.array([1, 2, 3])
>>> print type(a)
<type 'numpy.ndarray'>
>>> a.shape()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object is not callable
>>> print a.shape
(3,)
>>> print a[0]
1
>>> print a[1]
2
>>> print a[3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
>>> print a[2]
3
>>> print a[0][0]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: invalid index to scalar variable.
>>> a[0] = 5
>>> print a
[5 2 3]
>>> print a.shape[0]
3
>>> print a[a.shape[0]-1]
3
>>> b = np.array([[1, 2, 3], [4, 5, 6]])
>>> b
array([[1, 2, 3],
       [4, 5, 6]])
>>> b.shape
(2, 3)
>>> print b[0, 0]
1
>>> print b[0, 1]
2
>>> print b[0, 2]
3
>>> print b[2, 2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 2 is out of bounds for axis 0 with size 2
>>> print b[1, 2]
6
>>> c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> print c.shape
(3, 3)
>>> print c[c.shape[0]-1][c.shape[-1]-1]
9
>>> d = np.array([[[1,2,3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
>>> d.shape
(2, 2, 3)
>>> d[1][1][2]
12
>>> e = np.array([[[1,2,3]], [[10, 11, 12]]])
>>> e.shape
(2, 1, 3)
>>> f = np.array([[[1,2,3]], [[10, 11, 12, 13]]])
>>> f.shape
(2, 1)
>>> f[1][0]
[10, 11, 12, 13]
>>> f[1][0][3]
13
>>> f[0][0][3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> f[0][0][2]
3


>>> a = np.zeros((2, 2))
>>> a
array([[ 0.,  0.],
       [ 0.,  0.]])
>>> 1.
1.0
>>> type(a)
<type 'numpy.ndarray'>
>>> type(a[0][0])
<type 'numpy.float64'>
>>> a[0][0]
0.0
>>> b = np.ones((1, 2))
>>> b
array([[ 1.,  1.]])
>>> c = np.full((2, 2), 7)
/home/yong/workspace/nlp/nlp_env/local/lib/python2.7/site-packages/numpy/core/numeric.py:301: FutureWarning: in the future, full((2, 2), 7) will return an array of dtype('int64')
  format(shape, fill_value, array(fill_value).dtype), FutureWarning)
>>> c
array([[ 7.,  7.],
       [ 7.,  7.]])
>>> d = np.full((2, 2), 7.)
>>> d
array([[ 7.,  7.],
       [ 7.,  7.]])
>>> d[0][0]
7.0
>>> d[0][0] = 1
>>> d
array([[ 1.,  7.],
       [ 7.,  7.]])
>>> a[0][0]
0.0
>>> a[0][0] = 2
>>> a
array([[ 2.,  0.],
       [ 0.,  0.]])
>>> e = np.eye(2)
>>> d
array([[ 1.,  7.],
       [ 7.,  7.]])
>>> e
array([[ 1.,  0.],
       [ 0.,  1.]])
>>> f = np.random.random((2, 2))
>>> f
array([[ 0.12866224,  0.61469581],
       [ 0.36897623,  0.37287914]])

>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
>>> a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> b = a[:2, 1:3]
>>> b
array([[2, 3],
       [6, 7]])
>>> print a[0, 1]
2
>>> b[0,0] = 77
>>> print a[0, 1]
77
>>> a
array([[ 1, 77,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
>>> a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
>>> row_r1 = a[1, :]
>>> row_r1
array([5, 6, 7, 8])
>>> row_r2 = a[1:2, :]
>>> row_r2
array([[5, 6, 7, 8]])
>>> print row_r1, row_r1.shape
[5 6 7 8] (4,)
>>> print row_r2, row_r2.shape
[[5 6 7 8]] (1, 4)
>>> col_r1 = a[:, 1]
>>> row_r2 = a[:, 1:2]
>>> col_r2 = a[:, 1:2]
>>> print col_r1, col_r1.shape
[ 2  6 10] (3,)
>>> print col_r2, col_r2.shape
[[ 2]
 [ 6]
 [10]] (3, 1)
>>> 

>>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
>>> print a
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]
>>> b = np.array([0, 2, 0, 1])
>>> print b
[0 2 0 1]
>>> b
array([0, 2, 0, 1])
>>> print a[np.arange(4), b]
[ 1  6  7 11]
>>> print a[np.arange(3), b]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (4,) 
>>> a[np.arange(4), b] += 10
>>> print a
[[11  2  3]
 [ 4  5 16]
 [17  8  9]
 [10 21 12]]
>>> print a[np.arange(3), b]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (3,) (4,) 
>>> print a[np.arange(4), b]
[11 16 17 21]
>>> a[np.arange(4), b] += 10
>>> print a[np.arange(4), b]
[21 26 27 31]
>>> 

>>> a = np.array([[1, 2], [3, 4], [5, 6]])
>>> a > 2
array([[False, False],
       [ True,  True],
       [ True,  True]], dtype=bool)
>>> print a > 2
[[False False]
 [ True  True]
 [ True  True]]
>>> bol_idx = (a > 2)
>>> print bool_idx
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'bool_idx' is not defined
>>> bo0l_idx = (a > 2)
>>> bool_idx = (a > 2)
>>> print bool_idx
[[False False]
 [ True  True]
 [ True  True]]
>>> 

>>> x = np.array([1, 2], dtype=np.int64)
>>> print x.dtype
int64
>>> x = np.array([1, 2])
>>> print x.dtype
int64
>>> x = np.array([1.0, 2.0])
>>> print x.dtype
float64
>>> x = np.array([1.0, 2.0], dtype=np.int64)
>>> print x.dtype
int64
>>> print x
[1 2]
>>> 

>>> x = np.array([[1,2],[3,4]])
>>> y = np.array([[5,6],[7,8]])
>>> 
>>> v = np.array([9,10])
>>> w = np.array([11, 12])
>>> print v.dot(w)
219
>>> x.dot(y)
array([[19, 22],
       [43, 50]])
>>> np.dot(v, w)
219
>>> x
array([[1, 2],
       [3, 4]])
>>> y
array([[5, 6],
       [7, 8]])
>>> x.dot(y)
array([[19, 22],
       [43, 50]])
>>> print x.dot(v)
[29 67]
>>> x
array([[1, 2],
       [3, 4]])
>>> v
array([ 9, 10])
>>> x = np.array([[1, 2], [3, 4]])
>>> print np.sun(x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'sun'
>>> print np.sum(x)
10
>>> print np.sum(x, axis = 0)
[4 6]
>>> print np.sum(x, axis = 1)
[3 7]
>>> 

>>> x = np.array([[1, 2], [3, 4]])
>>> print np.sun(x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'module' object has no attribute 'sun'
>>> print np.sum(x)
10
>>> print np.sum(x, axis = 0)
[4 6]
>>> print np.sum(x, axis = 1)
[3 7]
>>> x = np.array([[1, 2], [3, 4]])
>>> print x
[[1 2]
 [3 4]]
>>> print x.T
[[1 3]
 [2 4]]
>>> v = np.array([1, 2, 3])
>>> print v
[1 2 3]
>>> print v.T
[1 2 3]
>>> w = np.array([1, 2])
>>> print w
[1 2]
>>> print w.T
[1 2]
>>> 

>>> x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
>>> v = np.array([1, 0, 1])
>>> y = np.empty_like(x)
>>> for i in range(4):
...     y[i, :] = x[i, :] + v
... 
>>> y
array([[ 2,  2,  4],
       [ 5,  5,  7],
       [ 8,  8, 10],
       [11, 11, 13]])
>>> y = np.empty_like(x)
>>> y
array([[0, 0, 0],
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]])
>>> y = np.zeros(x)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: only length-1 arrays can be converted to Python scalars
>>> y = np.zeros(x.shape)
>>> y
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
>>> 

>>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
>>> y = np.array([1, 0, 1)
  File "<stdin>", line 1
    y = np.array([1, 0, 1)
                         ^
SyntaxError: invalid syntax
>>> y = np.array([1, 0, 1])
>>> z = np.tile(y, (4, 1))
>>> z
array([[1, 0, 1],
       [1, 0, 1],
       [1, 0, 1],
       [1, 0, 1]])
>>> z = np.tile(y, (4, 0))
>>> z
array([], shape=(4, 0), dtype=int64)
>>> z = np.tile(y, (4, 3))
>>> z
array([[1, 0, 1, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 1, 0, 1, 1, 0, 1],
       [1, 0, 1, 1, 0, 1, 1, 0, 1]])
>>> z = np.tile(y, (4, 1))
>>> y = x + z
>>> y
array([[ 2,  2,  4],
       [ 5,  5,  7],
       [ 8,  8, 10],
       [11, 11, 13]])
>>> 

>>> array([[ 2,  2,  4],
...        [ 5,  5,  7],
...        [ 8,  8, 10],
...        [11, 11, 13]])

>>> v = np.array([1, 2, 3])
>>> w = np.array([4, 5])
>>> print np.reshape(v, (3, 1))
[[1]
 [2]
 [3]]
>>> print np.reshape(v, (2, 1))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/yong/workspace/nlp/nlp_env/local/lib/python2.7/site-packages/numpy/core/fromnumeric.py", line 224, in reshape
    return reshape(newshape, order=order)
ValueError: total size of new array must be unchanged
>>> print np.reshape(v, (3, 1)) * w
[[ 4  5]
 [ 8 10]
 [12 15]]
>>> np.array([1, 2]) * np.array([3, 4])
array([3, 8])
>>> np.array([1, 2]) * np.array([3])
array([3, 6])
>>> np.array([1, 2]).reshape((2, 1)) * np.array([3])
array([[3],
       [6]])
>>> v
array([1, 2, 3])
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> print x + v
[[2 4 6]
 [5 7 9]]
>>> print (x.T + w).T
[[ 5  6  7]
 [ 9 10 11]]
>>> v
array([1, 2, 3])
>>> x
array([[1, 2, 3],
       [4, 5, 6]])
>>> w
array([4, 5])
>>> x
array([[1, 2, 3],
       [4, 5, 6]])
>>> x*2
array([[ 2,  4,  6],
       [ 8, 10, 12]])
>>> print x*2
[[ 2  4  6]
 [ 8 10 12]]


