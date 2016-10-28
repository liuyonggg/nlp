>>> img[400-1][247]
array([86, 71, 50], dtype=uint8)
>>> img_tinted[400-1][247]
array([ 86.  ,  67.45,  45.  ])
>>> img_tined_resize = imresize(img_tinted, (300, 300))
>>> imsave('assets/cat_tinted_resized.jpg', img_tined_resize)
>>> img_tinted = img * [1, 0.1, 0.1]
>>> img_tined_resize = imresize(img_tinted, (300, 300))
>>> imsave('assets/cat_tinted_resized.jpg', img_tined_resize)
 from scipy.misc import imread, imsave, imresize

>>> 
>>> img = imread('assets/cat.jpg')

>>> import numpy as np
>>> from scipy.spatial.distance import pdist, squareform
>>> x = np.array([[0, 1], [1, 0], [2, 0]])
>>> print x
[[0 1]
 [1 0]
 [2 0]]
>>> d = squareform(pdist(x, "euclidean"))
>>> d
array([[ 0.        ,  1.41421356,  2.23606798],
       [ 1.41421356,  0.        ,  1.        ],
       [ 2.23606798,  1.        ,  0.        ]])
>>> print d
[[ 0.          1.41421356  2.23606798]
 [ 1.41421356  0.          1.        ]
 [ 2.23606798  1.          0.        ]]
>>> 

