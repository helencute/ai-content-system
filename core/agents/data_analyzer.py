# Use newer numpy APIs
import numpy as np

# Instead of np.matrix (deprecated in 2.x)
matrix = np.array([[1, 2], [3, 4]])

# Instead of np.int (deprecated)
integer_array = np.array([1, 2, 3], dtype=np.int64)