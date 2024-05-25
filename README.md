# imgClassification_cpp

const: Ensures the input matrix is not modified inside the function.
Matrix&: Passes the input by reference, avoiding copying.
int poolSize: The second parameter, an integer specifying the pool size. This is passed by value because it's a simple data type and doesn't incur significant overhead.



https://github.com/PulkitThakar/CNN-from-scratch-using-Numpy/blob/master/CNN.py#L277