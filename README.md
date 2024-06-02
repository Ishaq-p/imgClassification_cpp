# imgClassification_cpp

const: Ensures the input matrix is not modified inside the function.

https://github.com/PulkitThakar/CNN-from-scratch-using-Numpy/blob/master/CNN.py#L277

static_cast<float>(rand() %10)/10

When transferring a trained model's weights from PyTorch (Python) to a C++ implementation, several potential issues could cause a discrepancy in accuracy. Here are some common issues and suggestions to troubleshoot:

1. **Data Preprocessing**:
    - Ensure that the preprocessing of the MNIST images in your C++ code matches exactly with what you did in Python. This includes normalization, resizing, and any other transformations.
    - Check if the pixel value scaling is consistent (e.g., values between 0-1 or 0-255).

2. **Weight Initialization and Storage**:
    - Verify that the weights and biases are correctly imported into your C++ code. Double-check the dimensionality and ensure that they are loaded into the correct layers.
    - Ensure the weights are saved and loaded in the same format. If there is any conversion (e.g., endianness), it should be correctly handled.

3. **Model Architecture**:
    - Make sure the model architecture in C++ exactly mirrors the architecture defined in PyTorch, including layer types, order, and activation functions.
    - Pay attention to any potential differences in the implementation of layers and activation functions between PyTorch and your C++ code.

4. **Numerical Precision**:
    - Check if there are any numerical precision issues. PyTorch might use 32-bit floating-point numbers by default, and any reduction in precision in your C++ code can affect the accuracy.
    - Ensure consistent precision in all operations (e.g., weights, activations, intermediate calculations).

5. **Activation Functions and Operations**:
    - Ensure the implementation of activation functions and other operations (like convolutions, pooling, etc.) in your C++ code matches PyTorch’s behavior.
    - Pay attention to details like padding, stride, and dilation in convolutional layers.

6. **Loss Function and Evaluation Metrics**:
    - Ensure the loss function and evaluation metrics are correctly implemented and match those used in PyTorch.
    - Confirm that the accuracy calculation in your C++ code is correct.

7. **Debugging Tips**:
    - Print intermediate outputs (e.g., after each layer) in both PyTorch and C++ to compare and identify where the outputs start diverging.
    - Use a small subset of the MNIST dataset to manually verify the outputs layer by layer.


- **defining functions inside the class itself**:
    - Access Control:
    Use member functions if the function needs access to private or protected members of the class.
    Use non-member functions if the function only requires public access or should operate independently of the class's internal - state.
    - Code Organization:
    For utility functions that operate on multiple types or don’t naturally belong to any single class, consider non-member functions.
    - Efficiency:
    Modern C++ compilers are quite good at optimizing function calls, so the efficiency difference between member and non-member functions is often negligible. Focus more on design and readability.





### 1. Inlined Functions
- **Inlining**: If a function is marked as `inline`, or if the compiler decides to inline a function, it will be expanded directly into the caller's code. This can cause the function not to appear separately in the `gprof` output because it doesn’t exist as a standalone entity in the binary.
- **Optimization Level**: Higher optimization levels (`-O2`, `-O3`, etc.) often result in more aggressive inlining.

### 2. Optimization
- **Function Optimization**: Compilers may optimize away functions that are too small or simple, especially if they are not explicitly marked to prevent such optimizations.
- **Whole Program Optimization**: Some functions might be eliminated entirely if they are deemed unnecessary by the compiler during optimizations.

### 3. Profiling Information
- **Profiling Overhead**: If a function is very short and executes quickly, the profiling overhead might be significant enough that the profiler doesn’t accurately capture it.
- **Instrumentation**: Some profiling methods or tools might not instrument very small functions or might miss them if they are rarely called or execute too quickly.

### 5. Linking
- **Static vs. Dynamic Linking**: Ensure that the functions you want to profile are included in the profiling process. If you are dynamically linking libraries, ensure those libraries are also compiled with profiling enabled.

#### Code Example

```cpp
inline void inlinedFunction() {
    std::cout << "This is an inlined function." << std::endl;
}
void optimizedFunction() {
    std::cout << "This function might be optimized away." << std::endl;
}
```

In this case:
- `inlinedFunction()` may not appear in the `gprof` output because it is inlined.
- `optimizedFunction()` might be optimized away or merged into the main function if the compiler deems it trivial.

### How to Ensure Visibility

1. **Disable Inlining**:
   - Use `-fno-inline` or `-fno-inline-small-functions` compiler flags to prevent inlining.

2. **Lower Optimization Levels**:
   - Use `-O0` or `-O1` to reduce aggressive optimizations.

4. **Explicitly Prevent Optimization**:
   - Use `volatile` or other pragmas to prevent the compiler from optimizing away specific functions.
