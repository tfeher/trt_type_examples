# TensorRT examples to test type conversions

These examples are based on the TensorRT samples published at
 https://github.com/NVIDIA/TensorRT.

## How to compile

It is recommended to compile the examples in NVIDIA TensorRT docker container.

```
docker run  --rm -it -v $PWD:/data -w /data nvcr.io/nvidia/tensorrt:20.03-py3
```


## 1. Explicit type conversions
Let input be a float tensor. main1a.cpp defines the following network:

- x = shape(input)
- y = x + x

The shape op changes the input type to INT32, the elementwise addition keeps the
same shape.

```
nvcc main1a.cpp logger.cpp -lnvinfer
```

## 2. No implicit type conversion from INT32
Let input be a float tensor. main1b.cpp defines the following network:

- x = shape(input)
- y = - x

The shape op changes the input type to INT32, but elementwise - operation does not support INT32 type. The engine construction fails:

```
nvcc main1b.cpp logger.cpp -lnvinfer

./a.out
[07/01/2020-11:37:04] [I] Building and running inference engine for shape example
[07/01/2020-11:37:05] [E] [TRT] B: operation NEG not allowed on type Int32
[07/01/2020-11:37:05] [E] [TRT] B: operation NEG not allowed on type Int32
[07/01/2020-11:37:05] [E] [TRT] Layer B failed validation
[07/01/2020-11:37:05] [E] [TRT] Network validation failed.
&&&& FAILED Type1a # ./a.out
```

## 3. Implicit type conversion from INT8 works
Let input be an INT8 tensor. main1c.cpp defines the following network:

- x = RELU(input)
- y = - x

While the RELU op supports INT8, the elementwise op does not. The network is
still built and executed successfully due to automatic type conversion.
Note that the output type is FP32.

```
nvcc main1c.cpp logger.cpp -lnvinfer
```

## 4. Int8 input and output
Network with int8 input and output: `main1d_int8act.cpp` defines the following.
```
nvcc main1d_int8act.cpp logger.cpp -lnvinfer

./a.out

&&&& RUNNING Type1a # ./a.out
[01/25/2021-18:32:58] [I] Building and running inference engine for shape example
[01/25/2021-18:32:59] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
[01/25/2021-18:32:59] [W] [TRT] Tensor DataType is determined at build time for tensors not marked as input or output.
[01/25/2021-18:32:59] [I] Output type is INT8
[01/25/2021-18:32:59] [W] [TRT] Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32.
[01/25/2021-18:33:00] [W] [TRT] No implementation obeys reformatting-free rules, at least 2 reformatting nodes are needed, now picking the fastest path instead.
[01/25/2021-18:33:00] [I] [TRT] Detected 1 inputs and 1 output network tensors.
[01/25/2021-18:33:00] [I] Engine constructed successfully
[01/25/2021-18:33:00] [W] [TRT] Current optimization profile is: 0. Please ensure there are no enqueued operations pending in this context prior to switching profiles
Setting input value 0: -1
Setting input value 1: 0
Setting input value 2: 1
Setting input value 3: 2

Output:

&&&& PASSED Type1a # ./a.out

```
