# TensorRT examples to test type conversions

## How to compile

It is recommended to compile the examples in NVIDIA TensorRT docker container.

```
docker run  --rm -it -v $PWD:/data -w /data nvcr.io/nvidia/tensorrt:20.03-py3
```


## 1. Explicit type conversions
Let input be a float tensor. main1a.cpp defines the following network:

x = shape(input)
y = x + x

The shape op changes the input type to INT32, the elementwise addition keeps th
same time

```
nvcc main1a.cpp logger.cpp -lnvinfer
```

## 2. Implicit type conversion from INT32 does not work
Let input be a float tensor. main1b.cpp defines the following network:

x = shape(input)
y = - x

The shape op changes the input type to INT32, but elementwise - operation does not support INT32 type. The engine constructino fails:

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

x = RELU(input)
y = - x

While the RELU op supports INT8, the elementwise op does not. The network is
still build and executed successfully due to automatic type conversion.
Note that the output type is FP32.

```
nvcc main1c.cpp logger.cpp -lnvinfer
```
