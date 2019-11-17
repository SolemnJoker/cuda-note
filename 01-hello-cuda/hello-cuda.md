#  cuda基本流程
## host
```
cudaMalloc(void **devPtr,size_t size); //分配设备端内存
cudaMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind); //host数据到device
kernel<<<num_of_block,thread_per_block>>>(...) //执行kernel
cudaMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind); //device数据拷贝到host
cudaFree(void* mem);
```
## device
kernel没有返回值
```
__global__ void kernel(args...);

```
## Makefile
指定编译器为nvcc
加编译参数-Xcompiler可以加gcc参数
```
NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -Xcompiler -Wall 

hello-cuda:hello-cuda.cu
	${NVCC} ${NVCC_FLAGS} $^ -o $@
```
### 获取线程索引:
gridDim:Dim3,网络中线程线程块的数量<br>
blockDim:Dim3,每个线程块中线程的数量<br>
blockIdx: uint3,当前线程块在网络中的索引<br>
threadIdx:uint3,当前线程在线程块中的索引<br>

#### cuda定义了一些内置变量可以索引线程:<br>
threadIdx<br>
blockIdx<br>


```
//一维数据索引
int idx = blockIdx.x * blockDim.x + threadIdx.x;
//二维数据索引
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y*w+x;//w是从外部(host端传过来或者事先赋值的)赋值的二维数据的行宽
//或者
int idx  = blockDim.X*(blockIdx.x + blockIdx.y*gridDim.x)+threadIdx.x
```

# 补充
## kernel执行配置
调用核函数时<<<num_block,thread_per_block>>>设置:<br>
假设要处理的数据共需要N个线程,每个block指定了线程数量TPB那么block的数量应该这样计算:<br>
$\frac{N+TPB-1}{TPB}$ 而不是 $\frac{N}{TPB}$ 
这是由于$\frac{N}{TPB}$时整数除法当N不是TPB的倍数时会导致会有部分数据没有线程去计算
不过这样有一个问题就是线程数会超出数据的数量,因此需要在核函数中判断当前线程是否超出了范围


