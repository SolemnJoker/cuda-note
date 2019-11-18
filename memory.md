# 一.gpu内存类型(local memory)
## 1.本地内存
每个线程有自己的本地内存
## 2.共享内存(shared memory)
每个线程块包含有共享内存，可以被线程块中所有线程共享,其生命周期与线程块一致
### 共享内存访问冲突
共享内存分成相同大小的内存块，实现告诉缓存<\br>
**bank:** 
GPU 共享内存是基于存储体切换的架构（bank-switched-architecture）。在 Femi，Kepler，Maxwell 架构的设备上有 32 个存储体（也就是常说的共享内存分成 32 个bank），而在 G200 与 G80 的硬件上只有 16 个存储体。</br>
每个存储体（bank）每个周期只能指向一次操作（一个 32bit 的整数或者一个单精度的浮点型数据），一次读或者一次写，也就是说每个存储体（bank）的带宽为 每周期 32bit。

#### a.同常量内存一样，当一个 warp 中的所有线程访问同一地址的共享内存时，会触发一个广播（broadcast）机制到 warp 中所有线程，这是最高效的。
#### b.如果同一个 warp 中的线程访问同一个 bank 中的不同地址时将发生 bank conflict。
#### c.每个 bank 除了能广播（broadcast）还可以多播（mutilcast）（计算能力 >= 2.0），也就是说，如果一个 warp 中的多个线程访问同一个 bank 的同一个地址时（其他线程也没有访问同一个bank 的不同地址）不会发生 bank conflict。
#### d. 即使同一个 warp 中的线程 随机的访问不同的 bank，只要没有访问同一个 bank 的不同地址就不会发生 bank conflict。

https://blog.csdn.net/endlch/article/details/47043069
## 3.全局内存
所有线程都可以访问
全局内存访问是对齐的,
## 4. 只读内存
常量内存和纹理内存
## 5. L1 cache ,L2 cache

## 6.寄存器
在核函数中不加修饰的声明一个变量，此变量就储存在寄存及中，寄存器对每个线程都是私有的，寄存器通常保存陪频繁使用的私有变量。</br>
寄存器是sm的稀缺资源一个线程如果使用更少的寄存器，那么就会有更多的常驻线程块.sm上并发的线程块就越多。

# 内存优化策略
## 1.减少内存访问
pcie 很慢,减小cpu->GPU之间的内存传输，还可以使用锁页内存（cudaMallocHost替代malloc),
## 2.增加共享内存使用
共享内存比全局内存块一个数量级,
## 3.优化内存使用模式
### 全局内存对齐，合并访问
条件:</br>
a.数据类型是4,8,16字节</br>
b.数据连续</br>
方法:</br>
如下访问是不对齐的，因为数据类型vec3d是12bytes,不是4,8,16 bytes
```
    struct vec3d { float x, y, z; }; 
    ...
     
    __global__ void func(struct vec3d* data, float* output)
    {
     
    output[tid] = data[tid].x * data[tid].x + data[tid].y * data[tid].y + data[tid].z * data[tid].z;
     
    }
```
要解决这个问题，可以使用 __align(n)__，例如：
```
struct __align__(16) vec3d { float x, y, z; };
```
这会让 compiler 在 vec3d 后面加上一个空的 4 bytes，以补齐 16 bytes。

另一个方法，是把数据结构转换成三个连续float的数组,例如：
```
__global__ void func(float* x, float* y, float* z, float* output)
{
output[tid] = x[tid] * x[tid] + y[tid] * y[tid] + z[tid] * z[tid];
}
```
### 访问共享内存，避免bank冲突
