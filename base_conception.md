# gpu硬件结构
GPU的硬件结构，也不是具体的硬件结构，就是与CUDA相关的几个概念：thread，block，grid，warp，sp，sm。
**sp**: 最基本的处理单元，streaming processor  最后具体的指令和任务都是在sp上处理的。GPU进行并行计算，也就是很多个sp同时做处理
**sm**:多个sp加上其他的一些资源组成一个sm,  streaming multiprocessor. 其他资源也就是存储资源，共享内存，寄储器等。
**warp**:GPU执行程序时的调度单位，目前cuda的warp的大小为32，同在一个warp的线程，以**不同数据**资源**执行相同的指令**。
**grid、block、thread**：在利用cuda进行编程时，一个grid分为多个block，而一个block分为多个thread.其中任务划分到是否影响最后的执行效果。划分的依据是任务特性和

GPU本身的硬件特性
# cuda并行模式
## 1.cuda使用单指令多线程（SIMT）的并行模式
cuda gpu包含了大量的基础计算单元(core),每一个core包含了一个**逻辑计算单元(ALU)**和一个**浮点计算单元(FPU)**。
多个核集成一个**多流处理器*(SM)**

## 2.grid - block并行模型
cuda调用dev运行核函数时,需要配置grid和block两层并行运算,一个grid分为多个block，而一个block分为多个thread.
![cuda双层并行模型](https://www.github.com/SolemnJoker/image_for_storywriter/raw/master/小书匠/1554908289231.png)
之所以要分两层并行模型,涉及到cuda的一个关键特性：<br>
线程按照粗粒度的线程块和细粒度的线程两个层次进行组织、在细粒度并行的层次通过共享存储器和栅栏同步实现通信。<br>
同一线程块中的众多线程拥有相同的指令地址，不仅能够并行执行，而且能够通过共享存储器（Shared memory）和栅栏（barrier）实现块内通信。这样，同一网格内的不同块之间存在不需要通信的粗粒度并行，而一个块内的线程之间又形成了允许通信的细粒度并行

# 一些常用的优化方法
## 线程束->线程块
对于block和thread的分配问题，有这么一个技巧，每个block里面的thread个数最好是32的倍数，因为，这样可以让线程束一起执行，计算效率更高，促进memory coalescing

##  合理分配线程块 
一个block只会由一个sm调度，一个sm同一时间只会执行一个block里的warp，当该block里warp执行完才会执行其他block里的warp。进行划分时，最好保证每个block里的warp比较合理，那样可以一个sm可以交替执行里面的warp，从而提高效率，此外，在分配block时，要根据GPU的sm个数，分配出合理的block数，让GPU的sm都利用起来，提利用率。分配时，也要考虑到同一个线程block的资源问题，不要出现对应的资源不够。

一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。Nvidia把32个threads组成一个warp，warp是调度和运行的基本单元。warp中所有threads并行的执行相同的指令。一个warp需要占用一个SM运行，多个warps需要轮流进入SM。由SM的硬件warp scheduler负责调度。目前每个warp包含32个threads（Nvidia保留修改数量的权利）。所以，一个GPU上resident thread最多只有 SM*warp个。

## 共享内存内存
共享内存的使用量也是影响occupancy的一个重要因子，一块大核拥有一块共享内存。shared添加到变量声明中，这将使这个变量驻留在共享内存中。在声明共享内存变量后，线程块中的每个线程都共享这块内存，使得一个线程块中的多个线程能够在计算上进行通信和协作。

## 纹理缓存
纹理缓存是只读的内存，专门为内存访问存在大量空间局部性的设计，核函数需要特殊的函数告诉GPU读取纹理内存而不是全局内存。使用纹理内存，如果同一个线程束内的thread的访问地址很近的话，那么性能更高。

# cuda学习资源
## cuda 的官方sample
在cuda安装目录/usr/local/cuda/samples

