# cuda组织工程
1.写一个warp函数将kernel包装起来,对外只暴露这个warp函数,与cuda相关的内存分配，运行配置等在warp函数里执行。
对外只暴露一个warp函数:
```
void run_kernel(float* h_in, float* h_out, int n);
```
2. makefile中nvcc就可以只编译.cu文件。其他cpp文件可以交给gcc编译。
3.注意链接时需要链接cuda相关的库

