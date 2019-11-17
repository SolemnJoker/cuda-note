# 二维网络
## 配置二维网络
二维网络在<<<grid_dim,block_dim>>>中配置<br>
grid_dim,和block_dim都是dim3类型，可以表示x,y,z三个维度，<br>
可以这样进行配置
```
dim3 grid_dim(512,512);
dim3 block_dim(16,16);
kernel<<<grid_dim,block_dim>>>(args...);
```
这样表示执行一个512×512大小的网络，每个block中有16×16个线程。

## 网络大小限制
1.block中线程数量最好是32的整数倍，因为cuda执行并发的时候是以线程束（warp）为单位的，一个warp中有32个线程
2.一个block中线程数量有限,grid大小也有限(有最大限制，不同档次架构的显卡不一样)，可以使用cudaGetDeviceProperties(cudaDeviceProp\*, int)函数查询block和grid的大小限制。
