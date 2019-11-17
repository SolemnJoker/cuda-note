#include "warper.h"
__device__ float distance(float2 x1, float2 x2){
    return sqrt(pow(x1.x - x2.x,2) + pow(x1.y - x2.y,2));
}
__global__ void distance_kernel(float2 *data_in, float *data_out, int n){
    const int  i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        float2 ref;
        ref.x = 0.0;
        ref.y = 0.0;
        data_out[i] = distance(data_in[i], ref);
    }
}

void run_kernel(float* h_in, float* h_out, int n){
    float2 *d_in  = NULL;
    float *d_out = NULL;

    size_t in_size  = n*2*sizeof(float);
    size_t out_size = n*sizeof(float);
 
    //设备端分配内存
    cudaMalloc((void**)&d_in, in_size);
    cudaMalloc((void**)&d_out, out_size);

    //拷贝host数据到device
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

        
    distance_kernel<<<(N + TPB -1)/TPB,TPB>>>(d_in, d_out, n);
    //拷贝device端计算结果到host
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

