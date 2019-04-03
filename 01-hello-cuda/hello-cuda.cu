#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

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

void init_host_data(float2* h_in,int n){
    for(int i = 0; i< n;i++){
        h_in[i].x = (float)i /((n - 1) * M_PI * 100);
        h_in[i].y = sin(h_in[i].x);
    }
}

int main(){
    float *d_out = NULL;
    float2 *d_in  = NULL;
    float2 *h_in = NULL;
    float *h_out = NULL;
    int N           = 4096;
    int TPB         = 32;
    size_t in_size  = N*2*sizeof(float);
    size_t out_size = N*sizeof(float);
    h_in = (float2*)malloc(in_size);
    h_out = (float*)malloc(out_size);


    //设备端分配内存
    cudaMalloc((void**)&d_in, in_size);
    cudaMalloc((void**)&d_out, out_size);

    init_host_data(h_in, N);
    //拷贝host数据到device
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    distance_kernel<<<(N + TPB -1)/TPB,TPB>>>(d_in, d_out, N);
    //拷贝device端计算结果到host
    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    
    for(int i = 0;i < N;i++){
        std::cout<<i<<":<"<<h_in[i].x<<","<<h_in[1].y<<">, dist:"<<h_out[i]<<std::endl;
    }
    free(h_in);
    free(h_out);
    return 0;
}
