#include <math.h>
#include <iostream>
#include "warper.h"


void init_host_data(float* h_in,int n){
    for(int i = 0; i< n;i++){
        h_in[2*i]  = (float)i /((n - 1) * M_PI * 100);
        h_in[2*i+1] = sin(h_in[2*i+1]);
    }
}

int main(){
    float *h_out = NULL;
    float *h_in = NULL;
    size_t in_size  = N*2*sizeof(float);
    size_t out_size = N*sizeof(float);
    h_in = (float*)malloc(in_size);
    h_out = (float*)malloc(out_size);
    init_host_data(h_in, N);
    run_kernel(h_in,h_out,N);
    for(int i = 0;i < N;i++){
        std::cout<<i<<":<"<<h_in[2*i]<<","<<h_in[2*i+1]<<">, dist:"<<h_out[i]<<std::endl;
    }
    free(h_in);
    free(h_out);
    return 0;
}
