#include <math.h>
#include <iostream>
#include "warper.h"


void init_host_data(float* h_in,int w, int h){
    for(int x = 0; x < w;x++){
        for(int y = 0; y < h;y++){
            int i = w*y + x;
            h_in[2*i]  = (float)x /(w*y - 1);
            h_in[2*i+1] = (float)y /(w*y - 1);
        }
    }
}

int main(){
    float *h_out = NULL;
    float *h_in = NULL;
    int N = W*H;
    size_t in_size  = N*2*sizeof(float);
    size_t out_size = N*sizeof(float);
    h_in = (float*)malloc(in_size);
    h_out = (float*)malloc(out_size);
    init_host_data(h_in, W, H);
    run_kernel(h_in,h_out,W,H);
    // for(int i = 0;i < N;i++ ){
    //     std::cout<<i<<":<"<<h_in[2*i]<<","<<h_in[2*i+1]<<">, dist:"<<h_out[i]<<std::endl;
    // }
    free(h_in);
    free(h_out);
    return 0;
}
