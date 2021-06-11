#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>       /* fabsf */
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG 0
using namespace std;
//Error check-----
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}
//Error check-----
//This is a very good idea to wrap your calls with that function.. Otherwise you will not be able to see what is the error.
//Moreover, you may also want to look at how to use cuda-memcheck and cuda-gdb for debugging.


__global__ void updater(double* rv, double* cv,int* adj, int* xadj,int temp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<temp){
    int st = xadj[i+1];
    int ed = xadj[i];
    double rsum = 0;
    for(int k = ed;k<st;k++){
      rsum += cv[adj[k]];
    }
    rv[i]= 1/rsum;
    }
}
__global__ void updaters(double* rv, double* cv,int* tadj, int* txadj,int temp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<temp){
    int kt = txadj[i+1];
    int kd = txadj[i];
    double csum = 0;
    for(int m = kd;m<kt;m++){
      csum += rv[tadj[m]];
    }
    cv[i]=1/csum;   
    }
}

__global__ void updatere(double* rv, double* cv,int* adj, int* xadj,double* max,int temp)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i< temp){
    int st = xadj[i+1];
    int ed = xadj[i];
   
    double total = 0;
    for(int k = ed;k<st;k++){
        total += cv[adj[k]]* rv[i];
    }
    total = fabs(1-total);
    if(*max < total){
      *max = total;
    }
    }
}


void wrapper(int* adj, int* xadj, int* tadj, int* txadj, double* rv, double* cv, int* nov, int* nnz, int siter){
  
  printf("Wrapper here! \n");
  
  //TO DO: DRIVER CODE
    int* adj_d, *xadj_d, *tadj_d, *txadj_d;
    gpuErrchk(cudaMalloc( (void**) &adj_d, (*nnz) * sizeof(int)));
    gpuErrchk(cudaMemcpy(adj_d, adj, (*nnz) * sizeof(int), cudaMemcpyHostToDevice ));

    gpuErrchk(cudaMalloc( (void**) &xadj_d, (*nov) * sizeof(int)));
    gpuErrchk(cudaMemcpy(xadj_d, xadj, (*nov) * sizeof(int), cudaMemcpyHostToDevice ));

    gpuErrchk(cudaMalloc( (void**) &tadj_d, (*nnz) * sizeof(int)));
    gpuErrchk(cudaMemcpy(tadj_d, tadj,(*nnz) * sizeof(int), cudaMemcpyHostToDevice ));

    gpuErrchk(cudaMalloc( (void**) &txadj_d, (*nov) * sizeof(int)));
    gpuErrchk(cudaMemcpy(txadj_d, txadj,(*nov) * sizeof(int), cudaMemcpyHostToDevice ));

   for(int i = 0;i<*nov;i++){
        rv[i]=1;
        cv[i]=1;
    }
    cudaEvent_t start, stop;
    double* rv_d;
    gpuErrchk(cudaMalloc((void **)&rv_d, (*nov) * sizeof(double)));
    gpuErrchk(cudaMemcpy(rv_d, rv, (*nov) * sizeof(double), cudaMemcpyHostToDevice));
    double* cv_d;
    gpuErrchk(cudaMalloc((void **)&cv_d, (*nov) * sizeof(double)));
    gpuErrchk(cudaMemcpy(cv_d, cv, (*nov) * sizeof(double), cudaMemcpyHostToDevice));

  

    double *max = new double(0);
    double *max_d;
    int temp_d = (*nov)-1;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    for(int x = 0; x < siter;x++){
      gpuErrchk(cudaMalloc( (void**) &max_d, sizeof(double)));
      updater<<<(*nov + 1024 - 1)/1024,1024>>>(rv_d,cv_d,adj_d,xadj_d,temp_d);
      gpuErrchk(cudaPeekAtLastError());

      updaters<<<(*nov + 1024 - 1)/1024,1024>>>(rv_d,cv_d,tadj_d,txadj_d,temp_d);
      gpuErrchk(cudaPeekAtLastError());

      updatere<<<(*nov + 1024 - 1)/1024,1024>>>(rv_d,cv_d,adj_d,xadj_d,max_d,temp_d);

      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaMemcpy(max, max_d, sizeof(double), cudaMemcpyDeviceToHost));

      cout<<"iter "<< x <<" - error " <<*max<<endl;
      *max = 0;
    } 

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    gpuErrchk(cudaFree(max_d));
    gpuErrchk(cudaFree(xadj_d));
    gpuErrchk(cudaFree(adj_d));
    gpuErrchk(cudaFree(txadj_d));
    gpuErrchk(cudaFree(tadj_d));
    gpuErrchk(cudaFree(rv_d));
    gpuErrchk(cudaFree(cv_d));
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU scale took: %f s\n", elapsedTime/1000);  
}

