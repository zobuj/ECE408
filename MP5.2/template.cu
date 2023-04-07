// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void scan(float *input, float *output, int len, float *S) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len){
    T[threadIdx.x] = input[i];
  }
  if(i+blockDim.x < len){
    T[threadIdx.x+blockDim.x] = input[i+blockDim.x];
  }

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE){
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if(i < len){
    output[i] = T[threadIdx.x];
  }
  if(i + blockDim.x < len){
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }

  __syncthreads();
  if(threadIdx.x==blockDim.x-1){
    S[blockIdx.x] = T[2*BLOCK_SIZE-1];
  }

}

__global__ void scan2(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len){
    T[threadIdx.x] = input[i];
  }
  if(i+blockDim.x < len){
    T[threadIdx.x+blockDim.x] = input[i+blockDim.x];
  }

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }
  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE){
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  if(i < len){
    output[i] = T[threadIdx.x];
  }
  if(i + blockDim.x < len){
    output[i + blockDim.x] = T[threadIdx.x + blockDim.x];
  }

}
__global__ void thirdKernel(float* input,float* output, int len){
  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  if(blockIdx.x > 0){
      output[i] += input[blockIdx.x-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *S_arr;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //Allocating memory for S array
  cudaMalloc((void **)&S_arr,ceil(numElements/(2 * BLOCK_SIZE * 1.0)) * sizeof(float));
  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock1(BLOCK_SIZE,1,1);
  dim3 dimGrid1(ceil(numElements/(2 * BLOCK_SIZE * 1.0)),1,1);
  dim3 dimBlock2(BLOCK_SIZE,1,1);
  dim3 dimGrid2(1,1,1);
  dim3 dimBlock3(2 * BLOCK_SIZE,1,1);
  dim3 dimGrid3(ceil(numElements/(2 * BLOCK_SIZE * 1.0)),1,1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid1,dimBlock1>>>(deviceInput,deviceOutput,numElements, S_arr);
  scan2<<<dimGrid2,dimBlock2>>>(S_arr,S_arr,ceil(numElements/(2 * BLOCK_SIZE * 1.0)));
  thirdKernel<<<dimGrid3,dimBlock3>>>(S_arr,deviceOutput,numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  wbTime_start(GPU, "Freeing GPU Memory");

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(S_arr);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}