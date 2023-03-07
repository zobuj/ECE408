
#include <wb.h>

//Defining the Tile width
#define TILE_WIDTH 4

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  int outputWidth = numCColumns;
  int innerWidth = numBRows;
  int outerWidth = numBColumns;
  int currentSize = numAColumns;
  if((row < numCRows) && (col < numCColumns)){
    float Pvalue = 0;
    for(int k =0;k<currentSize;++k){
      Pvalue += A[row*innerWidth+k] * B[k*outerWidth+col];
    }
    C[row*outputWidth+col]=Pvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //Setting element number for each matrix
  int numElements_A = numAColumns*numARows;
  int numElements_B = numBColumns*numBRows;
  int numElements_C = numCColumns*numCRows;

  //Setting Memory Size for Each Matrix
  int mem_numElements_A = numElements_A*sizeof(float);
  int mem_numElements_B = numElements_B*sizeof(float);
  int mem_numElements_C = numElements_C*sizeof(float);

  //Output Matrix Width
  int Width;
  if(numCColumns>numCRows){
    Width = numCColumns;
  }else{
    Width = numCRows;

  }
  //@@ Allocate the hostC matrix
  hostC = (float*) malloc(mem_numElements_C);

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, mem_numElements_A);
  cudaMalloc((void **) &deviceB, mem_numElements_B);
  cudaMalloc((void **) &deviceC, mem_numElements_C);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, mem_numElements_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, mem_numElements_B, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, mem_numElements_C, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*Width)/TILE_WIDTH),ceil((1.0*Width)/TILE_WIDTH),1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,numBColumns,numCRows,numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostC,deviceC,mem_numElements_C,cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");

  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}