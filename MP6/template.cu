// Histogram Equalization

#include <wb.h>

#define TILE_WIDTH 4
#define HISTOGRAM_LENGTH 256
__global__ void castImage(int width, int height, int channel, float * inputImage, unsigned char * ucharImage){
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  if(row < height && col < width){
    ucharImage[(row*width+col)*channel+threadIdx.z] = (unsigned char)(255*inputImage[(row*width+col)*channel+threadIdx.z]);
  }
}

__global__ void RGBToGrayScale(int width, int height, int channel,unsigned char *grayImage,unsigned char *rgbImage){
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  if(row < height && col < width){
    int grayOffset = row*width+col;
    int rgbOffset = grayOffset*channel;

    unsigned char r = rgbImage[rgbOffset];
    unsigned char g = rgbImage[rgbOffset+1];
    unsigned char b = rgbImage[rgbOffset+2];
    grayImage[grayOffset]=0.21f*r+0.71f*g+0.07f*b;
  }
}
__global__ void histogram(int width, int height,unsigned char *grayImage,unsigned int *histogram){
  __shared__ unsigned int histo_private[256];

  if(threadIdx.x<256){
    histo_private[threadIdx.x]=0;
  }
  __syncthreads();
  
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;

  while(row<height && col<height){
    atomicAdd(&(histo_private[grayImage[row*width+col]]),1);
  }
  __syncthreads();

}
//@@ insert code here

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  /*
  Image Declarations
  */

  //Original Image(Device) 
  float *deviceInputImageData;

  //Unsigned Char Image(Device) 
  unsigned char *deviceUCharOutputImage;

  //Unsigned Gray Image(Device) 
  unsigned char *deviceGrayUCharImage;

  //Unsigned Int Histogram(Device)
  unsigned int *deviceHistogram;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  /*
  cudaMalloc Calls
  */

  //Unsigned Char Image(Device)
  cudaMalloc((void**)&deviceUCharOutputImage,imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  
  //Original Image(Device)
  cudaMalloc((void**)&deviceInputImageData,imageWidth*imageHeight*imageChannels*sizeof(float));
  
  //Unsigned Gray Image(Device)
  cudaMalloc((void**)&deviceGrayUCharImage,imageWidth*imageHeight*sizeof(unsigned char));
  
  //Unsigned Int Histogram(Device)
  cudaMalloc((void**)&deviceHistogram,HISTOGRAM_LENGTH*sizeof(unsigned int));

  /*
  cudaMemcpy Calls
  */
  //Float to Unsigned
  cudaMemcpy(deviceInputImageData,hostInputImageData,imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);

  /*
  Threads and Blocks
  */
  //Float to Unsigned
  dim3 dimGridFloatToUnsigned(ceil((1.0*imageWidth)/TILE_WIDTH),ceil((1.0*imageHeight)/TILE_WIDTH),1);
  dim3 dimBlockFloatToUnsigned(TILE_WIDTH,TILE_WIDTH,imageChannels);

  //RBG to Gray
  dim3 dimGridRGBtoGray(ceil((1.0*imageWidth)/TILE_WIDTH),ceil((1.0*imageHeight)/TILE_WIDTH),1);
  dim3 dimBlockRGBtoGray(TILE_WIDTH,TILE_WIDTH,imageChannels);

  //Histogram of Gray Image
  dim3 dimGridHistGray(ceil((1.0*imageWidth)/TILE_WIDTH),ceil((1.0*imageHeight)/TILE_WIDTH),1);
  dim3 dimBlockHistGray(TILE_WIDTH,TILE_WIDTH,1);
  /*
  Kernel Invocation
  */
  //Float to Unsigned
  castImage<<<dimGridFloatToUnsigned,dimBlockFloatToUnsigned>>>(imageWidth,imageHeight,imageChannels, deviceInputImageData, deviceUCharOutputImage);
  RGBToGrayScale<<<dimGridRGBtoGray,dimBlockRGBtoGray>>>(imageWidth,imageHeight,imageChannels,deviceGrayUCharImage, deviceUCharOutputImage);
  


  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaMemcpy(hostOutputImageData,deviceGrayUCharImage,imageWidth*imageHeight*sizeof(unsigned char),cudaMemcpyDeviceToHost);

  cudaFree(deviceUCharOutputImage);
  cudaFree(deviceInputImageData);
  cudaFree(deviceGrayUCharImage);
  return 0;
}