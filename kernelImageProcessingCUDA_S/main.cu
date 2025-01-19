#include <iostream>
#include <opencv.hpp>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdlib.h>


/*<--------------------------------------DEFS-------------------------------------->*/
#define CUDA_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#define MASK_WIDTH  3
#define MASK_HEIGHT 3
#define MASK_RADIUS_X (MASK_WIDTH / 2)
#define MASK_RADIUS_Y (MASK_HEIGHT / 2)
#define TILE_WIDTH 8
#define w_x (TILE_WIDTH + MASK_WIDTH - 1)
#define w_y (TILE_WIDTH + MASK_HEIGHT - 1)

static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) <<
            "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

/*<--------------------------------------CONSTANT MEM-------------------------------------->*/
__constant__ float KERNEL[MASK_WIDTH * MASK_HEIGHT];
__constant__ int CHANNELS;
__constant__ int IMG_WIDTH;
__constant__ int IMG_HEIGHT;

/*<--------------------------------------HOST FUNCTIONS-------------------------------------->*/

__host__ void buildImage(unsigned char *finalDatasArr, int width, int height, int channels, cv::Mat *finalImage) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            finalImage->at<cv::Vec3b>(y, x) = cv::Vec3b(
                finalDatasArr[(y * width + x) * channels + 2],
                finalDatasArr[(y * width + x) * channels + 1],
                finalDatasArr[(y * width + x) * channels]
            ); /*BGR format*/
        }
    }
}

__host__ float *getKernel(int choosenKernel) {
    switch (choosenKernel) {
        case 0: /* Identity */
            return new float[9]{0, 0, 0, 0, 1, 0, 0, 0, 0};
        case 1: /* Blur */
            return new float[9]{0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
        case 2: /* Emboss */
            return new float[9]{2, -1, 0, -1, 1, 1, 0, 1, 2};
        case 3: /* Sharpen */
            return new float[9]{0, -1, 0, -1, 5, -1, 0, -1, 0};
        case 4: /* Outline */
            return new float[9]{-1, -1, -1, -1, 8, -1, -1, -1, -1};
        case 5: /* Bottom sobel */
            return new float[9]{-1, -2, -1, 0, 0, 0, 1, 2, 1};
        case 6: /* Ridge */
            return new float[9]{0, -1, 0, -1, 4, -1, 0, -1, 0};
        case 7: /* Edge detection */
            return new float[9]{-1, -1, -1, -1, 8, -1, -1, -1, -1};
        case 8: /* Box Blur */
            return new float[9]{0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111};
        default: /* Default to identity */
            return new float[9]{0, 0, 0, 0, 1, 0, 0, 0, 0};
    }
}

__host__ void assignDatas(unsigned char *imageDatas, int width, int height, int channels, cv::Mat *image) {
    int arrI = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = image->at<cv::Vec3b>(y, x);
            for (int c = channels - 1; c >= 0; --c) {
                imageDatas[arrI++] = pixel[c]; //RGB format
            }
        }
    }
}

__host__ void printRGBValues(float *imageDatas, int totalSize, int channels) {
    int pixelID = 0;
    for (int i = 0; i < totalSize; i += channels) {
        printf("-----------PIXEL ID:%d-----------\n", pixelID);
        printf("R:%f, G:%f, B:%f\n", imageDatas[i], imageDatas[i + 1], imageDatas[i + 2]);
        pixelID++;
    }
}

/*<--------------------------------------END OF HOST FUNCTIONS-------------------------------------->*/

/*<--------------------------------------DEVICE FUNCTIONS-------------------------------------->*/
__device__ float devCLamp(float val) {
    return val < 0 ? 0 : val > 255 ? 255 : val;
}

/*<--------------------------------------END OF DEVICE FUNCTIONS-------------------------------------->*/

/*<--------------------------------------GLOBAL KERNELS-------------------------------------->*/

__global__ void kernelConvolution(unsigned char *d_imageDatas, unsigned char *d_outputImageDatas) {
    __shared__ unsigned char sharedMem[w_y][w_x];
    int globalY = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int globalX = blockIdx.x * TILE_WIDTH + threadIdx.x;

    /*FOR EVERY CHANNEL*/
    for (int k = 0; k < CHANNELS; ++k) {
        /*Perform first load on shared memory (top part of corresponding matrix)*/
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w_x; /*Y coordinate with respect to shared memory*/
        int destX = dest % w_x; /*X coordinate with respect to shared memory*/

        int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
        int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;
        int src = (srcY * IMG_WIDTH + srcX) * CHANNELS + k;

        if (srcY >= 0 && srcY < IMG_HEIGHT && srcX >= 0 && srcX < IMG_WIDTH) {
            sharedMem[destY][destX] = d_imageDatas[src];
        } else {
            sharedMem[destY][destX] = 0;
        }
        /*Second loading*/
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w_x;
        destX = dest % w_x;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;
        src = (srcY * IMG_WIDTH + srcX) * CHANNELS + k;

        if (destY < w_y && destX < w_x) {
            if (srcY >= 0 && srcY < IMG_HEIGHT && srcX >= 0 && srcX < IMG_WIDTH) {
                sharedMem[destY][destX] = d_imageDatas[src];
            } else {
                sharedMem[destY][destX] = 0;
            }
        }
        /*Threads sync after loading on shared mem*/
        __syncthreads();
        if (globalY < IMG_HEIGHT && globalX < IMG_WIDTH) {
            /* IF thread in Image bounds, perform convolution*/
            int x = 0;
            int y = 0;
            float sum = 0;
            for (y = 0; y < MASK_HEIGHT; ++y) {
                for (x = 0; x < MASK_WIDTH; ++x) {
                    sum += KERNEL[y * MASK_WIDTH + x] * sharedMem[threadIdx.y + y][threadIdx.x + x];
                }
            }
            y = blockIdx.y * TILE_WIDTH + threadIdx.y;
            x = blockIdx.x * TILE_WIDTH + threadIdx.x;
            if (y < IMG_HEIGHT && x < IMG_WIDTH) {
                d_outputImageDatas[(y * IMG_WIDTH + x) * CHANNELS + k] = (unsigned char) devCLamp(sum);
            }
        }
    }
}

/*<--------------------------------------END OF GLOBAL KERNELS-------------------------------------->*/

int main() {

    cv::Mat image = cv::imread(
        R"(C:\Users\simon\Desktop\Uni\Magistrale\Progetti\Parallel Programming CUDA\Datasets\kernelImgProcessingDataset\22.jpg)");
    if (image.empty()) {
        std::cerr << "ERROR: Image NOT found!" << std::endl;
        return -1;
    }

    cv::Mat outputImage = image.clone();
    int channels = 3;
    int imageWidth = outputImage.cols;
    int imageHeight = outputImage.rows;
    /*<--------------------------------DEVICE MEM ALLOC-------------------------------->*/
    auto beginCUDATime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO START*/
    cudaSetDevice(0);
    unsigned char *d_imageDatas;
    unsigned char *d_outputImageDatas;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    size_t totalSize = imageWidth * imageHeight * channels * sizeof(unsigned char);

    CUDA_CHECK(cudaMallocAsync(&d_imageDatas, totalSize, stream));
    CUDA_CHECK(cudaMallocAsync(&d_outputImageDatas, totalSize, stream));

    auto endCUDATime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO END*/
    std::chrono::duration<double, std::milli> ms_double_cuda = endCUDATime - beginCUDATime;


    /*<--------------------------------HOST MEM ALLOC-------------------------------->*/
    auto beginHOSTTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO START*/
    unsigned char *imageDatas = (unsigned char *) malloc(imageWidth * imageHeight * channels * sizeof(unsigned char));
    unsigned char *outputImageDatas = (unsigned char *) malloc(
        imageWidth * imageHeight * channels * sizeof(unsigned char));

    /*<------------------------------IMAGE RGB DATA ASSIGNMENT------------------------------>*/
    assignDatas(imageDatas, imageWidth, imageHeight, channels, &outputImage);
    auto endHOSTTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO END*/
    std::chrono::duration<double, std::milli> ms_double_host = endHOSTTime - beginHOSTTime;

    printf(
        "PLEASE CHOOSE A KERNEL TO BE USED:\n 0: Identity\n 1: Blur\n 2: Emboss\n 3: Sharpen\n 4: Outline\n 5: Bottom sobel\n 6: Ridge\n 7: Edge detection\n 8: Box Blur\n NOTE: If no valid input is provided, the IDENTITY kernel will be used!\n");
    int choosenKernel;
    std::cin >> choosenKernel;
    float *kernel = getKernel(choosenKernel);

    /*<--------------------------------DEVICE MEM COPY-------------------------------->*/
    auto beginCUDAMEMTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO START*/
    CUDA_CHECK(cudaMemcpyAsync(d_imageDatas, imageDatas, totalSize, cudaMemcpyHostToDevice, stream));


    /*<------------------------------DEVICE CONSTANT MEM SET------------------------------>*/
    CUDA_CHECK(cudaMemcpyToSymbol(KERNEL, kernel, MASK_WIDTH * MASK_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(CHANNELS, &channels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(IMG_WIDTH, &imageWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(IMG_HEIGHT, &imageHeight, sizeof(int)));

    auto endCUDAMEMTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO END*/
    std::chrono::duration<double, std::milli> ms_double_dev_mem = endCUDAMEMTime - beginCUDAMEMTime;

    /*<--------------------------------KERNEL CALL-------------------------------->*/
    auto beginKernelTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO START*/

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

    kernelConvolution<<<gridDim, blockDim, 0, stream>>>(d_imageDatas, d_outputImageDatas);

    auto endKernelTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO END*/
    std::chrono::duration<double, std::milli> ms_double_kernel = endKernelTime - beginKernelTime;

    /*<----------------------------------------------DEVICE -----> HOST DATA COPY-------------------------------->*/

    auto beginDevHostTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO START*/
    CUDA_CHECK(cudaMemcpyAsync(outputImageDatas, d_outputImageDatas, totalSize, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto endDevHostTime = std::chrono::high_resolution_clock::now(); /*CUDA CHRONO END*/
    std::chrono::duration<double, std::milli> ms_double_dev_host = endDevHostTime - beginDevHostTime;

    /*<------------------------------FINAL IMAGE REBUILD------------------------------>*/
    buildImage(outputImageDatas, imageWidth, imageHeight, channels, &outputImage);
    CUDA_CHECK(cudaStreamDestroy(stream));

    /*<-------------------------CHRONOS PRINTING------------------------->*/


    printf("CUDA MEM ALLOC TIME: %f ms\n", ms_double_cuda.count());
    printf("HOST->DEV MEM COPY TIME: %f ms\n", ms_double_dev_mem.count());
    printf("KERNEL TIME: %f ms\n", ms_double_kernel.count());
    printf("DEV->HOST COPY TIME: %f ms\n", ms_double_dev_host.count());
    printf("HOST TIME: %f ms\n", ms_double_host.count());
    printf("TOTAL TIMING: %f ms\n", ms_double_cuda.count() + ms_double_dev_mem.count() + ms_double_kernel.count() +
                                    ms_double_dev_host.count() + ms_double_host.count());


    /*<------------------------------IMAGES RENDERING------------------------------>*/
        cv::imshow("Original Image", image);
        cv::imshow("Output Image", outputImage);
        cv::waitKey(0);

    return 0;
}