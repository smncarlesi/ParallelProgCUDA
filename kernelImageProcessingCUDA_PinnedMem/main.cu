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

__global__ void kernelConvolution(unsigned char *d_imageDatas, unsigned char *d_outputImageDatas, int startRow,
                                  int numRows) {
    __shared__ unsigned char sharedMem[w_y][w_x];
    int globalY = blockIdx.y * TILE_WIDTH + threadIdx.y + startRow;
    int globalX = blockIdx.x * TILE_WIDTH + threadIdx.x;

    /*FOR EVERY CHANNEL*/
    for (int k = 0; k < CHANNELS; ++k) {
        /*Perform first load on shared memory (top part of corresponding matrix)*/
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w_x; /*Y coordinate with respect to shared memory*/
        int destX = dest % w_x; /*X coordinate with respect to shared memory*/

        int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
        int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;
        srcY = max(0, min(IMG_HEIGHT - 1, srcY));
        srcX = max(0, min(IMG_WIDTH - 1, srcX));

        if (destY < w_y && destX < w_x) {
            sharedMem[destY][destX] = d_imageDatas[(srcY * IMG_WIDTH + srcX) * CHANNELS + k];
        }

        /*Second loading*/
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w_x;
        destX = dest % w_x;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS_X;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS_Y;

        srcY = max(0, min(IMG_HEIGHT - 1, srcY));
        srcX = max(0, min(IMG_WIDTH - 1, srcX));

        if (destY < w_y && destX < w_x) {
            sharedMem[destY][destX] = d_imageDatas[(srcY * IMG_WIDTH + srcX) * CHANNELS + k];
        }

        /*Threads sync after loading on shared mem*/
        __syncthreads();
        if (globalY >= startRow + MASK_RADIUS_Y && globalY < startRow + numRows + MASK_RADIUS_Y && globalX <
            IMG_WIDTH) {
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
                d_outputImageDatas[((globalY - startRow) * IMG_WIDTH + x) * CHANNELS + k] = (unsigned char)
                        devCLamp(sum);
            }
        }
    }
}

/*<--------------------------------------END OF GLOBAL KERNELS-------------------------------------->*/
int main() {
    cv::Mat image = cv::imread(
        R"(path_to_image)");
    if (image.empty()) {
        std::cerr << "ERROR: Image NOT found!" << std::endl;
        return -1;
    }

    cv::Mat outputImage = image.clone();
    int channels = 3;
    int imageWidth = outputImage.cols;
    int imageHeight = outputImage.rows;

    printf(
        "PLEASE CHOOSE A KERNEL TO BE USED:\n 0: Identity\n 1: Blur\n 2: Emboss\n 3: Sharpen\n 4: Outline\n 5: Bottom sobel\n 6: Ridge\n 7: Edge detection\n 8: Box Blur\n NOTE: If no valid input is provided, the IDENTITY kernel will be used!\n");
    int choosenKernel;
    std::cin >> choosenKernel;
    float *kernel = getKernel(choosenKernel);
    /*<--------------------------------DEVICE MEM ALLOC-------------------------------->*/
    auto beginTime = std::chrono::high_resolution_clock::now();
    cudaSetDevice(0);
    size_t totalSize = imageWidth * imageHeight * channels * sizeof(unsigned char);
    unsigned char *d_imageDatas, *d_outputImageDatas;

    CUDA_CHECK(cudaMalloc(&d_imageDatas, totalSize));
    CUDA_CHECK(cudaMalloc(&d_outputImageDatas, totalSize));


    /*<--------------------------------HOST MEM ALLOC (PINNED MEMORY)-------------------------------->*/
    unsigned char *imageDatas, *outputImageDatas;
    CUDA_CHECK(cudaMallocHost(&imageDatas, totalSize)); // Pinned memory allocation
    CUDA_CHECK(cudaMallocHost(&outputImageDatas, totalSize)); // Pinned memory allocation
    assignDatas(imageDatas, imageWidth, imageHeight, channels, &outputImage);

    /*<--------------------------------DEVICE CONSTANT MEM SET-------------------------------->*/
    CUDA_CHECK(cudaMemcpyToSymbol(KERNEL, kernel, MASK_WIDTH * MASK_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(CHANNELS, &channels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(IMG_WIDTH, &imageWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(IMG_HEIGHT, &imageHeight, sizeof(int)));

    /*<--------------------------------MULTI-STREAM SETUP-------------------------------->*/
    const int streamsNumber = 4;
    cudaStream_t streams[streamsNumber];
#pragma unroll
    for (int i = 0; i < streamsNumber; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int rowsPerStream = imageHeight / streamsNumber;
    int overlap = MASK_RADIUS_Y;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    for (int i = 0; i < streamsNumber; ++i) {
        int startRow = i * rowsPerStream - (i > 0 ? overlap : 0);
        int numRows = rowsPerStream + (i > 0 ? overlap : 0) + overlap + 1;

        if (startRow + numRows > imageHeight) {
            numRows = imageHeight - startRow;
        }

        unsigned char *h_chunkSrc = imageDatas + startRow * imageWidth * channels;
        unsigned char *h_chunkDst = outputImageDatas + (i * rowsPerStream) * imageWidth * channels;
        unsigned char *d_chunkSrc = d_imageDatas + startRow * imageWidth * channels;
        unsigned char *d_chunkDst = d_outputImageDatas + (i * rowsPerStream) * imageWidth * channels;

        size_t chunkBytes = numRows * imageWidth * channels;

        /* Asynchronous memory transfer (Host to Device) */
        CUDA_CHECK(cudaMemcpyAsync(d_chunkSrc, h_chunkSrc, chunkBytes, cudaMemcpyHostToDevice, streams[i]));

        /* Kernel launch */
        dim3 gridDim((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, (numRows + TILE_WIDTH - 1) / TILE_WIDTH);

        kernelConvolution<<<gridDim, blockDim, 0, streams[i]>>>(d_chunkSrc, d_chunkDst, startRow, numRows);

        /* Asynchronous memory transfer (Device to Host) */
        CUDA_CHECK(
            cudaMemcpyAsync(h_chunkDst, d_chunkDst, rowsPerStream * imageWidth * channels, cudaMemcpyDeviceToHost,
                streams[i]));
    }
    /* Synchronize all streams */
    for (int i = 0; i < streamsNumber; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = endTime - beginTime;
    printf("Duration ms: %f\n", ms_double.count());

    /*<------------------------------FINAL IMAGE REBUILD------------------------------>*/
    buildImage(outputImageDatas, imageWidth, imageHeight, channels, &outputImage);

    /* Memory freeing */
    CUDA_CHECK(cudaFreeHost(imageDatas)); // Free pinned memory
    CUDA_CHECK(cudaFreeHost(outputImageDatas)); // Free pinned memory
    CUDA_CHECK(cudaFree(d_imageDatas));
    CUDA_CHECK(cudaFree(d_outputImageDatas));


    cv::imshow("Original Image", image);
    cv::imshow("Output Image", outputImage);
    cv::waitKey(0);

    return 0;
}