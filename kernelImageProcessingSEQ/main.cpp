#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct ImageStruct {
private:
    int rows;
    int cols;

public:
    std::vector<unsigned char> R;
    std::vector<unsigned char> G;
    std::vector<unsigned char> B;

    ImageStruct(cv::Mat *image) {
        this->rows = image->rows;
        this->cols = image->cols;
        R.resize(rows * cols);
        G.resize(rows * cols);
        B.resize(rows * cols);
        assignValues(image);
    }

    int getRows() { return rows; }
    int getCols() { return cols; }

private:
    void assignValues(cv::Mat *image) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                cv::Vec3b actPixelRGB = image->at<cv::Vec3b>(y, x);
                int index = cols * y + x;
                B[index] = actPixelRGB[0];
                G[index] = actPixelRGB[1];
                R[index] = actPixelRGB[2];
            }
        }
    }
};

ImageStruct kernelProcessing(ImageStruct *imageStruct, const std::vector<float> &kernel, cv::Mat *finalImage) {
    int imgCols = imageStruct->getCols();
    int imgRows = imageStruct->getRows();
    int kernelSize = std::sqrt(kernel.size());
    int pad = kernelSize / 2;

    cv::Mat paddedImage;
    cv::copyMakeBorder(*finalImage, paddedImage, pad, pad, pad, pad, cv::BORDER_REFLECT);
    ImageStruct newImageStruct = ImageStruct(&paddedImage);

    for (int y = 0; y < imgRows; ++y) {
        for (int x = 0; x < imgCols; ++x) {
            float actR = 0, actG = 0, actB = 0;

            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int py = y + ky;
                    int px = x + kx;
                    int effCols = imgCols + 2 * pad;
                    unsigned char rVal = newImageStruct.R[py * effCols + px];
                    unsigned char gVal = newImageStruct.G[py * effCols + px];
                    unsigned char bVal = newImageStruct.B[py * effCols + px];

                    float weight = kernel[ky * kernelSize + kx];
                    actB += weight * (float) bVal;
                    actG += weight * (float) gVal;
                    actR += weight * (float) rVal;
                }
            }
            newImageStruct.R[y * imgCols + x] = static_cast<unsigned char>(std::clamp(actR, 0.0f, 255.0f));
            newImageStruct.G[y * imgCols + x] = static_cast<unsigned char>(std::clamp(actG, 0.0f, 255.0f));
            newImageStruct.B[y * imgCols + x] = static_cast<unsigned char>(std::clamp(actB, 0.0f, 255.0f));
        }
    }
    return (newImageStruct);
}

std::vector<float> getKernel(int choosenKernel) {
    const float gaussianCoeff = 0.003906251;
    switch (choosenKernel) {
        case 0: /* Identity */
            return {0, 0, 0, 0, 1, 0, 0, 0, 0};
        case 1: /* Blur */
            return {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
        case 2: /* Emboss */
            return {2, -1, 0, -1, 1, 1, 0, 1, 2};
        case 3: /* Sharpen */
            return {0, -1, 0, -1, 5, -1, 0, -1, 0};
        case 4: /* Outline */
            return {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        case 5: /* Bottom sobel */
            return {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        case 6: /* Ridge */
            return {0, -1, 0, -1, 4, -1, 0, -1, 0};
        case 7: /* Edge detection */
            return {-1, -1, -1, -1, 8, -1, -1, -1, -1};
        case 8: /* Box Blur */
            return {0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111};
        default: /* Default to identity */
            return {0, 0, 0, 0, 1, 0, 0, 0, 0};
    }
}

void buildFinalImage(ImageStruct *finalImageStruct, cv::Mat *outputImage, int kernelSize) {
    int cols = outputImage->cols;
    int rows = outputImage->rows;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            outputImage->at<cv::Vec3b>(y, x) = cv::Vec3b(
                finalImageStruct->B[y * cols + x],
                finalImageStruct->G[y * cols + x],
                finalImageStruct->R[y * cols + x]);
        }
    }
}

int main() {
    cv::Mat image = cv::imread(
        R"(path_to_image)");
    if (image.empty()) {
        std::cerr << "ERROR: Image NOT found!" << std::endl;
        return -1;
    }

    cv::Mat outputImage = image.clone();
    printf(
        "PLEASE CHOOSE A KERNEL TO BE USED:\n 0: Identity\n 1: Blur\n 2: Emboss\n 3: Sharpen\n 4: Outline\n 5: Bottom sobel\n 6: Ridge\n 7: Edge detection\n 8: Box Blur\n NOTE: If no valid input is provided, the IDENTITY kernel will be used!\n");
    int choosenKernel;
    std::cin >> choosenKernel;
    /*<------------------CHRONO START------------------>*/
    auto beginTime = std::chrono::high_resolution_clock::now();
    ImageStruct imgStruct(&outputImage);
    std::vector<float> kernel = getKernel(choosenKernel);
    ImageStruct finalStruct = kernelProcessing(&imgStruct, kernel, &outputImage);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = endTime - beginTime;
    buildFinalImage(&finalStruct, &outputImage, kernel.size());
    /*<------------------CHRONO END------------------>*/
    printf("Duration ms: %f\n", ms_double.count());


    cv::imshow("Original Image", image);
    cv::imshow("Processed Image", outputImage);
    cv::waitKey(0);
    return 0;
}
