cmake_minimum_required(VERSION 3.30)
project(ImageProcessingSEQ)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "C:\\Users\\simon\\Desktop\\Uni\\Magistrale\\Progetti\\Parallel Programming CUDA\\NiceLibs\\opencv\\build")

find_package(OpenCV REQUIRED)

add_executable(ImageProcessingSEQ main.cpp)
target_link_libraries(ImageProcessingSEQ ${OpenCV_LIBS})