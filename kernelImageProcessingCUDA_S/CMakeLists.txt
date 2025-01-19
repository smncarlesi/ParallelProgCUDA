cmake_minimum_required(VERSION 3.30)
project(ImageProcessingCUDA_S CUDA)

set(CMAKE_CUDA_STANDARD 20)

# Percorso di OpenCV e CUDA
set(OpenCV_DIR "C:\\Users\\simon\\Desktop\\Uni\\Magistrale\\Progetti\\Parallel Programming CUDA\\NiceLibs\\opencv\\build")
set(CUDA_LIB_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib\\x64")

# Aggiungi il percorso delle librerie
link_directories(${CUDA_LIB_DIR})

# Trova OpenCV
find_package(OpenCV REQUIRED)

# Includi le directory di OpenCV
include_directories("${OpenCV_DIR}/include")
include_directories("${OpenCV_DIR}/include/opencv2")
include_directories(${OpenCV_INCLUDE_DIRS})

# Crea l'eseguibile
add_executable(ImageProcessingCUDA_S main.cu)

# Collega le librerie necessarie
target_link_libraries(ImageProcessingCUDA_S
        PRIVATE
        ${OpenCV_LIBS}
        "${OpenCV_DIR}/x64/vc16/lib/opencv_world4100.lib"
        cuda)

# Proprietà specifiche di CUDA
set_target_properties(ImageProcessingCUDA_S PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
