cmake_minimum_required(VERSION 3.10)
project(Projet_OpenCV_CMake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(Projet_OpenCV_CMake
    include/histogram.hpp
    src/histogram.cpp
        src/main_test_opencv_old.cpp
        src/main_test_opencv.cpp
)
target_link_libraries(Projet_OpenCV_CMake ${OpenCV_LIBS})
