cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")

project(yolov5 VERSION 0.1.0)

find_package(OpenCV 4.5.4 REQUIRED)

#add_executable(main yolov5.cpp)
add_executable(main ObjectDetector.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
