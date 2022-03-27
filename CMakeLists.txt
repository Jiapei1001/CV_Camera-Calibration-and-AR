cmake_minimum_required(VERSION 3.15)

project(A4_CALIBRATION_AND_AR)
find_package(OpenCV REQUIRED)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS})
# Bring the headers, such as *.h into the project
include_directories(include)
# Include project compile path
include_directories(${PROJECT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Can manually add the sources using the set command as follows:
# set(SOURCES src/imgDisplay.cpp)

file(GLOB SOURCES "src/*.cpp")

add_executable(calibrationAndAR src/calibrationAndAR.cpp src/calibration.cpp)
add_executable(AR src/cameraAndAR.cpp src/ar.cpp)

target_link_libraries(calibrationAndAR ${OpenCV_LIBS})
target_link_libraries(AR ${OpenCV_LIBS})

