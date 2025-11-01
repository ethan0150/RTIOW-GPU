cmake_minimum_required(VERSION 3.20)

project(RayTracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages installed via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

# Define the executable
add_executable(raytracer main.cpp)

# Link libraries
target_link_libraries(raytracer PRIVATE
    glfw
    GLEW::GLEW
    glm::glm
    OpenGL::GL
)

# Ensure the executable is placed in the binary directory
set_target_properties(raytracer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Optional: Add include directories if needed (usually handled by vcpkg)
# target_include_directories(raytracer PRIVATE ${GLFW3_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} ${GLM_INCLUDE_DIR})