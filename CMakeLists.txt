# 1. Set the minimum version and project name
cmake_minimum_required(VERSION 3.10)
project(WineNeuralNetwork)

# 2. Tell it to use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Add your source files
set(SOURCES 
    main.cpp 
    core_engine.cpp 
    person_b_pipeline.cpp 
    interface.cpp
)

# 4. Create the executable
add_executable(main ${SOURCES})

# 5. Tell it where to find SFML
target_include_directories(main PRIVATE ./include)
target_link_directories(main PRIVATE ./lib)
target_link_libraries(main PRIVATE sfml-graphics sfml-window sfml-system)