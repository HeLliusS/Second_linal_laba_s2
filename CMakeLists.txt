cmake_minimum_required(VERSION 4.0)
project(LinalSecondLaba)

set(CMAKE_CXX_STANDARD 20)

add_executable(LinalSecondLaba main.cpp
        perceptron.cpp
        data_generator.cpp
        experiments.cpp
)

add_executable(experimentsSecondLaba run_experiments.cpp
        perceptron.cpp
        data_generator.cpp
        experiments.cpp
)

add_executable(nonlinearSecondLaba nonlinear_main.cpp
        perceptron.cpp
        data_generator.cpp
        experiments.cpp
)
