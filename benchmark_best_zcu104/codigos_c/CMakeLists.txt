cmake_minimum_required(VERSION 3.10)
project(benchmark_metano_zcu104 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV QUIET opencv4)
if(NOT OPENCV_FOUND)
    pkg_check_modules(OPENCV REQUIRED opencv)
endif()
find_package(Threads REQUIRED)

add_library(benchmark_support STATIC
    dataset.cpp preprocess.cpp xmodel_runner.cpp postprocess.cpp metricas.cpp
    pipeline.cpp power.cpp estatisticas.cpp resultados.cpp validacao.cpp)
target_include_directories(benchmark_support PUBLIC ${OPENCV_INCLUDE_DIRS})
target_compile_options(benchmark_support PRIVATE -O3 -Wall -Wextra)
target_link_libraries(benchmark_support PUBLIC ${OPENCV_LIBRARIES}
    vart-runner vart-util vitis_ai_library-graph_runner xir Threads::Threads)

add_executable(benchmark_vitis benchmark_vitis.cpp)
target_link_libraries(benchmark_vitis PRIVATE benchmark_support)

add_executable(sweep_vitis sweep.cpp)
target_compile_options(sweep_vitis PRIVATE -O2 -Wall -Wextra)
target_link_libraries(sweep_vitis PRIVATE Threads::Threads)

add_executable(self_test_support self_test_support.cpp)
target_link_libraries(self_test_support PRIVATE benchmark_support)
