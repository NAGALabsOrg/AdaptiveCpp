cmake_minimum_required(VERSION 3.24)
project(naga_sg_tests LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(AdaptiveCpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(Threads REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

file(GLOB HOST_SRC CONFIGURE_DEPENDS host/*.cpp)
file(GLOB COMPILER_SRC CONFIGURE_DEPENDS compiler/*.cpp)
file(GLOB DRIVER_SRC CONFIGURE_DEPENDS driver/*.cpp)
file(GLOB DEVICE_SRC CONFIGURE_DEPENDS device/*.cpp)

add_executable(sg_host_tests ${HOST_SRC})
add_sycl_to_target(TARGET sg_host_tests SOURCES ${HOST_SRC})
target_link_libraries(sg_host_tests PRIVATE Boost::unit_test_framework Threads::Threads)
target_compile_definitions(sg_host_tests PRIVATE BOOST_TEST_DYN_LINK)

add_executable(sg_compiler_tests ${COMPILER_SRC})
target_include_directories(sg_compiler_tests PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS}
  "$ENV{PREFIX}/include/AdaptiveCpp" "$ENV{PREFIX}/include")
target_compile_definitions(sg_compiler_tests PRIVATE BOOST_TEST_DYN_LINK)
target_link_libraries(sg_compiler_tests PRIVATE Boost::unit_test_framework
  clang-cpp LLVM)

add_executable(sg_driver_tests ${DRIVER_SRC})
add_sycl_to_target(TARGET sg_driver_tests SOURCES ${DRIVER_SRC})
target_link_libraries(sg_driver_tests PRIVATE Boost::unit_test_framework Threads::Threads)
target_compile_definitions(sg_driver_tests PRIVATE BOOST_TEST_DYN_LINK)

add_executable(sg_tests ${DEVICE_SRC})
add_sycl_to_target(TARGET sg_tests SOURCES ${DEVICE_SRC})
target_link_libraries(sg_tests PRIVATE Boost::unit_test_framework Threads::Threads)
target_compile_definitions(sg_tests PRIVATE BOOST_TEST_DYN_LINK)

add_executable(sg_jit_bench bench/main.cpp)
add_sycl_to_target(TARGET sg_jit_bench SOURCES bench/main.cpp)
target_link_libraries(sg_jit_bench PRIVATE Threads::Threads)

install(TARGETS sg_host_tests sg_compiler_tests sg_driver_tests sg_tests sg_jit_bench RUNTIME DESTINATION bin)
install(DIRECTORY lit/ DESTINATION share/naga-sg-tests/lit)
