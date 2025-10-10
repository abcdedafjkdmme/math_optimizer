cmake_minimum_required(VERSION 3.10)
project(mfo_proj)


add_library(mfo_lib INTERFACE)
target_include_directories(mfo_lib INTERFACE "include/mfo")

add_subdirectory(examples)