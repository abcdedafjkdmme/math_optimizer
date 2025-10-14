add_executable(neural_net_func_approx_example main.c)

target_link_libraries(neural_net_func_approx_example PUBLIC mfo_lib)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(neural_net_func_approx_example PUBLIC ${MATH_LIBRARY})
endif()

add_custom_target(copy-runtime-files-2 ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
    DEPENDS neural_net_func_approx_example)
