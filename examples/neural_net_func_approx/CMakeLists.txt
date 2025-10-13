add_executable(neural_net_func_approx_example main.c)
target_compile_options(neural_net_func_approx_example PUBLIC
    -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code 
    $<$<CONFIG:Debug>: -pg -fsanitize=address -fsanitize=undefined> 
    $<$<CONFIG:RelWithDebInfo>: -pg -fsanitize=address -fsanitize=undefined>
    )
target_link_options(neural_net_func_approx_example PUBLIC
    $<$<CONFIG:Debug>: -pg -fsanitize=address -fsanitize=undefined>
    $<$<CONFIG:RelWithDebInfo>: -pg -fsanitize=address -fsanitize=undefined>
    )

target_link_libraries(neural_net_func_approx_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(neural_net_func_approx_example PUBLIC ${MATH_LIBRARY})
endif()

add_custom_target(copy-runtime-files-2 ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
    DEPENDS neural_net_func_approx_example)
