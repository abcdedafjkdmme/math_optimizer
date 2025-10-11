add_executable(neural_net_example main.c)
target_compile_options(neural_net_example PUBLIC
    $<$<CONFIG:Debug>: -Wall -Wextra -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align -Wstrict-prototypes -Wstrict-overflow=5 -Wwrite-strings -Waggregate-return -Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code -fsanitize=address -fsanitize=undefined> )
target_link_options(neural_net_example PUBLIC
    $<$<CONFIG:Debug>: -fsanitize=address -fsanitize=undefined>)

target_link_libraries(neural_net_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(neural_net_example PUBLIC ${MATH_LIBRARY})
endif()
