add_executable(approx_example main.c)
target_compile_options(approx_example PUBLIC -Wall -fsanitize=address -fsanitize=undefined)
target_link_options(approx_example PUBLIC -fsanitize=address -fsanitize=undefined)

target_link_libraries(approx_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(approx_example PUBLIC ${MATH_LIBRARY})
endif()

