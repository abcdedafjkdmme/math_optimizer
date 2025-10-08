add_executable(optimizer_example main.c)
target_compile_options(optimizer_example PUBLIC -Wall -fsanitize=address -fsanitize=undefined)
target_link_options(optimizer_example PUBLIC -fsanitize=address -fsanitize=undefined)

target_link_libraries(optimizer_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(optimizer_example PUBLIC ${MATH_LIBRARY})
endif()
