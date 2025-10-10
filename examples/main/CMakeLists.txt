add_executable(main_example main.c)
target_compile_options(main_example PUBLIC -Wall -fsanitize=address -fsanitize=undefined)
target_link_options(main_example PUBLIC -fsanitize=address -fsanitize=undefined)

target_link_libraries(main_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(main_example PUBLIC ${MATH_LIBRARY})
endif()
