add_executable(err_func_example main.c)

target_link_libraries(err_func_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(err_func_example PUBLIC ${MATH_LIBRARY})
endif()
