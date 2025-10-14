add_executable(approx_example main.c)

target_link_libraries(approx_example PUBLIC mfo_lib)
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(approx_example PUBLIC ${MATH_LIBRARY})
endif()

