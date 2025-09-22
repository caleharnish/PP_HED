##################################
# Define the homework #1 binary. #
##################################
add_executable(hw1 HW1.cpp)
target_include_directories(hw1 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hw1 pp_hed)


