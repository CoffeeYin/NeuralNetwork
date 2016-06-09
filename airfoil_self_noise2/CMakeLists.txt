
add_executable(airfoil_self_noise main.cpp)
target_link_libraries(airfoil_self_noise opennn)


install(DIRECTORY data/ DESTINATION data/)
install(TARGETS airfoil_self_noise RUNTIME DESTINATION bin)


set(dir ${CMAKE_CURRENT_SOURCE_DIR}/../../build/examples/airfoil_self_noise/data/)

install(DIRECTORY ${dir} DESTINATION ${dir})
install(TARGETS airfoil_self_noise RUNTIME DESTINATION bin)
