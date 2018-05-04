set(project_SOURCES ${project_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
    )

##### ======= BUILD TESTS (-DBUILD_TESTING=ON)=======
if(BUILD_TESTING)
    message(STATUS "tests enabled")
    add_subdirectory(src/gtest)
else()
    message(STATUS "tests disabled")
endif(BUILD_TESTING)
##### ======= BUILD TESTS (-DBUILD_TESTING=ON)=======
