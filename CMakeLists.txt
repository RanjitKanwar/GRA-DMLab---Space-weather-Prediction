cmake_minimum_required(VERSION 3.15...3.19)

project(swdatatoolkit VERSION 0.0.1 LANGUAGES CXX)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE src/swdatatoolkit/main.cpp)
pybind11_add_module(_core2 MODULE src/swdatatoolkit/main2.cpp)

target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_definitions(_core2 PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _core LIBRARY DESTINATION _core)
install(TARGETS _core2 LIBRARY DESTINATION _core2)