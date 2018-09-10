set(PYBIND11_DIR ${CMAKE_CURRENT_LIST_DIR}/../external/pybind11 CACHE PATH "Pybind11 root directory")
set(PYTHON_VERSION "3" CACHE STRING "Version of Python to build pybind11 modules for")
find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
add_subdirectory(${PYBIND11_DIR} "pybind11_build")

add_subdirectory(pyvxl)
add_subdirectory(pyface3d)
