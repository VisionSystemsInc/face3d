set(PYBIND11_DIR ${CMAKE_CURRENT_LIST_DIR}/../external/pybind11 CACHE PATH "Pybind11 root directory")
find_package(PythonLibs 3 REQUIRED)
add_subdirectory(${PYBIND11_DIR} "pybind11_build")

add_subdirectory(pyvxl)
add_subdirectory(pyface3d)
