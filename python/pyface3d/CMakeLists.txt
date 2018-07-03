pybind11_add_module(pyface3d pybind_util.h pyface3d.cxx)
if (FACE3D_USE_CUDA)
  TARGET_COMPILE_DEFINITIONS(pyface3d PRIVATE FACE3D_USE_CUDA=1)
endif()
target_link_libraries(pyface3d PRIVATE face3d_basic face3d dlib)

set_target_properties(pyface3d PROPERTIES OUTPUT_NAME face3d)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE_DEFAULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(PYTHON_SITE ${PYTHON_SITE_DEFAULT} CACHE STRING "Python installation directory")

install(TARGETS pyface3d DESTINATION ${PYTHON_SITE})
