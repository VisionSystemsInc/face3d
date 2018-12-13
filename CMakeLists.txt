project(face3d)
cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif(POLICY CMP0042)


set(OpenGL_GL_PREFERENCE "GLVND" CACHE STRING "Prefer Vendor-neutral OpenGL loader")
find_package(OpenGL)

message(STATUS "OPENGL_egl_LIBRARY = " ${OPENGL_egl_LIBRARY})
if (OPENGL_egl_LIBRARY)
  set(FACE3D_USE_EGL TRUE CACHE BOOL "Use EGL for OpenGL context generation")
else()
  set(FACE3D_USE_EGL FALSE CACHE BOOL "Use EGL for OpenGL context generation")
endif()
message("FACE3D_USE_EGL = ${FACE3D_USE_EGL}")

if (FACE3D_USE_EGL)
  # Some platforms may not have recent EGL headers installed to the system
  set(EGL_INCLUDE_DIR "" CACHE PATH "path containing EGL headers dir")
  if (EGL_INCLUDE_DIR)
    include_directories(BEFORE ${EGL_INCLUDE_DIR})
    message(STATUS "EGL_INCLUDE_DIR = " ${EGL_INCLUDE_DIR})
  endif()
  add_definitions("-DFACE3D_USE_EGL=1")
endif()


set(GLAD_OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/external/glad_out")
set(GLAD_INCLUDE_DIR "${GLAD_OUT_DIR}/include")
set(GLAD_SOURCES "${GLAD_OUT_DIR}/src/glad.c")
include_directories(${GLAD_INCLUDE_DIR})
message("GLAD_INCLUDE_DIR = ${GLAD_INCLUDE_DIR}")


SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# setup vxl
set(vxl_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/vxl)
set(WITH_FFMPEG FALSE CACHE BOOL "Include FFMPEG support if available")
set(VXL_USE_GEOTIFF FALSE CACHE BOOL "Use the GEOTIFF Library if available")
add_subdirectory(${vxl_SRC_DIR} EXCLUDE_FROM_ALL)
set(VXL_CORE_INCLUDE_DIR ${VXL_BINARY_DIR}/core ${vxl_SRC_DIR}/core)
set(VXL_VCL_INCLUDE_DIR ${VXL_BINARY_DIR}/vcl ${vxl_SRC_DIR}/vcl)
include_directories(${VXL_CORE_INCLUDE_DIR})
include_directories(${VXL_VCL_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 11)

# setup dlib
set(DLIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/dlib CACHE PATH "root dlib directory")
set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "USE SSE4 instructions in Dlib")
SET(DLIB_GIF_SUPPORT OFF CACHE BOOL "DLIB GIF Support") # incompatible DLIB API call with system libgif
SET(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable DLIB GUI Support") # missing X11 header under anaconda
SET(DLIB_GIF_SUPPORT OFF CACHE BOOL "DLIB GIF Support")
SET(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable DLIB GUI Support")
add_subdirectory(${DLIB_SRC_DIR}/dlib dlib_build)
include_directories(${DLIB_SRC_DIR})

# setup libigl
# raise a compiler error if any LPGL code is including from Eigen
add_definitions("-DEIGEN_MPL2_ONLY")
set(libigl_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/libigl)
list(APPEND CMAKE_MODULE_PATH "${libigl_SRC_DIR}/cmake")

set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "Use LibIGL as static library" FORCE)
set(LIBIGL_WITH_VIEWER OFF CACHE BOOL "Use OpenGL viewer"  FORCE)
set(LIBIGL_WITH_COMISO OFF CACHE BOOL "Use CoMiso" FORCE)
set(LIBIGL_WITH_LIM OFF CACHE BOOL "Use LIM" FORCE)
set(LIBIGL_WITH_NANOGUI OFF CACHE BOOL "Use Nanogui menu" FORCE)
set(LIBIGL_WITH_EMBREE OFF CACHE BOOL "Use Embree" FORCE)
set(LIBIGL_WITH_TRIANGLE ON CACHE BOOL "Use Triangle" FORCE)
set(NANOGUI_INSTALL OFF CACHE BOOL "Install nanogui to system")
if(${FACE3D_USE_EGL})
  set(LIBIGL_WITH_OPENGL_GLFW  OFF CACHE BOOL "Use OpenGL GLFW" FORCE)
  set(LIBIGL_WITH_OPENGL OFF CACHE BOOL  "Use OpenGL"  FORCE)
  set(LIBIGL_LIBRARIES igl::core igl::triangle)
else()
  set(LIBIGL_WITH_OPENGL_GLFW  ON CACHE BOOL "Use OpenGL GLFW" FORCE)
  set(LIBIGL_WITH_OPENGL       ON CACHE BOOL  "Use OpenGL"  FORCE)
  set(LIBIGL_LIBRARIES igl::core igl::triangle igl::opengl_glfw)
endif()
include(libigl)

set(LIBIGL_INCLUDE_DIR ${libigl_SRC_DIR}/include)
set(GLFW_INCLUDE_DIR ${libigl_SRC_DIR}/external/glfw/include)
include_directories(${LIBIGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})

# Use libigl's version of Eigen
set(EIGEN_INCLUDE_DIR ${libigl_SRC_DIR}/external/eigen)
message(STATUS "EIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}")
include_directories(${EIGEN_INCLUDE_DIR})





include_directories(${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(face3d_basic)
add_subdirectory(face3d)
add_subdirectory(python)
