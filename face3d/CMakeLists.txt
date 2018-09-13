# configure the directory where the glsl shaders will live
set(FACE3D_SHADER_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
# add definition so that directory can be used in source code
add_definitions(-DFACE3D_SHADER_DIR="${FACE3D_SHADER_DIR}")

find_package(CUDA)
set(FACE3D_USE_CUDA ${CUDA_FOUND} CACHE BOOL "Use CUDA Optimizations in face3d")
message(STATUS "FACE3D_USE_CUDA=${FACE3D_USE_CUDA} CUDA_FOUND=${CUDA_FOUND}")

set(FACE3D_SOURCES
  media_coefficient_from_semantic_map_estimator.h media_coefficient_from_semantic_map_estimator.cxx
  semantic_map.h semantic_map.cxx
  camera_estimation.h camera_estimation.cxx
  triangle_mesh.h triangle_mesh.cxx
  textured_triangle_mesh.h
  mesh_io.h mesh_io.cxx
  mesh_renderer.h mesh_renderer.cxx
  offset_correction.h offset_correction.cxx
  media_jitterer.h
  texture_map_operations.h texture_map_operations.cxx
  head_mesh.h head_mesh.cxx
  background_renderer.h
  mesh_background_renderer.h
  media_coefficient_from_PNCC_and_offset_estimator.h media_coefficient_from_PNCC_and_offset_estimator.cxx
  pose_jitterer.h pose_jitterer.cxx
  face3d_cuda_context.h face3d_cuda_context.cxx
  coeffs_to_pixmap.h coeffs_to_pixmap.cxx
  estimation_results.h
  ${GLAD_SOURCES}
  )
if(FACE3D_USE_CUDA)
   set(FACE3D_SOURCES ${FACE3D_SOURCES} localize_points.cu)
endif()

message(STATUS "LIBIGL_LIBRARIES = ${LIBIGL_LIBRARIES}")

if(FACE3D_USE_CUDA)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -shared -Xcompiler -fPIC -std=c++11")
  CUDA_ADD_LIBRARY(face3d ${FACE3D_SOURCES})
  TARGET_COMPILE_DEFINITIONS(face3d PRIVATE FACE3D_USE_CUDA=1)
  target_link_libraries(face3d face3d_basic ${LIBIGL_LIBRARIES} dlib vnl vnl_algo vul vgl vgl_algo vpgl vpgl_algo vil ${CUDA_LIBRARY} -ldl)
else()
  add_library(face3d ${FACE3D_SOURCES})
  target_link_libraries(face3d face3d_basic ${LIBIGL_LIBRARIES} dlib vnl vnl_algo vul vgl vgl_algo vpgl vpgl_algo vil  -ldl)
endif()
set_target_properties(face3d PROPERTIES POSITION_INDEPENDENT_CODE True)

if (FACE3D_USE_EGL)
  target_link_libraries(face3d ${OPENGL_egl_LIBRARY})
endif()
