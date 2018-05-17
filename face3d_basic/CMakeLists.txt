set(FACE3D_BASIC_SOURCES
  image_conversion.h image_conversion.cxx
  dlib_object_detector.h dlib_object_detector.cxx
  dlib_face_detector.h dlib_face_detector.cxx
  ortho_camera_parameters.h ortho_camera_parameters.txx
  perspective_camera_parameters.h perspective_camera_parameters.txx
  io_utils.h io_utils.cxx
  face3d_util.h face3d_util.cxx
  face3d_img_util.h face3d_img_util.cxx
  affine_camera_approximator.h
  subject_sighting_coefficients.h
  sighting_coefficients.h
  cmd_line_util.h
)
aux_source_directory(templates FACE3D_BASIC_SOURCES)

add_library(face3d_basic ${FACE3D_BASIC_SOURCES})

target_link_libraries(face3d_basic vnl vnl_algo vul vgl vgl_algo vpgl vpgl_algo vil vil_algo dlib pthread)
set_target_properties(face3d_basic PROPERTIES POSITION_INDEPENDENT_CODE True)

add_subdirectory(tests)
