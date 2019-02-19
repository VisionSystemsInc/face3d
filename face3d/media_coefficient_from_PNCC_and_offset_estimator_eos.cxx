#include "media_coefficient_from_PNCC_and_offset_estimator_eos.h"

face3d::media_coefficient_from_PNCC_and_offset_estimator_eos::
  media_coefficient_from_PNCC_and_offset_estimator_eos(head_mesh const& mesh,
                                                       std::string model_filename,
                                                       std::string blendshapes_filename,
                                                       bool debug_mode, std::string debug_dir,
                                                       double fixed_focal_len):
    base_mesh_(mesh), debug_mode_(debug_mode), debug_dir_(debug_dir),modelfile_(model_filename), blendshapesfile_(blendshapes_filename), fixed_focal_len_(fixed_focal_len)
{
  mean_face_mesh_ = mesh.face_mesh();
#ifndef FACE3D_USE_CUDA
  mean_face_mesh_tree_.init(mean_face_mesh_.V(), mean_face_mesh_.F());
#endif //FACE3D_USE_CUDA
}
