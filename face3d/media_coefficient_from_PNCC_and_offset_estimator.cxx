#include "media_coefficient_from_PNCC_and_offset_estimator.h"

face3d::media_coefficient_from_PNCC_and_offset_estimator::
media_coefficient_from_PNCC_and_offset_estimator(head_mesh const& mesh,
                                                 vnl_matrix<double> const& subject_pca_components,
                                                 vnl_matrix<double> const& expression_pca_components,
                                                 vnl_matrix<double> const& subject_pca_ranges,
                                                 vnl_matrix<double> const& expression_pca_ranges,
                                                 bool debug_mode, std::string debug_dir)
  : base_mesh_(mesh),
  subject_pca_components_(subject_pca_components), expression_pca_components_(expression_pca_components),
  subject_pca_ranges_(subject_pca_ranges), expression_pca_ranges_(expression_pca_ranges),
  debug_mode_(debug_mode), debug_dir_(debug_dir)
{
  mean_face_mesh_ = mesh.face_mesh();
#ifndef FACE3D_USE_CUDA
  mean_face_mesh_tree_.init(mean_face_mesh_.V(), mean_face_mesh_.F());
#endif //FACE3D_USE_CUDA
}
