#include "media_coefficient_from_semantic_map_estimator.h"
#include <array>
#include "semantic_map.h"
#include <vil/algo/vil_binary_erode.h>
#include <vil/algo/vil_threshold.h>
#include <vil/algo/vil_structuring_element.h>
#include <vil/vil_save.h>
#include <face3d/head_mesh.h>
#include <dlib/image_processing.h>

using namespace face3d;
using std::string;

media_coefficient_from_semantic_map_estimator::
media_coefficient_from_semantic_map_estimator(head_mesh mesh,
                                              vnl_matrix<double> const& subject_pca_components,
                                              vnl_matrix<double> const& expression_pca_components,
                                              vnl_matrix<double> const& subject_pca_ranges,
                                              vnl_matrix<double> const& expression_pca_ranges,
                                              bool debug_mode, std::string debug_dir) :
  base_mesh_(mesh),
  subject_pca_components_(subject_pca_components),
  expression_pca_components_(expression_pca_components),
  subject_pca_ranges_(subject_pca_ranges),
  expression_pca_ranges_(expression_pca_ranges),
  debug_mode_(debug_mode), debug_dir_(debug_dir)
{
  mean_face_mesh_ = mesh.face_mesh();
  mean_face_mesh_tree_.init(mean_face_mesh_.V(), mean_face_mesh_.F());
}
