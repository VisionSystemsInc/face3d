#pragma once
#include <string>
#include <vector>
#include <cmath>

#include "head_mesh.h"
#include "semantic_map.h"
#include "camera_estimation.h"

#include <vnl/vnl_matrix.h>

#include <face3d_basic/io_utils.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <igl/AABB.h>
#include "estimation_results.h"

#include <Eigen/Dense>

namespace face3d {

class media_coefficient_from_PNCC_and_offset_estimator
{
public:
  media_coefficient_from_PNCC_and_offset_estimator(head_mesh const& mesh,
                                                   vnl_matrix<double> const& subject_pca_components,
                                                   vnl_matrix<double> const& expression_pca_components,
                                                   vnl_matrix<double> const& subject_pca_ranges,
                                                   vnl_matrix<double> const& expression_pca_ranges,
                                                   bool debug_mode, std::string debug_dir);

  template<class T>
    estimation_results_t estimate_coefficients(std::vector<std::string> const& img_ids,
                               std::vector<dlib::array2d<vgl_point_3d<float> > > const& semantic_maps,
                               std::vector<dlib::array2d<vgl_vector_3d<float> > > const& offsets,
                               subject_sighting_coefficients<T> &results);

private:
  head_mesh base_mesh_;
  vnl_matrix<double> subject_pca_components_;
  vnl_matrix<double> expression_pca_components_;
  vnl_matrix<double> subject_pca_ranges_;
  vnl_matrix<double> expression_pca_ranges_;

  triangle_mesh mean_face_mesh_;
  igl::AABB<triangle_mesh::VTYPE, 3> mean_face_mesh_tree_;

  bool debug_mode_;
  std::string debug_dir_;
};
}

template<class CAM_T>
face3d::estimation_results_t face3d::media_coefficient_from_PNCC_and_offset_estimator::
estimate_coefficients(std::vector<std::string> const& img_ids,
                      std::vector<dlib::array2d<vgl_point_3d<float> > > const& semantic_maps,
                      std::vector<dlib::array2d<vgl_vector_3d<float> > > const& offsets,
                      subject_sighting_coefficients<CAM_T> &results)
{
  const int num_subject_components = subject_pca_components_.rows();
  const int num_expression_components = expression_pca_components_.rows();

  std::vector<CAM_T> all_cam_params;

  const int num_images = semantic_maps.size();
  int total_num_data = 0;
  std::vector<Eigen::MatrixXd> As;
  std::vector<Eigen::MatrixXd> Bs;
  std::vector<Eigen::VectorXd> constraints;
  std::vector<int> valid_images;

  // keep track of all used mesh vertices. This vector will remain empty if not in debug mode.
  std::vector<std::vector<int> > vertex_indices_dbg(num_images);
  estimation_results_t retval(num_images);

  // for each image
  for (int n=0; n<num_images; ++n) {
    // get the image locations of the mean face vertices via the PNCC
    std::map<int, vgl_point_2d<double> > vertex_projection_map;
    extract_vertex_projections(semantic_maps[n], mean_face_mesh_, vertex_projection_map, mean_face_mesh_tree_);
    retval.vertices_found_[n] = vertex_projection_map.size();
    std::cout << "Found " << retval.vertices_found_[n] << " out of " << mean_face_mesh_.num_vertices() << " vertex projections." << std::endl;

    // fill in lists containing per-vertex information
    std::vector<vgl_point_2d<double> > vertex_projections;
    std::vector<int> vertex_indices;
    std::vector<vgl_vector_3d<double> > vertex_offsets;
    std::vector<vgl_point_3d<double> > vertex_locations;
    for (auto vp : vertex_projection_map) {
      vertex_projections.push_back(vp.second);
      vertex_indices.push_back(vp.first);
      vgl_point_3d<double> base_mesh_vert(mean_face_mesh_.vertex(vp.first));
      // TODO: interpolation
      int pix_r = static_cast<int>(std::round(vp.second.y()));
      int pix_c = static_cast<int>(std::round(vp.second.x()));
      vgl_vector_3d<float> offset_f = offsets[n][pix_r][pix_c];
      vgl_vector_3d<double> offset(offset_f.x(), offset_f.y(), offset_f.z());
      vertex_offsets.push_back(offset);
      vertex_locations.push_back(base_mesh_vert + offset);
    }

    if (debug_mode_) {
      std::string vert_projections_fname = debug_dir_ + "/vertex_projections_" + std::to_string(n) + ".txt";
      face3d::io_utils::write_points(vertex_projections, vert_projections_fname);
    }
    const int min_corrs = 10;
    if (vertex_projections.size() < min_corrs) {
      std::cerr << "ERROR: found " << vertex_projections.size() << " correspondences, need " << min_corrs << std::endl;
      return retval;
    }

    // compute camera parameters for this image
    CAM_T cam_params;
    const int nx = semantic_maps[n].nc();
    const int ny = semantic_maps[n].nr();
    camera_estimation::compute_camera_params(vertex_projections, vertex_locations, nx, ny, cam_params);
    all_cam_params.push_back(cam_params);

    const int num_data = vertex_projections.size();
    total_num_data += num_data;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3*num_data, num_subject_components);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3*num_data, num_expression_components);
    Eigen::VectorXd constraint(num_data*3);

    vnl_matrix<double> subject_pca_components_transpose = subject_pca_components_.get_n_rows(0,num_subject_components).transpose();
    vnl_matrix<double> expression_pca_components_transpose = expression_pca_components_.get_n_rows(0,num_expression_components).transpose();

    for (int i = 0; i < num_data; i++) {
      int vert_idx = vertex_indices[i];
      vnl_matrix<double> ai = subject_pca_components_transpose.get_n_rows(3*vert_idx,3);

      if ((ai.rows() != 3) || (ai.cols() != num_subject_components)) {
        throw std::runtime_error("got unexpected matrix size for ai");
      }
      Eigen::VectorXd ai_row = Eigen::Map<Eigen::VectorXd>(ai.get_row(0).data_block(), ai.cols());
      A.row(i*3) = Eigen::Map<Eigen::VectorXd>(ai.get_row(0).data_block(), ai.cols());
      A.row(i*3+1) = Eigen::Map<Eigen::VectorXd>(ai.get_row(1).data_block(), ai.cols());
      A.row(i*3+2) = Eigen::Map<Eigen::VectorXd>(ai.get_row(2).data_block(), ai.cols());

      vnl_matrix<double> bi = expression_pca_components_transpose.get_n_rows(3*vert_idx,3);
      if ((bi.rows() != 3) || (bi.cols() != num_expression_components)) {
        throw std::runtime_error("got unexpected matrix size for ai");
      }
      B.row(i*3) = Eigen::Map<Eigen::VectorXd> (bi.get_row(0).data_block(), bi.cols());
      B.row(i*3+1) = Eigen::Map<Eigen::VectorXd> (bi.get_row(1).data_block(), bi.cols());
      B.row(i*3+2) = Eigen::Map<Eigen::VectorXd> (bi.get_row(2).data_block(), bi.cols());

      vgl_vector_3d<double> offset = vertex_offsets[i];
      constraint(i*3+0) = offset.x();
      constraint(i*3+1) = offset.y();
      constraint(i*3+2) = offset.z();
    }
    As.push_back(A);
    Bs.push_back(B);
    constraints.push_back(constraint);
    valid_images.push_back(n);
    retval.image_success_[n] = true;

    if (debug_mode_) {
      vertex_indices_dbg[n] = std::move(vertex_indices);
    }
  }
  /**
    construct D = |A0  B0 0  0 .. |
                  |A1  0  B1 0 .. |
                  |A2  0  0  B2 ..|
                  | :             |

   **/

  const int num_valid_images = valid_images.size();
  const int num_vars = num_subject_components + num_valid_images*num_expression_components;

  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3*total_num_data, num_vars);
  Eigen::VectorXd all_constraints(3*total_num_data);
  if (D.cols() == 0) {
    std::cerr << "Error: D matrix has " << D.cols() << " columns." << std::endl;
    return retval;
  }

  int row_offset = 0;
  for (int v = 0; v < num_valid_images; v++) {
    int num_rows = As[v].rows();
    int num_cols_B = Bs[v].cols();
    int num_cols_A = As[v].cols();
    D.block(row_offset, 0, num_rows, num_cols_A) = As[v];
    int b_col = num_subject_components + v*num_expression_components;
    D.block(row_offset, b_col, num_rows, num_cols_B) = Bs[v];
    all_constraints.segment(row_offset, num_rows) = constraints[v];
    row_offset += num_rows;
  }
  // add regularization terms
  const int num_regularization_terms = num_vars;
  Eigen::MatrixXd reg_matrix(num_vars, num_vars);
  reg_matrix.fill(0.0);
  const double subj_reg_weight = 1e-1;
  const double expr_reg_weight = 1e-1;
  for (int i = 0;  i < num_subject_components; i++) {
    reg_matrix(i, i) = subj_reg_weight/subject_pca_ranges_(i,1);    // multiplier for subject coeffs
  }

  for (int i = num_subject_components; i < num_vars; i++) {
    const int expr_coeff_idx = (i-num_subject_components) % num_expression_components;
    // It would be best to have different multiplier for pos and negative values,
    // since cost may not be symmetric, but awkward to express in SVD formulation
    // Just use the maximum value to get the weight instead.
    reg_matrix(i, i) = expr_reg_weight/expression_pca_ranges_(expr_coeff_idx, 1);
  }

  Eigen::MatrixXd D_transpose = D.transpose();
  Eigen::MatrixXd D_inverse;

  // solve regularized SVD using Tikhonov Regularization
  D_inverse = (D_transpose*D + reg_matrix.transpose()*reg_matrix).inverse()*D_transpose;
  Eigen::VectorXd coeffs_raw = D_inverse*all_constraints;

  vnl_vector<double> subject_coeffs(num_subject_components, 0.0);

  for (int i=0; i<num_subject_components; ++i) {
    subject_coeffs[i] = coeffs_raw[i];
  }

  std::vector<vnl_vector<double> > expression_coeffs;
  for (int n=0; n<num_images; ++n) {
    vnl_vector<double> these_expression_coeffs(num_expression_components,0.0);
    for (int i=0; i<num_expression_components; ++i) {
      int raw_idx = num_subject_components + n*num_expression_components + i;
      these_expression_coeffs[i] = coeffs_raw[raw_idx];
    }
    expression_coeffs.push_back(these_expression_coeffs);
  }
  results = face3d::subject_sighting_coefficients<CAM_T>(subject_coeffs,
                                                     img_ids,
                                                     expression_coeffs,
                                                     all_cam_params);
  if (debug_mode_) {
    // write out projected optimized 3-d mesh coordinates
    for (int s=0; s<num_images; ++s) {
      std::vector<vgl_point_3d<double> >  verts;
      base_mesh_.apply_coefficients(subject_pca_components_, expression_pca_components_,
                                    results.subject_coeffs(), results.expression_coeffs(s));
      base_mesh_.face_mesh().get_vertices(verts);
      std::vector<vgl_point_2d<double> > projected_verts;
      auto cam = results.camera(s).to_camera();
      for (int vert_idx : vertex_indices_dbg[s]) {
        vgl_point_3d<double> v3d = verts[vert_idx];
        // project into image
        vgl_point_2d<double> v2d = cam.project(v3d);
        projected_verts.push_back(v2d);
      }
      std::string opt_verts_filename = debug_dir_ + "/projected_verts_final_" + std::to_string(s) + ".txt";
        face3d::io_utils::write_points(projected_verts, opt_verts_filename);
      }
  }
  retval.success_ = true;
  return retval;
}
