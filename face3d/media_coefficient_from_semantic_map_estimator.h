#pragma once
#include <map>
#include <vector>
#include <string>
#include <utility>

#include <dlib/array2d.h>
#include <dlib/pixel.h>

#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_point_2d.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_matrix_inverse.h>

#include "semantic_map.h"
#include "head_mesh.h"
#include "camera_estimation.h"
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d_basic/affine_camera_approximator.h>
#include <face3d_basic/io_utils.h>

#include <vnl/vnl_vector.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_least_squares_function.h>

#include <igl/AABB.h>

#include "estimation_results.h"

namespace face3d {


class media_coefficient_from_semantic_map_estimator
{
public:
  media_coefficient_from_semantic_map_estimator(head_mesh mesh,
                                                vnl_matrix<double> const& subject_pca_components,
                                                vnl_matrix<double> const& expression_pca_components,
                                                vnl_matrix<double> const& subject_pca_ranges,
                                                vnl_matrix<double> const& expression_pca_ranges,
                                                bool debug_mode, std::string debug_dir,
                                                double fixed_focal_len=-1.0);

  template<class T>
    estimation_results_t estimate_coefficients(std::vector<std::string> const& img_ids,
                               std::vector<dlib::array2d<vgl_point_3d<float> > > const& semantic_maps,
                               subject_sighting_coefficients<T> &results);

private:

  triangle_mesh mean_face_mesh_;
  igl::AABB<triangle_mesh::VTYPE, 3> mean_face_mesh_tree_;
  head_mesh base_mesh_;
  const vnl_matrix<double> subject_pca_components_;
  const vnl_matrix<double> expression_pca_components_;
  const vnl_matrix<double> subject_pca_ranges_;
  const vnl_matrix<double> expression_pca_ranges_;
  bool debug_mode_;
  std::string debug_dir_;
  double fixed_focal_len_;
};


template<class T>
estimation_results_t media_coefficient_from_semantic_map_estimator::
estimate_coefficients(std::vector<std::string> const& img_ids,
                      std::vector<dlib::array2d<vgl_point_3d<float> > > const& semantic_maps,
                      subject_sighting_coefficients<T> &results)
{
  const int num_subject_components = subject_pca_components_.rows();
  const int num_expression_components = expression_pca_components_.rows();

  std::vector<T> all_cam_params;

  const int num_images = semantic_maps.size();
  int total_num_data = 0;
  std::vector<vnl_matrix<double> > As;
  std::vector<vnl_matrix<double> > Bs;
  std::vector<vnl_vector<double> > offsets;
  std::vector<int> valid_images;

  // keep track of all used mesh vertices. This vector will remain empty if not in debug mode.
  std::vector<std::vector<int> > vertex_indices_dbg(num_images);
  // for each image
  estimation_results_t retval(num_images);
  for (int n=0; n<num_images; ++n) {
    std::map<int, vgl_point_2d<double> > vertex_projection_map;
    extract_vertex_projections(semantic_maps[n], mean_face_mesh_, vertex_projection_map, mean_face_mesh_tree_);
    retval.vertices_found_[n] = vertex_projection_map.size();
    std::vector<vgl_point_2d<double> > vertex_projections;
    std::vector<int> vertex_indices;
    std::vector<vgl_point_3d<double> > mesh_vertices;
    for (auto vp : vertex_projection_map) {
      vertex_projections.push_back(vp.second);
      vertex_indices.push_back(vp.first);
      mesh_vertices.push_back(mean_face_mesh_.vertex(vp.first));
    }
    if (debug_mode_) {
      std::string vert_projections_fname = debug_dir_ + "/vertex_projections_" + std::to_string(n) + ".txt";
      face3d::io_utils::write_points(vertex_projections, vert_projections_fname);
      std::string mesh_vertices_fname = debug_dir_ + "/mesh_vertices_" + std::to_string(n) + ".txt";
      face3d::io_utils::write_points(mesh_vertices, mesh_vertices_fname);
    }
    const int min_corrs = 10;
    if (vertex_projections.size() < min_corrs) {
      std::cerr << "ERROR: found " << vertex_projections.size() << " correspondences, need " << min_corrs << std::endl;
      return retval;
    }
    // compute camera parameters for this image
    T cam_params;
    const int nx = semantic_maps[n].nc();
    const int ny = semantic_maps[n].nr();
    if (fixed_focal_len_ > 0) {
      vgl_point_2d<double> principal_pt(static_cast<double>(nx)/2, static_cast<double>(ny)/2);
      camera_estimation::compute_camera_params(vertex_projections, mesh_vertices, nx, ny, fixed_focal_len_, principal_pt, cam_params);
    }
    else{
      camera_estimation::compute_camera_params(vertex_projections, mesh_vertices, nx, ny, cam_params);
    }
    all_cam_params.push_back(cam_params);

    const int num_data = mesh_vertices.size();
    total_num_data += num_data;

    vnl_matrix<double> A(2*num_data, num_subject_components, 0.0);
    vnl_matrix<double> B(2*num_data, num_expression_components, 0.0);
    vnl_vector<double> img_offsets(num_data*2);

    auto pcam = cam_params.to_camera();
    affine_camera_approximator<double> local_affine(pcam);

    vnl_matrix<double> subject_pca_components_transpose = subject_pca_components_.transpose().get_n_columns(0,num_subject_components);
    vnl_matrix<double> expression_pca_components_transpose = expression_pca_components_.transpose().get_n_columns(0,num_expression_components);
    for (int i=0; i<num_data; ++i) {
      int vert_idx = vertex_indices[i];
      const vpgl_affine_camera<double> aff_cam = local_affine(vgl_homg_point_3d<double>(mesh_vertices[i]));
      const vnl_matrix<double> Pfull = aff_cam.get_matrix();
      const vnl_matrix<double> Psub = Pfull.get_n_rows(0,2).get_n_columns(0,3);
      vnl_matrix<double> ai = Psub * subject_pca_components_transpose.get_n_rows(3*vert_idx,3);
      if ((ai.rows() != 2) || (ai.cols() != num_subject_components)) {
        throw std::runtime_error("got unexpected matrix size for ai");
      }
      A.set_row(i*2, ai.get_row(0));
      A.set_row(i*2+1, ai.get_row(1));

      vnl_matrix<double> bi = Psub * expression_pca_components_transpose.get_n_rows(3*vert_idx,3);
      if ((bi.rows() != 2) || (bi.cols() != num_expression_components)) {
        throw std::runtime_error("got unexpected matrix size for ai");
      }
      B.set_row(i*2, bi.get_row(0));
      B.set_row(i*2+1, bi.get_row(1));

      // project base mesh vertex in to image with estimated camera
      vgl_point_2d<double> base_vertex_projection = pcam.project(mean_face_mesh_.vertex(vertex_indices[i]));
      vgl_vector_2d<double> offset = vertex_projections[i] - base_vertex_projection;
      img_offsets[i*2+0] = offset.x();
      img_offsets[i*2+1] = offset.y();
    }
    As.push_back(A);
    Bs.push_back(B);
    offsets.push_back(img_offsets);
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
  vnl_matrix<double> D(2*total_num_data, num_vars, 0.0);
  vnl_vector<double> coeffs_raw(0);
  if (D.columns()> 0) {
    vnl_vector<double> all_offsets(2*total_num_data,0.0);
    int row_offset = 0;
    for (int v=0; v<num_valid_images; ++v) {
      int num_rows = offsets[v].size();
      D.update(As[v], row_offset, 0);
      int b_col = num_subject_components + v*num_expression_components;
      D.update(Bs[v], row_offset, b_col);
      all_offsets.update(offsets[v], row_offset);
      row_offset += num_rows;
    }
    // add regularization terms
    const int num_regularization_terms = num_vars;
    vnl_matrix<double> reg_matrix(num_vars, num_vars);
    reg_matrix.fill(0.0);
    const double subj_reg_weight = 10.0; // 1e-1;
    const double expr_reg_weight = 1.0; // 1e-6;
    for (int i=0; i<num_subject_components; ++i) {
      reg_matrix(i,i) = subj_reg_weight/subject_pca_ranges_(i,1); // multiplier for subject coeffs
    }
    for (int i=num_subject_components; i<num_vars; ++i) {
      const int expr_coeff_idx = (i-num_subject_components) % num_expression_components;
      // It would be best to have different multiplier for pos and negative values,
      // since cost may not be symmetric, but awkward to express in SVD formulation
      // Just use the maximum value to get the weight instead.
      reg_matrix(i,i) = expr_reg_weight/expression_pca_ranges_(expr_coeff_idx,1);
    }
    vnl_matrix<double> D_transpose = D.transpose();
    // solve regularized SVD using Tikhonov Regularization
    coeffs_raw = vnl_matrix_inverse<double>(D_transpose*D + reg_matrix.transpose()*reg_matrix)*D_transpose*all_offsets;
  }
  else {
    std::cerr << "Error: D matrix has " << D.cols() << " columns." << std::endl;
    return retval;
  }
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
  results = face3d::subject_sighting_coefficients<T>(subject_coeffs,
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

}
