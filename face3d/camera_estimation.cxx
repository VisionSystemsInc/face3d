#include <algorithm>

#include "camera_estimation.h"
#include "semantic_map.h"
#include "triangle_mesh.h"

#include <vnl/vnl_crs_index.h>
#include <vpgl/vpgl_calibration_matrix.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vgl/vgl_vector_3d.h>
#include <vgl/algo/vgl_compute_similarity_3d.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vpgl/algo/vpgl_optimize_camera.h>
#include <vpgl/algo/vpgl_camera_compute.h>
#include <vpgl/algo/vpgl_bundle_adjust.h>

#include <Eigen/Dense>

using namespace face3d;

void compute_similarity_RANSAC(std::vector<vgl_point_3d<double>> const& pts1,
                               std::vector<vgl_point_3d<double>> const& pts2,
                               double &scale, vgl_rotation_3d<double> &rot,
                               vgl_vector_3d<double> &translation)
{
  const int num_random_pts = 4;
  const int num_trials = 200;
  const double inlier_thresh = 1.0;  // mm
  const int num_pts = pts1.size();

  std::vector<int> seed_indices(num_pts);
  for (int i=0; i<num_pts; ++i) {
    seed_indices[i] = i;
  }

  std::vector<int> best_inliers;
  for (int t=0; t<num_trials; ++t) {
    std::random_shuffle(seed_indices.begin(), seed_indices.end());
    std::vector<vgl_point_3d<double>> pts1_subset(num_random_pts);
    std::vector<vgl_point_3d<double>> pts2_subset(num_random_pts);
    for (int i=0; i<num_random_pts; ++i) {
      pts1_subset[i] = pts1[seed_indices[i]];
      pts2_subset[i] = pts2[seed_indices[i]];
    }
    vgl_compute_similarity_3d<double> compute_sim(pts1_subset, pts2_subset);
    compute_sim.estimate();
    double this_scale = compute_sim.scale();
    vgl_rotation_3d<double> this_rot = compute_sim.rotation();
    vgl_vector_3d<double> this_translation = compute_sim.translation();

    // apply transform to points and tally inliers
    std::vector<int> inliers;
    for (int i=0; i<num_pts; ++i) {
      vgl_vector_3d<double> pt1v(pts1[i].x(), pts1[i].y(), pts1[i].z());
      pt1v = this_scale*(this_rot*pt1v) + this_translation;
      vgl_point_3d<double> pt1_transformed(pt1v.x(), pt1v.y(), pt1v.z());
      double dist = (pt1_transformed - pts2[i]).length();
      if (dist <= inlier_thresh) {
        inliers.push_back(i);
      }
    }
    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
    }
  }
  // recompute transform based on all inliers
  const int num_inliers = best_inliers.size();
  std::cout << "Computing final similarity using " << num_inliers << " inliers out of " << num_pts << std::endl;
  std::vector<vgl_point_3d<double>> pts1_subset;
  std::vector<vgl_point_3d<double>> pts2_subset;
  for (int i=0; i<num_inliers; ++i) {
    pts1_subset.push_back(pts1[best_inliers[i]]);
    pts2_subset.push_back(pts2[best_inliers[i]]);
  }
  vgl_compute_similarity_3d<double> compute_sim(pts1_subset, pts2_subset);
  compute_sim.estimate();
  scale = compute_sim.scale();
  rot = compute_sim.rotation();
  translation = compute_sim.translation();
}

void estimate_coefficients(vnl_matrix<double> const& subj_components,
                           vnl_matrix<double> const& expr_components,
                           std::map<int,vgl_vector_3d<double>> vert_offsets,
                           vnl_vector<double> &subj_coeffs,
                           vnl_vector<double> &expr_coeffs)
{
  int num_constraints = vert_offsets.size();
  int num_subj_coeffs = subj_components.rows();
  int num_expr_coeffs = expr_components.rows();
  int num_coeffs = num_subj_coeffs + num_expr_coeffs;

  Eigen::MatrixXd A(num_constraints*3, num_coeffs);
  A.fill(0.0);
  Eigen::VectorXd  b(num_constraints*3);
  b.fill(0.0);
  int r=0;
  for (auto const& constraint : vert_offsets) {
    const int vert_idx = constraint.first;
    const vgl_vector_3d<double> vert_offset = constraint.second;
    for (int c=0; c<num_subj_coeffs; ++c) {
      for (int d=0; d<3; ++d) {
        A(r+d, c) = subj_components(c, vert_idx*3 + d);
      }
    }
    for (int c=0; c<num_expr_coeffs; ++c) {
      for (int d=0; d<3; ++d) {
        A(r+d, num_subj_coeffs + c) = expr_components(c, vert_idx*3 + d);
      }
    }
    b(r+0) = vert_offset.x();
    b(r+1) = vert_offset.y();
    b(r+2) = vert_offset.z();
    r += 3;
  }
  Eigen::MatrixXd reg_matrix(num_coeffs, num_coeffs);
  reg_matrix.fill(0.0);
  const double subj_reg_weight = 1e-3;
  const double expr_reg_weight = 1e-3;
  for (int i = 0;  i < num_subj_coeffs; i++) {
    reg_matrix(i, i) = subj_reg_weight;
  }
  for (int i = num_subj_coeffs;  i < num_coeffs; i++) {
    reg_matrix(i, i) = expr_reg_weight;
  }
  Eigen::MatrixXd A_transpose = A.transpose();
  Eigen::MatrixXd A_inverse;

  // solve regularized SVD using Tikhonov Regularization
  A_inverse = (A_transpose*A + reg_matrix.transpose()*reg_matrix).inverse()*A_transpose;
  Eigen::VectorXd coeffs_raw = A_inverse*b;

  subj_coeffs.set_size(num_subj_coeffs);
  for (int i=0; i<num_subj_coeffs; ++i) {
    subj_coeffs[i] = coeffs_raw[i];
  }
  expr_coeffs.set_size(num_expr_coeffs);
  for (int i=0; i<num_expr_coeffs; ++i) {
      expr_coeffs[i] = coeffs_raw[num_subj_coeffs + i];
  }
  return;
}

bool face3d::camera_estimation::
compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                      std::vector<vgl_point_3d<double> > const& pts3d,
                      int nx, int ny,
                      ortho_camera_parameters<double> &cam_params)
{

  // estimate the affine camera matrix
  vpgl_affine_camera<double> unconstrained_cam;
  vpgl_affine_camera_compute::compute(pts2d, pts3d, unconstrained_cam);

  cam_params.from_camera(unconstrained_cam, nx, ny);
  return true;
}

bool face3d::camera_estimation::
compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                      std::vector<vgl_point_3d<double> > const& pts3d,
                      int nx, int ny,
                      perspective_camera_parameters<double> &cam_params)
{
  std::vector<vgl_homg_point_3d<double> > hpts3d;
  for (auto p : pts3d) {
    hpts3d.push_back(vgl_homg_point_3d<double>(p));
  }
  vpgl_affine_camera<double> affine_cam;
  vpgl_affine_camera_compute::compute(pts2d, pts3d, affine_cam);

  ortho_camera_parameters<double> ortho_cam;
  ortho_cam.from_camera(affine_cam, nx, ny);

  vgl_point_2d<double> pp_init(static_cast<double>(nx)/2, static_cast<double>(ny)/2);
  vgl_rotation_3d<double> R_init(ortho_cam.rotation());
  double D_init = 5000.0;  // distance from camera to face (5m)
  vgl_vector_3d<double> T_init(0,0,D_init);
  double f_init = ortho_cam.scale()*D_init;
  vpgl_calibration_matrix<double> K_init(f_init, pp_init);
  vpgl_perspective_camera<double> cam_init(K_init, R_init, T_init);
  const double xtol = 1e-3;
  vpgl_perspective_camera<double> cam = vpgl_optimize_camera::opt_orient_pos_f(cam_init, hpts3d, pts2d, xtol);
  cam_params.from_camera(cam, nx, ny);

  return true;
}

bool face3d::camera_estimation::
compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                      std::vector<vgl_point_3d<double> > const& pts3d,
                      int nx, int ny,
                      double focal_len,
                      vgl_point_2d<double> principal_pt,
                      perspective_camera_parameters<double> &cam_params)
{
  std::vector<vgl_homg_point_3d<double> > hpts3d;
  for (auto p : pts3d) {
    hpts3d.push_back(vgl_homg_point_3d<double>(p));
  }
  vpgl_affine_camera<double> affine_cam;
  std::cout << "computing affine camera first" << std::endl;
  vpgl_affine_camera_compute::compute(pts2d, pts3d, affine_cam);
  std::cout << "done" << std::endl;

  ortho_camera_parameters<double> ortho_cam;
  ortho_cam.from_camera(affine_cam, nx, ny);

  vgl_rotation_3d<double> R_init(ortho_cam.rotation());
  double D_init = focal_len / ortho_cam.scale();
  vgl_vector_3d<double> T_init(0,0,D_init);
  vpgl_calibration_matrix<double> K(focal_len, principal_pt);
  vpgl_perspective_camera<double> cam_init(K, R_init, T_init);
  vpgl_perspective_camera<double> cam = vpgl_optimize_camera::opt_orient_pos(cam_init, hpts3d, pts2d);
  cam_params.from_camera(cam, nx, ny);

  return true;
}

bool face3d::camera_estimation::
compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                      std::vector<vgl_point_3d<double> > const& pts3d,
                      int nx, int ny,
                      double focal_len,
                      vgl_point_2d<double> principal_pt,
                      ortho_camera_parameters<double> &cam_params)

{
  throw std::runtime_error("Orthographic cameras do not have a finite focal length");
}

bool face3d::camera_estimation::compute_camera_params_calibrated::
operator()(std::vector<vgl_point_2d<double> > const& pts2d,
           std::vector<vgl_point_3d<double> > const& pts3d,
           int nx, int ny,
           perspective_camera_parameters<double> &cam_params)
{
  vpgl_perspective_camera<double> unconstrained_cam;
  double err;
  if (!vpgl_perspective_camera_compute::compute_dlt(pts2d, pts3d, unconstrained_cam, err)) {
    std::cerr << "vpgl_perspective_camera_compute::compute_dlt failed." << std::endl;
    return false;
  }
  vpgl_calibration_matrix<double> uK = unconstrained_cam.get_calibration();
  uK.set_skew(0.0);
  uK.set_principal_point(principal_pt_);
  uK.set_focal_length(focal_len_);
  uK.set_x_scale(1.0);
  uK.set_y_scale(1.0);
  vpgl_perspective_camera<double> constrained_cam = unconstrained_cam;
  constrained_cam.set_calibration(uK);
  std::vector<vgl_homg_point_3d<double> > hpts3d;
  for (auto p : pts3d) {
    hpts3d.push_back(vgl_homg_point_3d<double>(p));
  }
  vpgl_calibration_matrix<double> K = constrained_cam.get_calibration();


  constrained_cam = vpgl_optimize_camera::opt_orient_pos(constrained_cam, hpts3d, pts2d);
  //constrained_cam = vpgl_optimize_camera::opt_orient_pos_cal(constrained_cam, hpts3d, pts2d);

  // extract the new camera. K should be unchanged.
  cam_params.from_camera(constrained_cam, nx, ny);
  std::cout << "calibrated: focal_len = " << cam_params.focal_len() << std::endl;
  return true;
}

void face3d::camera_estimation::
compute_camera_params_bundle_adjust(std::vector<std::map<int, vgl_point_2d<double>>> const& vertex_img_locs,
                                    vnl_matrix<double> const& subject_pca_components,
                                    vnl_matrix<double> const& expression_pca_components,
                                    face3d::head_mesh &base_mesh,
                                    std::vector<face3d::perspective_camera_parameters<double>> const& init_cameras,
                                    std::vector<vgl_point_3d<double>> &points_out,
                                    face3d::subject_sighting_coefficients<face3d::perspective_camera_parameters<double>> &coeffs_out)
{
  std::vector<face3d::perspective_camera_parameters<double>> cameras_out(init_cameras);

  face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T>  mean_face_mesh(base_mesh.face_mesh());
  const int num_images = vertex_img_locs.size();
  if (init_cameras.size() != num_images) {
    throw std::runtime_error("incorrect number of initial cameras (intrinsics)");
  }
  // compute initial estimates for cameras independently
  std::vector<vpgl_perspective_camera<double>> cameras;
  for (int i=0; i<num_images; ++i) {
    std::vector<vgl_point_2d<double>> pts2d;
    std::vector<vgl_point_3d<double>> pts3d;
    for (auto const& item : vertex_img_locs[i]) {
      pts3d.push_back(mean_face_mesh.vertex(item.first));
      pts2d.push_back(item.second);
    }
    face3d::perspective_camera_parameters<double> result;
    double focal_len = init_cameras[i].focal_len();
    int nx = init_cameras[i].nx();
    int ny = init_cameras[i].ny();
    vgl_point_2d<double> pp(init_cameras[i].principal_point());

    std::cout << "computing init camera for image " << i << std::endl;
    if (!compute_camera_params(pts2d, pts3d,
                               nx, ny,
                               focal_len, pp, result)) {
      throw std::runtime_error("Error computing initial camera parameters");
    }
    vpgl_perspective_camera<double> cam = result.to_camera();
    cameras.push_back(cam);
    std::cout << "camera " << i << ": " << cam << std::endl;
  }
  // remove points not seen in any image
  const int min_corrs_per_pt = 2;
  std::vector<int> ba_idx_to_vertex;
  std::vector<vgl_point_3d<double>> world_pts;
  for (int v=0; v<mean_face_mesh.num_vertices(); ++v) {
    int num_correspondences = 0;
    for (int i=0; i<num_images; ++i) {
      if (vertex_img_locs[i].find(v) != vertex_img_locs[i].end()) {
        ++num_correspondences;
      }
    }
    if (num_correspondences >= min_corrs_per_pt) {
      ba_idx_to_vertex.push_back(v);
      world_pts.push_back(mean_face_mesh.vertex(v));
    }
  }
  const int num_ba_points = ba_idx_to_vertex.size();
  // save a copy of world_pts before being transformed by bundle adjustment
  const std::vector<vgl_point_3d<double>> mean_face_pts(world_pts.begin(), world_pts.end());

  std::cout << num_ba_points << " out of " << mean_face_mesh.num_vertices() << " points being used for bundle adjustment." << std::endl;

  // perform bundle adjustment
  std::vector<std::vector<bool>> point_mask(num_images, std::vector<bool>(num_ba_points, true));
  for (int i=0; i<num_images; ++i) {
    for (int p=0; p<num_ba_points; ++p) {
      int vidx = ba_idx_to_vertex[p];
      auto img_loc_it = vertex_img_locs[i].find(vidx);
      if (img_loc_it == vertex_img_locs[i].end()) {
        point_mask[i][p] = false;
      }
    }
  }
  vnl_crs_index crs(point_mask);
  std::vector<vgl_point_2d<double>> image_points(crs.num_non_zero());
  std::cout << "num_images = " << num_images << " num_ba_points = " << num_ba_points << " crs.num_rows = " << crs.num_rows() << " crs.num_cols = " << crs.num_cols() << std::endl;
  for (int i=0; i<crs.num_rows(); ++i) {
    for (int j=0; j<crs.num_cols(); ++j) {
      int k = crs(i,j);
      if (k >= 0) {
        int vidx = ba_idx_to_vertex[j];
        image_points[k] = vertex_img_locs[i].at(vidx);
      }
    }
  }

  std::cout << "starting optimization.." << std::endl;
  vpgl_bundle_adjust ba;
  ba.set_normalize_data(true);
  ba.set_self_calibrate(false);
  ba.set_verbose(true);
  //ba.set_x_tolerence(1e-4);
  ba.set_max_iterations(100);
  //ba.set_epsilon(1.0);
  ba.set_use_m_estimator(false);
  //ba.set_m_estimator_scale(100.0);
  if (!ba.optimize(cameras, world_pts, image_points, point_mask)) {
    throw std::runtime_error("Bundle Adjustment returned error");
  }
  // cameras and world points are now optimized
  const int num_subj_coeffs = subject_pca_components.rows();
  const int num_expr_coeffs = expression_pca_components.rows();
  vnl_vector<double> subj_coeffs(num_subj_coeffs, 0.0);
  vnl_vector<double> expr_coeffs(num_expr_coeffs, 0.0);
  const int NUM_ITS = 4;
  for (int it=0; it<NUM_ITS; ++it) {
    face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T> const& mesh = base_mesh.face_mesh();

    std::vector<vgl_point_3d<double>> face_verts_all;
    mesh.get_vertices(face_verts_all);
    std::vector<vgl_point_3d<double>> face_verts;
    for (int vidx : ba_idx_to_vertex) {
      face_verts.push_back(face_verts_all[vidx]);
    }
    // compute similarity transform that best aligns 3D vertices with current mesh
    double scale;
    vgl_rotation_3d<double> rot;
    vgl_vector_3d<double> translation;
    compute_similarity_RANSAC(world_pts, face_verts, scale, rot, translation);
    std::cout << "similarity transform scale = " << scale << " rotation = " << rot << " translation = " << translation << std::endl;
    // apply computed similarity transform to points and cameras
    for (auto &pt : world_pts){
      vgl_vector_3d<double> ptv(pt.x(), pt.y(), pt.z());
      ptv = scale*(rot*ptv) + translation;
      pt = vgl_point_3d<double>(ptv.x(), ptv.y(), ptv.z());
    }
    // transform cameras
    vgl_rotation_3d<double> inv_rot = rot.inverse();
    for (int i=0; i<num_images; ++i) {
      perspective_camera_parameters<double> params;
      int nx = init_cameras[i].nx();
      int ny = init_cameras[i].ny();
      params.from_camera(cameras[i], nx, ny);
      vgl_rotation_3d<double> Rnew = params.rotation() * inv_rot;
      vgl_vector_3d<double> Tnew = Rnew * -translation + scale * params.translation();
      params.set_rotation(Rnew);
      params.set_translation(Tnew);
      cameras_out[i] = params;
    }
    // estimate new coefficients based on transformed points
    const double VERT_OFFSET_THRESH = 10.0;  // mm
    std::map<int, vgl_vector_3d<double>> vert_offsets;
    for (int ba_idx=0; ba_idx < ba_idx_to_vertex.size(); ++ba_idx) {
      int vidx = ba_idx_to_vertex[ba_idx];
      vgl_vector_3d<double> vert_offset = world_pts[ba_idx] - mean_face_pts[ba_idx];
      if (vert_offsets[vidx].length() > VERT_OFFSET_THRESH) {
        std::cout << "BIG VERT OFFSET: " << vidx << " = " << vert_offsets[vidx] << std::endl;
        continue;
      }
      vert_offsets[vidx] = vert_offset;
    }
    std::cout << "estimating new coefficients.." << std::endl;
    estimate_coefficients(subject_pca_components, expression_pca_components,
                          vert_offsets, subj_coeffs, expr_coeffs);
    std::cout << "estimated coeffs:\n subj = " << subj_coeffs << "\n expr = " << expr_coeffs << std::endl;
    base_mesh.apply_coefficients(subject_pca_components, expression_pca_components,
                                 subj_coeffs, expr_coeffs);

  }
  points_out = world_pts;
  std::vector<std::string> img_labels;
  std::vector<vnl_vector<double>> expr_coeff_list;
  for (int i=0; i<num_images; ++i) {
    img_labels.push_back("unknown");
    expr_coeff_list.push_back(expr_coeffs);
  }
  coeffs_out = face3d::subject_sighting_coefficients<face3d::perspective_camera_parameters<double>>(subj_coeffs, img_labels, expr_coeff_list, cameras_out);
  return;
}

void face3d::camera_estimation::
compute_camera_params_bundle_adjust(std::vector<dlib::array2d<vgl_point_3d<float>>> const& PNCCs,
                                    vnl_matrix<double> const& subject_pca_components,
                                    vnl_matrix<double> const& expression_pca_components,
                                    face3d::head_mesh &base_mesh,
                                    std::vector<perspective_camera_parameters<double>> const& init_cameras,
                                    std::vector<vgl_point_3d<double>> &points_out,
                                    face3d::subject_sighting_coefficients<face3d::perspective_camera_parameters<double>> &coeffs_out
                                   )
{
  const int num_images = PNCCs.size();
  // extract vertex locations from PNCC images
  face3d::vertex_localizer localize_vertices(base_mesh.face_mesh());
  std::vector<std::map<int, vgl_point_2d<double>>> vertex_img_locs;
  for (auto const& PNCC : PNCCs) {
    vertex_img_locs.push_back(localize_vertices(PNCC));
  }
  compute_camera_params_bundle_adjust(vertex_img_locs, subject_pca_components, expression_pca_components, base_mesh, init_cameras, points_out, coeffs_out);
}
