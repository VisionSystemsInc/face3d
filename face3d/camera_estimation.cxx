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


using namespace face3d;

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
  vpgl_affine_camera_compute::compute(pts2d, pts3d, affine_cam);

  ortho_camera_parameters<double> ortho_cam;
  ortho_cam.from_camera(affine_cam, nx, ny);

  vgl_rotation_3d<double> R_init(ortho_cam.rotation());
  double D_init = focal_len / ortho_cam.scale();
  vgl_vector_3d<double> T_init(0,0,D_init);
  std::cout << "D_init = " << D_init << std::endl;
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
compute_camera_params_bundle_adjust(std::vector<dlib::array2d<vgl_point_3d<float>>> const& PNCCs,
                                    face3d::triangle_mesh const& base_mesh,
                                    std::vector<perspective_camera_parameters<double>> &cameras_out,
                                    std::vector<vgl_point_3d<double>> &points_out
                                   )
{
  const int num_images = PNCCs.size();
  // extract vertex locations from PNCC images
  face3d::vertex_localizer localize_vertices(base_mesh);
  std::vector<std::map<int, vgl_point_2d<double>>> vertex_img_locs;
  for (auto const& PNCC : PNCCs) {
    vertex_img_locs.push_back(localize_vertices(PNCC));
  }
  // compute initial estimates for cameras independently
  std::vector<vpgl_perspective_camera<double>> cameras;
  for (int i=0; i<num_images; ++i) {
    std::vector<vgl_point_2d<double>> pts2d;
    std::vector<vgl_point_3d<double>> pts3d;
    for (auto const& item : vertex_img_locs[i]) {
      pts3d.push_back(base_mesh.vertex(item.first));
      pts2d.push_back(item.second);
    }
    face3d::perspective_camera_parameters<double> result;
    double focal_len = 2000.;
    vgl_point_2d<double> pp(PNCCs[i].nc()/2.0,
                            PNCCs[i].nr()/2.0);

    if (!compute_camera_params(pts2d, pts3d,
                               PNCCs[i].nc(), PNCCs[i].nr(),
                               focal_len, pp, result)) {
      throw std::runtime_error("Error computing initial camera parameters");
    }
    vpgl_perspective_camera<double> cam = result.to_camera();
    cameras.push_back(cam);
    std::cout << "camera " << i << ": " << cam << std::endl;
  }
  // remove points not seen in any image
  const int min_corrs_per_pt = std::max(2, num_images/2);
  std::vector<int> ba_idx_to_vertex;
  std::vector<vgl_point_3d<double>> world_pts;
  for (int v=0; v<base_mesh.num_vertices(); ++v) {
    int num_correspondences = 0;
    for (int i=0; i<num_images; ++i) {
      if (vertex_img_locs[i].find(v) != vertex_img_locs[i].end()) {
        ++num_correspondences;
      }
    }
    if (num_correspondences >= min_corrs_per_pt) {
      ba_idx_to_vertex.push_back(v);
      world_pts.push_back(base_mesh.vertex(v));
    }
  }
  const int num_ba_points = ba_idx_to_vertex.size();
  //world_pts = std::vector<vgl_point_3d<double>>(world_pts.begin(), world_pts.begin()+num_ba_points);

  std::cout << num_ba_points << " out of " << base_mesh.num_vertices() << " points being used for bundle adjustment." << std::endl;

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
        image_points[k] = vertex_img_locs[i][vidx];
      }
    }
  }

  std::vector<vgl_point_3d<double>> world_pts_init(world_pts);
  vpgl_bundle_adjust ba;
  ba.set_normalize_data(true);
  ba.set_self_calibrate(true);
  ba.set_verbose(true);
  //ba.set_x_tolerence(1e-4);
  ba.set_max_iterations(100);
  //ba.set_epsilon(1.0);
  ba.set_use_m_estimator(false);
  //ba.set_m_estimator_scale(1.0);
  if (!ba.optimize(cameras, world_pts, image_points, point_mask)) {
    throw std::runtime_error("Bundle Adjustment returned error");
  }
  // cameras and world points are now optimized
  // compute similarity transform that best aligns 3D vertices with base mesh
  vgl_compute_similarity_3d<double> compute_sim(world_pts, world_pts_init);
  compute_sim.estimate();
  double scale = compute_sim.scale();
  vgl_rotation_3d<double> rot = compute_sim.rotation();
  vgl_vector_3d<double> translation = compute_sim.translation();
  std::cout << "similarity transform scale = " << scale << " rotation = " << rot << " translation = " << translation << std::endl;
  // apply computed similarity transform to points and cameras
  points_out.clear();
  for(auto pt : world_pts) {
    // turn points into vectors because no scalar multiply operator on points
    vgl_vector_3d<double> ptv(pt.x(), pt.y(), pt.z());
    vgl_vector_3d<double> ptv_new = scale*(rot*ptv) + translation;
    points_out.emplace_back(ptv_new.x(), ptv_new.y(), ptv_new.z());
  }
  vgl_rotation_3d<double> inv_rot = rot.inverse();
  // transform cameras
  cameras_out.clear();
  for (int i=0; i<num_images; ++i) {
    perspective_camera_parameters<double> params;
    params.from_camera(cameras[i], PNCCs[i].nc(), PNCCs[i].nr());

    vgl_rotation_3d<double> Rnew = params.rotation() * inv_rot;
    vgl_vector_3d<double> Tnew = Rnew * -translation + scale * params.translation();
    params.set_rotation(Rnew);
    params.set_translation(Tnew);
    cameras_out.push_back(params);
  }
  return;
}
