#include "camera_estimation.h"

#include <vpgl/vpgl_calibration_matrix.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vgl/vgl_vector_3d.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vpgl/algo/vpgl_optimize_camera.h>
#include <vpgl/algo/vpgl_camera_compute.h>


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
#if 0
  vpgl_perspective_camera<double> unconstrained_cam;
  double err;
  if (!vpgl_perspective_camera_compute::compute_dlt(pts2d, pts3d, unconstrained_cam, err)) {
    std::cerr << "vpgl_perspective_camera_compute::compute_dlt failed." << std::endl;
    return false;
  }
  vpgl_calibration_matrix<double> uK = unconstrained_cam.get_calibration();
  uK.set_skew(0.0);
  vpgl_perspective_camera<double> constrained_cam = unconstrained_cam;
  constrained_cam.set_calibration(uK);
  constrained_cam = vpgl_optimize_camera::opt_orient_pos_cal(constrained_cam, hpts3d, pts2d);
  cam_params.from_camera(constrained_cam, nx, ny);

  cam_params.to_camera(constrained_cam);
  vpgl_calibration_matrix<double> K = constrained_cam.get_calibration();

  constrained_cam = vpgl_optimize_camera::opt_orient_pos(constrained_cam, hpts3d, pts2d);

  // extract the new camera. K should be unchanged.
  cam_params.from_camera(constrained_cam, nx, ny);
#else
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
#endif

  return true;
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
