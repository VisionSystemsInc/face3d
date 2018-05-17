#include <face3d_basic/affine_camera_approximator.h>
#include <vpgl/vpgl_calibration_matrix.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vpgl/vpgl_affine_camera.h>
#include <vgl/vgl_homg_point_3d.h>
#include <vgl/vgl_homg_point_2d.h>

#include <iostream>


int main(int, char**)
{
  int nx = 1000;
  int ny = 500;
  double f = 200.0;
  vpgl_calibration_matrix<double> K(f, vgl_point_2d<double>(double(nx)/2, double(ny)/2));
  vgl_rotation_3d<double> R(vnl_vector_fixed<double,3>(0.03,0.05,0.5));
  vgl_vector_3d<double> T(0,0,-100);

  vpgl_perspective_camera<double> pcam(K,R,T);

  face3d::affine_camera_approximator<double> approx(pcam);

  vgl_point_3d<double> x(-35,12,1175.234);
  vgl_point_2d<double> u = pcam.project(x);
  vpgl_affine_camera<double> acam = approx(vgl_homg_point_3d<double>(x));
  //vpgl_affine_camera<double> acam = approx(vgl_homg_point_2d<double>(u));

  vgl_point_2d<double> projected_a0 = acam.project(x);
  vgl_point_2d<double> projected_p0 = pcam.project(x);
  std::cout << "projected_p0 = " << projected_p0 << std::endl;
  std::cout << "projected_a0 = " << projected_a0 << std::endl;

  vgl_vector_3d<double> dx(2,-2,0);
  vgl_point_2d<double> projected_a1 = acam.project(vgl_homg_point_3d<double>(x + dx));
  vgl_point_2d<double> projected_p1 = pcam.project(vgl_homg_point_3d<double>(x + dx));
  std::cout << "dx projected p = " << projected_p1 - projected_p0 << std::endl;
  std::cout << "dx projected a = " << projected_a1 - projected_a0 << std::endl;

  std::cout << "done." << std::endl;
  return 0;
}
