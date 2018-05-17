#ifndef perspective_camera_parameters_txx_included_
#define perspective_camera_parameters_txx_included_

#include "perspective_camera_parameters.h"
#include <vpgl/vpgl_perspective_camera.h>
#include <vpgl/vpgl_calibration_matrix.h>

namespace face3d {

template<class T>
perspective_camera_parameters<T>::perspective_camera_parameters()
  : focal_len_(1.0), rot_(vnl_vector_fixed<double,3>(vnl_math::pi,0,0)), trans_()
{}

template<class T>
perspective_camera_parameters<T>::
perspective_camera_parameters(T focal_len, vgl_point_2d<T> const& principal_pt,
                              vgl_rotation_3d<T> const& rotation,
                              vgl_vector_3d<T> const& translation,
                              int nx, int ny)
  : focal_len_(focal_len), principal_pt_(principal_pt),
  rot_(rotation), trans_(translation), nx_(nx), ny_(ny)
{}

template<class T>
vpgl_perspective_camera<T> perspective_camera_parameters<T>::to_camera() const
{
  vpgl_perspective_camera<T> camera;
  to_camera(camera);
  return camera;
}


template<class T>
void perspective_camera_parameters<T>::to_camera(vpgl_perspective_camera<T> &camera) const
{
  vpgl_calibration_matrix<T> K(focal_len_, principal_pt_);
  camera = vpgl_perspective_camera<T>(K, rot_, trans_);
}

template<class T>
void perspective_camera_parameters<T>::from_camera(vpgl_perspective_camera<T> const& camera, int nx, int ny)
{
  vpgl_calibration_matrix<T> K = camera.get_calibration();
  double mean_scale_factor = (K.x_scale() + K.y_scale())/2.0;
  focal_len_ = K.focal_length() * mean_scale_factor;
  principal_pt_ = K.principal_point();
  rot_ = camera.get_rotation();
  trans_ = camera.get_translation();
  nx_ = nx;
  ny_ = ny;
}

template<class T >
void perspective_camera_parameters<T >::pretty_print(std::ostream &os) const
{
  os << "focal_len: " << focal_len_ << "principal_point: " << principal_pt_ << " rotation: " << rot_ << " translation: " << trans_ << " nx: " << nx_ << " ny: " << ny_;
  return;
}

template<class T>
vnl_vector<double> perspective_camera_parameters<T>::pack() const
{
  vnl_vector<double> x(num_params());
  x[0] = focal_len_;
  x[1] = principal_pt_.x();
  x[2] = principal_pt_.y();
  vnl_vector_fixed<double,4> q = rot_.as_quaternion();
  x[3] = q[0];
  x[4] = q[1];
  x[5] = q[2];
  x[6] = q[3];
  x[7] = trans_.x();
  x[8] = trans_.y();
  x[9] = trans_.z();

  return x;
}

template<class T>
void perspective_camera_parameters<T>::unpack(vnl_vector<double> const& x)
{
  if(x.size() != num_params()) {
    throw std::runtime_error("Argument of wrong size passed to unpack()");
  }
  focal_len_ = x[0];
  principal_pt_ = vgl_point_2d<double>(x[1],x[2]);
  vnl_vector_fixed<double,4> q(x[3],x[4],x[5],x[6]);
  rot_ = vgl_rotation_3d<double>(q);
  trans_ = vgl_vector_3d<double>(x[7],x[8],x[9]);
}

template<class T >
void perspective_camera_parameters<T >::write(std::ostream &os) const
{
  os << focal_len_ << " " << principal_pt_.x() << " " << principal_pt_.y() << " " << rot_ << " " << trans_.x() << " " << trans_.y() << " " << trans_.z() << " " << nx_ << " " << ny_;
  return;
}


template<class T >
void perspective_camera_parameters<T >::read(std::istream &is)
{
  is >> focal_len_;
  double ppx, ppy;
  is >> ppx >> ppy;
  principal_pt_ = vgl_point_2d<T>(ppx,ppy);
  is >> rot_;
  double tx,ty,tz;
  is >> tx >> ty >> tz;
  trans_ = vgl_vector_3d<double>(tx,ty,tz);
  is >> nx_ >> ny_;
  return;
}

//: adjust the camera for a scaled image
template<class T>
perspective_camera_parameters<T > perspective_camera_parameters<T >::scaled(double scale_factor) const
{
  int new_nx = static_cast<int>(nx_ * scale_factor);
  int new_ny = static_cast<int>(ny_ * scale_factor);
  // focal length has units pixels, so need to scale it with the resolution
  T new_focal_len = focal_len_*scale_factor;
  // principal point also has units pixels, so need to scale it too.
  vgl_point_2d<T> new_pp(principal_pt_.x()*scale_factor,
                         principal_pt_.y()*scale_factor);
  perspective_camera_parameters scaled_params(new_focal_len, new_pp, rot_, trans_, new_nx, new_ny);
  return scaled_params;
}


// Code for easy instantiation.
#undef PERSPECTIVE_CAMERA_PARAMETERS_INSTANTIATE
#define PERSPECTIVE_CAMERA_PARAMETERS_INSTANTIATE(T) \
template class face3d::perspective_camera_parameters<T >

}

#endif
