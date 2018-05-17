#ifndef ortho_camera_parameters_txx_
#define ortho_camera_parameters_txx_

#include "ortho_camera_parameters.h"

#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_qr.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <vnl/vnl_det.h>

namespace face3d {

//: default constructor
template<class T >
ortho_camera_parameters<T >::ortho_camera_parameters()
  : scale_(1), offset_(), rotation_(vnl_vector_fixed<double,3>(vnl_math::pi,0,0)), nx_(0), ny_(0)
{
  H_3d_.set_identity();
}

//: construct assuming uniform scale and 0 skew
template<class T >
ortho_camera_parameters<T >:: ortho_camera_parameters(T scale, vgl_vector_2d<T> offset, vgl_rotation_3d<T> rotation, int nx, int ny)
  : scale_(scale), offset_(offset), rotation_(rotation), nx_(nx), ny_(ny)
{
  H_3d_.set_identity();
}

//: construct assuming uniform scale and 0 skew
template<class T >
ortho_camera_parameters<T >::ortho_camera_parameters(T scale, vgl_vector_2d<T> offset, vgl_rotation_3d<T> rotation, vnl_matrix_fixed<T,3,3> H_3d, int nx, int ny)
  : scale_(scale), offset_(offset), rotation_(rotation), H_3d_(H_3d), nx_(nx), ny_(ny)
{
}

//: get 3x3 intrinsic parameter matrix
template<class T >
void ortho_camera_parameters<T >::getK(vnl_matrix_fixed<T,3,3> &K) const
{
  K.fill(0);
  K[0][0] = scale_;
  K[1][1] = scale_;
  K[0][2] = offset_.x();
  K[1][2] = offset_.y();
  K[2][2] = 1;

  return;
}

//: adjust the camera for a scaled image
template<class T>
ortho_camera_parameters<T > ortho_camera_parameters<T >::scaled(T scale_factor) const
{
  vgl_vector_2d<T> new_offset(offset_.x()*scale_factor, offset_.y()*scale_factor);
  T new_scale = scale_*scale_factor;
  int new_nx = static_cast<int>(nx_ * scale_factor);
  int new_ny = static_cast<int>(ny_ * scale_factor);
  ortho_camera_parameters scaled_params(new_scale, new_offset, rotation_, H_3d_, new_nx, new_ny);
  return scaled_params;
}

template<class T>
vpgl_affine_camera<T> ortho_camera_parameters<T>::to_camera() const
{
  vpgl_affine_camera<T> camera;
  to_camera(camera);
  return camera;
}

//: get camera
template<class T >
void ortho_camera_parameters<T >::to_camera(vpgl_affine_camera<T> &camera, bool force_ortho) const
{
  vnl_matrix_fixed<T,3,3> K;
  getK(K);
  const vgl_vector_3d<T> trans(T(0), T(0), T(0));

  vnl_matrix_fixed<T,3,3> R = rotation_.as_matrix();
  if (!force_ortho) {
    R *= H_3d_;
  }
  vnl_vector_fixed<T,4> row1(R[0][0]*scale_, R[0][1]*scale_, R[0][2]*scale_, scale_*trans.x() + K[0][2]);
  vnl_vector_fixed<T,4> row2(R[1][0]*scale_, R[1][1]*scale_, R[1][2]*scale_, scale_*trans.y() + K[1][2]);

  camera.set_rows(row1, row2);

  camera.set_viewing_distance( 1000.0 );
  // 3rd row of rotation matrix is camera z axis in world coordinate system
  vgl_vector_3d<T> look_dir(R[2][0], R[2][1], R[2][2]);
  camera.orient_ray_direction( look_dir );
}

//: fit from affine camera
template<class T >
void ortho_camera_parameters<T >::from_camera(vpgl_affine_camera<T> const& camera, int nx, int ny)
{
  nx_ = nx;
  ny_ = ny;
  vnl_matrix_fixed<T,3,4> const P = camera.get_matrix();
  vnl_matrix_fixed<T,2,3> A = P.get_n_rows(0,2).get_n_columns(0,3);

  // We want RQ decomposition. vnl only provides QR, so jump through some hoops
  vnl_matrix_fixed<T,3,2> A1 = A.transpose().fliplr();

  vnl_qr<T> QR(A1);
  vnl_matrix_fixed<T,2,2> AR1 = QR.R().get_n_rows(0,2);
  vnl_matrix_fixed<T,3,2> AQ1 = QR.Q().get_n_columns(0,2);

  vnl_matrix_fixed<T,2,2> AR = AR1.transpose().flipud().fliplr();
  vnl_matrix_fixed<T,2,3> AQ = AQ1.transpose().flipud();

  // AR, AQ is now the RQ decomposition of A

  // make sure diagonal elements of AR are positive
  vnl_matrix_fixed<T,2,2> sign_swap;
  sign_swap.set_identity();
  for (unsigned r=0; r<2; ++r) {
    if (AR[r][r] < 0) {
      sign_swap[r][r] = -1;
    }
  }
  AR = AR * sign_swap;
  AQ = sign_swap * AQ;

  // recompose as sanity check
  vnl_matrix_fixed<T,2,3> testA = AR*AQ;
  assert((testA - A).absolute_value_max() < 1e-3);

  // enforce scale_x == scale_y by taking mean
  scale_ = (AR[0][0] + AR[1][1]) / 2.0;
  // grab offset directly from projection matrix
  offset_ = vgl_vector_2d<T>(P[0][3], P[1][3]);
  // create rotation object from orthonormal marix AQ
  vgl_vector_3d<T> rx(AQ[0][0], AQ[0][1], AQ[0][2]);
  vgl_vector_3d<T> ry(AQ[1][0], AQ[1][1], AQ[1][2]);
  vgl_vector_3d<T> rz = cross_product(rx,ry);
  vnl_matrix_fixed<T,3,3> rot_matrix;
  rot_matrix[0][0] = rx.x(); rot_matrix[0][1] = rx.y(); rot_matrix[0][2] = rx.z();
  rot_matrix[1][0] = ry.x(); rot_matrix[1][1] = ry.y(); rot_matrix[1][2] = ry.z();
  rot_matrix[2][0] = rz.x(); rot_matrix[2][1] = rz.y(); rot_matrix[2][2] = rz.z();
  rotation_ = vgl_rotation_3d<T>(rot_matrix);


  vnl_matrix_fixed<T,3,3> Kortho(0.0);
  Kortho[0][0] = scale_;
  Kortho[1][1] = scale_;
  Kortho[0][2] = offset_.x();
  Kortho[1][2] = offset_.y();
  Kortho[2][2] = 1.0;
  vnl_matrix_fixed<T,3,3> K(0.0);
  for (unsigned r=0; r<2; ++r) {
    for (unsigned c=0; c<2; ++c) {
      K[r][c] = AR[r][c];
    }
  }
  K[0][2] = offset_.x();
  K[1][2] = offset_.y();
  K[2][2] = 1.0;

  vnl_matrix_fixed<T,3,3> Kresidual = vnl_matrix_inverse<T>(K) * Kortho.as_ref();

  // now construct the transform H that can be applied to 3-d points to account for A
  vnl_matrix_fixed<T,3,3> H = rotation_.inverse().as_matrix() * (vnl_matrix_inverse<T>(Kresidual) * rotation_.as_matrix().as_ref());
  //std::cout << "H = " << H << std::endl;

  // factor rotation component out of H and apply to extrinsics
  vnl_qr<T> Hqr(H);
  vnl_matrix_fixed<T,3,3> H_rot = Hqr.Q();
  vnl_matrix_fixed<T,3,3> H_tri = Hqr.R();
  // make sure diagonal components of H_tri are positive
  vnl_matrix_fixed<T,3,3> H_sign_swap;
  H_sign_swap.set_identity();
  for (unsigned r=0; r<3; ++r) {
    if (H_tri[r][r] < 0) {
      H_sign_swap[r][r] = -1;
    }
  }
  H_tri = H_sign_swap * H_tri;
  H_rot = H_rot * H_sign_swap;

  // sanity check - recompose H
  vnl_matrix_fixed<T,3,3> testH = H_rot * H_tri;
  assert((testH - H).absolute_value_max() < 1e-3);

  rotation_ = rotation_ * vgl_rotation_3d<T>(H_rot);
  H_3d_ = H_tri;

  // final sanity check - recompose affine camera
  vpgl_affine_camera<T> test_cam;
  this->to_camera(test_cam, false);
  vnl_matrix_fixed<T,3,4> testP = test_cam.get_matrix();
  assert((testP - P).absolute_value_max() < 1e-3);

  nx_ = nx;
  ny_ = ny;
}

template<class T >
void ortho_camera_parameters<T >::pretty_print(std::ostream &os) const
{
  os << "scale: " << scale_ << " offset: " << offset_ << " rotation: " << rotation_;
  return;
}

template<class T >
void ortho_camera_parameters<T >::write(std::ostream &os) const
{
  os << nx_ << " " << ny_ << " " << scale_ << " " << offset_.x() << " " << offset_.y() << " " << rotation_;
  return;
}

template<class T>
vnl_vector<double> ortho_camera_parameters<T>::pack() const
{
  vnl_vector<double> x(num_params());
  x[0] = scale_;
  x[1] = offset_.x();
  x[2] = offset_.y();
  vnl_vector_fixed<double,4> q = rotation_.as_quaternion();
  x[3] = q[0];
  x[4] = q[1];
  x[5] = q[2];
  x[6] = q[3];

  return  x;
}

template<class T>
void ortho_camera_parameters<T>::unpack(vnl_vector<double> const& x)
{
  if(x.size() != num_params()) {
    throw std::runtime_error("Argument of wrong size passed to unpack()");
  }
  scale_ = x[0];
  offset_ = vgl_vector_2d<double>(x[1],x[2]);
  vnl_vector_fixed<double,4> q(x[3],x[4],x[5],x[6]);
  rotation_ = vgl_rotation_3d<double>(q);
}

template<class T >
void ortho_camera_parameters<T >::read(std::istream &is)
{
  T ox,oy;
  is >> nx_ >> ny_ >> scale_ >> ox >> oy >> rotation_;
  offset_ = vgl_vector_2d<T>(ox,oy);
  H_3d_.set_identity();
  return;
}

template<class T >
void ortho_camera_parameters<T >::write_no_img_dims(std::ostream &os) const
{
  os << scale_ << " " << offset_.x() << " " << offset_.y() << " " << rotation_;
  return;
}

template<class T >
void ortho_camera_parameters<T >::read_no_img_dims(std::istream &is)
{
  T ox,oy;
  is >> scale_ >> ox >> oy >> rotation_;
  offset_ = vgl_vector_2d<T>(ox,oy);
  H_3d_.set_identity();
  nx_ = 0;
  ny_ = 0;
  return;
}

// Code for easy instantiation.
#undef ORTHO_CAMERA_PARAMETERS_INSTANTIATE
#define ORTHO_CAMERA_PARAMETERS_INSTANTIATE(T) \
template class face3d::ortho_camera_parameters<T >

}


#endif // ortho_camera_parameters.txx
