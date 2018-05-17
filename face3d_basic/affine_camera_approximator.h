#ifndef affine_camera_approximator_h_included_
#define affine_camera_approximator_h_included_

#include <vpgl/vpgl_proj_camera.h>
#include <vpgl/vpgl_affine_camera.h>
#include <vgl/vgl_homg_point_3d.h>
#include <vgl/vgl_homg_point_2d.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_matrix_fixed.h>

namespace face3d {

template<class T>
class affine_camera_approximator {
public:
  affine_camera_approximator(vpgl_proj_camera<T> const& cam)
  : P_(cam.get_matrix()) {}

  //: returns an affine camera that approximates cam_ at world point p
  vpgl_affine_camera<T> operator()(vgl_homg_point_3d<T> const p)
  {
    double w = P_(2,0)*p.x() + P_(2,1)*p.y() + P_(2,2)*p.z() + P_(2,3)*p.w();
    vnl_vector_fixed<T,4> row1(P_.get_row(0)/w);
    vnl_vector_fixed<T,4> row2(P_.get_row(1)/w);
    return vpgl_affine_camera<T>(row1, row2);
  }

private:
  vnl_matrix_fixed<T,3,4> P_;
};

}

#endif
