#ifndef ortho_camera_parameters_h_
#define ortho_camera_parameters_h_

#include <vnl/vnl_matrix_fixed.h>
#include <vgl/vgl_vector_2d.h>
#include <vgl/vgl_vector_3d.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vpgl/vpgl_affine_camera.h>

namespace face3d {

template<class T>
class ortho_camera_parameters
{
  public:
    typedef vpgl_affine_camera<T> camera_type;

    //: default constructor
    ortho_camera_parameters();

    //: construct assuming uniform scale and 0 skew
    ortho_camera_parameters(T scale, vgl_vector_2d<T> offset, vgl_rotation_3d<T> rotation, int nx, int ny);

    //: construct assuming uniform scale and 0 skew
    ortho_camera_parameters(T scale, vgl_vector_2d<T> offset, vgl_rotation_3d<T> rotation, vnl_matrix_fixed<T,3,3> H_3d, int nx, int ny);

    //: construct from a vpgl affine_camera
    ortho_camera_parameters(vpgl_affine_camera<T> const& camera, int nx, int ny) {this->from_camera(camera, nx, ny);}

    //: get 3x3 intrinsic parameter matrix
    void getK(vnl_matrix_fixed<T,3,3> &K) const;

    //: get camera
    void to_camera(vpgl_affine_camera<T> &camera, bool force_ortho=true) const;
    vpgl_affine_camera<T> to_camera() const;

    //: best fit to affine camera
    void from_camera(vpgl_affine_camera<T> const& camera, int nx, int ny);

    // getters
    T scale() const { return scale_; }
    vgl_vector_2d<T> const& offset() const { return offset_; }
    vgl_rotation_3d<T> const& rotation() const { return rotation_; }
    vnl_matrix_fixed<T,3,3> H_3d() const { return H_3d_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

    // setters
    void set_scale(T scale) { scale_ = scale; }
    void set_offset(vgl_vector_2d<T> const& offset) { offset_ = offset;}
    void set_rotation(vgl_rotation_3d<T> const& rotation) { rotation_ = rotation; }
    void set_image_size(int nx, int ny) { nx_=nx; ny_=ny; }

    // return copy for scaled image
    ortho_camera_parameters<T> scaled(T scale_factor) const;

    void write(std::ostream &os) const;
    void read(std::istream &is);

    // for backwards compatibility with old format that did not include the image size
    void write_no_img_dims(std::ostream &os) const;
    void read_no_img_dims(std::istream &is);

    void pretty_print(std::ostream &os) const;

    vnl_vector<double> pack() const;
    void unpack(vnl_vector<double> const& packed);

    static constexpr int num_params() { return 7; }

  private:

    T scale_;
    vgl_vector_2d<T> offset_;
    vgl_rotation_3d<T> rotation_;
    //: transform to be applied to 3-d points
    vnl_matrix_fixed<T,3,3> H_3d_;

    // image dimensions
    int nx_;
    int ny_;
};

template<class T >
std::ostream& operator << (std::ostream &os, ortho_camera_parameters<T > const& ocp)
{
  ocp.write(os);
  return os;
}

template<class T >
std::istream& operator >> (std::istream &is, ortho_camera_parameters<T > & ocp)
{
  ocp.read(is);
  return is;
}

}

#endif
