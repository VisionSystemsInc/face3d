#ifndef perspective_camera_parameters_h_included_
#define perspective_camera_parameters_h_included_

#include <iostream>
#include <vgl/vgl_vector_3d.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vpgl/vpgl_perspective_camera.h>

namespace face3d {

template<class T>
class perspective_camera_parameters
{
  public:
    typedef vpgl_perspective_camera<T> camera_type;

    perspective_camera_parameters();

    perspective_camera_parameters(T focal_len, vgl_point_2d<T> const& principal_pt,
                                  vgl_rotation_3d<T> const& rotation,
                                  vgl_vector_3d<T> const& translation,
                                  int nx, int ny);

    void to_camera(vpgl_perspective_camera<T> &camera) const;
    vpgl_perspective_camera<T> to_camera() const;

    void from_camera(vpgl_perspective_camera<T> const& camera, int nx, int ny);

    void write(std::ostream &os) const;
    void read(std::istream &is);
    void pretty_print(std::ostream &os) const;

    vgl_rotation_3d<T> rotation() const {return rot_;}
    void set_rotation(vgl_rotation_3d<T> const& rot) { rot_ = rot;}
    vgl_vector_3d<T> translation() const {return trans_;}
    void set_translation(vgl_vector_3d<T> const& trans) { trans_ = trans;}
    vgl_point_2d<T> principal_point() const {return principal_pt_;}
    T focal_len() const { return focal_len_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

    perspective_camera_parameters<T > scaled(double scale_factor) const;

    vnl_vector<double> pack() const;
    void unpack(vnl_vector<double> const& packed);

    static constexpr int num_params() { return 10; }

    void set_image_size(int nx, int ny) { nx_ = nx; ny_ = ny; }

  private:
    T focal_len_;
    vgl_point_2d<T> principal_pt_;
    vgl_rotation_3d<T> rot_;
    vgl_vector_3d<T> trans_;
    int nx_;
    int ny_;
};

template<class T >
std::ostream& operator << (std::ostream &os, perspective_camera_parameters<T > const& ocp)
{
  ocp.write(os);
  return os;
}

template<class T >
std::istream& operator >> (std::istream &is, perspective_camera_parameters<T > & ocp)
{
  ocp.read(is);
  return is;
}

}

#endif
