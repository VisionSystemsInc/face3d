#ifndef io_utils_h_included_
#define io_utils_h_included_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vil/vil_rgb.h>
#include <vil/vil_image_view.h>
#include <vil/vil_save.h>
#include <vil/vil_load.h>
#include <vul/vul_file.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_io.h>
#include "ortho_camera_parameters.h"
#include "perspective_camera_parameters.h"

#include <dlib/array2d.h>
#include <dlib/geometry.h>

namespace face3d {

class io_utils
{
public:

  static vnl_matrix<double> read_numpy_matrix(std::string const&filename);

  static vnl_matrix<double> read_matrix(std::string const& filename);

  static void write_matrix(vnl_matrix<double> const&m, std::string const& filename);

  static vnl_vector<double> read_vector(std::string const& filename);

  static void write_vector(vnl_vector<double> const&v, std::string const& filename);

  static void write_strings(std::vector<std::string> const& strings, std::string const& filename);

  static bool read_strings(std::string const& filename, std::vector<std::vector<std::string> > &strings);
  static std::vector<std::string> read_strings(std::string const& filename);

  static bool read_string(std::string const& filename, std::string& str);

  // determine correct camera type to write at runtime
  static bool write_camera(const vpgl_proj_camera<double> *cam, std::string const& filename);

  // helper function
  template<class T, long D>
  static T get_vector_element(dlib::vector<T,D> const& v, int d)
  {
    switch(d) {
      case 0: return v.x();
      case 1: return v.y();
      case 2: return v.z();
    }
    throw std::runtime_error("Unhandled vector dimension in get_vector_element()");
  }

  template<class T, long D>
  static void save_tiff(dlib::array2d<dlib::vector<T,D> > const& array, std::string const& filename)
  {
    const int nx = array.nc();
    const int ny = array.nr();
    // dlib does not have support for writing TIFFs, so go through vxl/vil
    vil_image_view<T> img(nx, ny, D);
    for (int y=0; y<ny; ++y) {
      for (int x=0; x<nx; ++x) {
        for (int d=0; d<D; ++d) {
          img(x,y,d) = get_vector_element(array[y][x],d);
        }
      }
    }
    // use vil to save as tiff
    vil_save(img, filename.c_str());
  }

  template<class T>
  static void save_3d_tiff(dlib::array2d<T> const& array, std::string const& filename)
  {
    const int nx = array.nc();
    const int ny = array.nr();
    // dlib does not have support for writing TIFFs, so go through vxl/vil
    vil_image_view<float> img(nx, ny, 3);
    for (int y=0; y<ny; ++y) {
      for (int x=0; x<nx; ++x) {
        T pt(array[y][x]);
        img(x,y,0) = pt.x();
        img(x,y,1) = pt.y();
        img(x,y,2) = pt.z();
      }
    }
    // use vil to save as tiff
    vil_save(img, filename.c_str());
  }

  template <class T>
  static void load_3d_tiff(dlib::array2d<T> &img, std::string const& filename);

  // wrapper around dlib's image IO that chooses format based on extension.
  template<class IMG_T>
  static void save_image(IMG_T const& img, std::string const& filename);

  template<class T>
  static bool write_t(T const& obj, std::string const& filename)
  {
    std::ofstream ofs(filename.c_str());
    if(!ofs.good()){
      std::cerr << "ERROR opening file " << filename << " for write." << std::endl;
      return false;
    }
    ofs << obj;
    return true;
  }

  template<class T>
  static bool read_t(std::string const& filename, T& obj)
  {
    std::ifstream ifs(filename.c_str());
    if(!ifs.good()){
      std::cerr << "ERROR opening file " << filename << " for read." << std::endl;
      return false;
    }
    ifs >> obj;
    return true;
  }

  template<class T>
  static void write_points(std::vector<vgl_point_2d<T> > const& pts, std::string const& filename);
  template<class T>
  static void write_points(std::vector<vgl_point_3d<T> > const& pts, std::string const& filename);

  template<class T>
  static void read_points(std::string const& filename, std::vector<vgl_point_2d<T> > &pts);
  template<class T>
  static void read_points(std::string const& filename, std::vector<vgl_point_3d<T> > &pts);

  // read items from file, one per line
  template<class T>
  static void read_items(std::string const& filename, std::vector<T> &items);

  static vpgl_perspective_camera<double> read_camera(std::string const& filename);
  static vpgl_affine_camera<double> read_affine_camera(std::string const& filename);

  static bool write_ply( std::vector<vgl_point_3d<double> > const& points,
                        std::vector<vil_rgb<unsigned char> > const& colors,
                        std::string const& filename);
};


// ---- Template Definitions ----//

template<class T>
void io_utils::read_items(std::string const& filename, std::vector<T> &items)
{
  items.clear();
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    throw std::runtime_error("ERROR opening file for read");
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss(line);
    T item;
    ss >> item;
    items.push_back(item);
  }
  return;
}

template<class T>
void io_utils::write_points(std::vector<vgl_point_2d<T> > const& pts, std::string const& filename)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()) {
    std::cerr << "ERROR opening file " << filename << " for write!" << std::endl;
    throw std::runtime_error("Error opening points file for write");
  }
  for (int i=0; i<pts.size(); ++i) {
    ofs << pts[i].x() << " " << pts[i].y() << std::endl;
  }
}

template<class T>
void io_utils::write_points(std::vector<vgl_point_3d<T> > const& pts, std::string const& filename)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()) {
    std::cerr << "ERROR opening file " << filename << " for write!" << std::endl;
    throw std::runtime_error("Error opening points file for write");
  }
  for (int i=0; i<pts.size(); ++i) {
    ofs << pts[i].x() << " " << pts[i].y() << " " << pts[i].z() << std::endl;
  }
}

template<class T>
void io_utils::read_points(std::string const& filename, std::vector<vgl_point_2d<T> >& pts)
{
  pts.clear();
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    std::cerr << "ERROR opening file " << filename << " for read" << std::endl;
    throw std::runtime_error("Error opening points file for read");
  }
  std::string line;
  while(std::getline(ifs,line))  {
    std::stringstream ss(line);
    T x,y;
    if(!(ss >> x >> y )) {
      throw std::runtime_error("Error reading x,y from line");
    }
    pts.push_back(vgl_point_2d<T>(x,y));
  }
}

template<class T>
void io_utils::read_points(std::string const& filename, std::vector<vgl_point_3d<T> >& pts)
{
  pts.clear();
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    std::cerr << "ERROR opening file " << filename << " for read" << std::endl;
    throw std::runtime_error("Error opening points file for read");
  }
  std::string line;
  while(std::getline(ifs,line))  {
    std::stringstream ss(line);
    T x,y,z;
    if(!(ss >> x >> y >> z )) {
      throw std::runtime_error("Error reading x,y,z from line");
    }
    pts.push_back(vgl_point_3d<T>(x,y,z));
  }
}

template<class IMG_T>
void io_utils::save_image(IMG_T const& img, std::string const& filename)
{
  std::string ext = vul_file::extension(filename);
  for (char &c : ext) { c = std::tolower(c); }
  if ((ext == ".jpg") || (ext == ".jpeg")) {
    dlib::save_jpeg(img, filename.c_str());
  }
  else if (ext == ".png") {
    dlib::save_png(img, filename.c_str());
  }
  else {
    std::cerr << "ERROR: unsupported image extension: " << ext << std::endl;
  }
}

template<class T>
void io_utils::load_3d_tiff(dlib::array2d<T> &img, std::string const& filename)
{
  vil_image_view<float> img_vil = vil_load(filename.c_str());
  if (img_vil.nplanes() != 3) {
    std::cerr << "ERROR: io_utils::load_3d_tiff(): image does not have 3 planes" << std::endl;
    return;
  }
  const int nx = img_vil.ni();
  const int ny = img_vil.nj();
  img.set_size(ny, nx);
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      img[y][x] = T(img_vil(x,y,0), img_vil(x,y,1), img_vil(x,y,2));
    }
  }
}

}
#endif
