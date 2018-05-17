#ifndef face3d_textured_triangle_mesh_h_included_
#define face3d_textured_triangle_mesh_h_included_

#include <vector>
#include <Eigen/Dense>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <vgl/vgl_point_3d.h>
#include <igl/per_vertex_normals.h>
#include <igl/readPLY.h>
#include <igl/writeOBJ.h>
#undef basename
#include <vul/vul_file.h>
#include "triangle_mesh.h"
#include <face3d_basic/face3d_img_util.h>

namespace face3d
{
  template <class TEX_T>
  class textured_triangle_mesh : public triangle_mesh
  {
    public:
      textured_triangle_mesh(){}
      textured_triangle_mesh(VTYPE const& V, FTYPE const &F, NTYPE const& N,
                             TTYPE const& T, TEX_T const& texture);

      textured_triangle_mesh(triangle_mesh const& mesh, TEX_T const& texture);

      textured_triangle_mesh(std::string const& ply_filename, std::string const& texture_filename);

      textured_triangle_mesh(std::string const& ply_filename, TEX_T const& texture);

      textured_triangle_mesh(textured_triangle_mesh<TEX_T> const& other);
      textured_triangle_mesh& operator = (textured_triangle_mesh const& other);
      ~textured_triangle_mesh();

      template<class PIX>
      void set_texture(dlib::array2d<PIX> const& tex)
      {
        dlib::assign_all_pixels(texture_, dlib::rgb_alpha_pixel(0,0,0,0));
        face3d_img_util::assign_image(texture_, tex);
      }

      TEX_T const& texture() const { return texture_;}

      void save_obj(std::string const& mesh_filename) const;

    private:
      TEX_T texture_;

  };
}


// ----------------- Template Definitions -----------------//

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
textured_triangle_mesh(triangle_mesh const& mesh, TEX_T const& texture) :
  triangle_mesh(mesh)
{
  face3d_img_util::assign_image(texture_,texture);
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
textured_triangle_mesh(VTYPE const& V, FTYPE const &F, NTYPE const& N,
                       TTYPE const& T, TEX_T const& texture) :
  triangle_mesh(V,F,N,T)
{
  face3d_img_util::assign_image(texture_,texture);
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
textured_triangle_mesh(std::string const& ply_filename, std::string const& texture_filename)
  : triangle_mesh(ply_filename)
{
  if (texture_filename == "") {
    // use dummy texture image
    texture_.set_size(64,64);
    dlib::assign_all_pixels(texture_, dlib::rgb_alpha_pixel(0,255,0,255));
  }
  else {
    dlib::load_image(texture_, texture_filename.c_str());
  }
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
textured_triangle_mesh(std::string const& ply_filename, TEX_T const& texture)
  : triangle_mesh(ply_filename)
{
  face3d_img_util::assign_image(texture_, texture);
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
textured_triangle_mesh(textured_triangle_mesh const& other)
: triangle_mesh(other)
{
  face3d_img_util::assign_image(texture_, other.texture_);
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>&
face3d::textured_triangle_mesh<TEX_T>::
operator = (textured_triangle_mesh const& other)
{
  triangle_mesh::operator=(other);
  face3d_img_util::assign_image(texture_, other.texture_);
  return *this;
}

template<class TEX_T>
face3d::textured_triangle_mesh<TEX_T>::
~textured_triangle_mesh()
{ }


template<class TEX_T>
void face3d::textured_triangle_mesh<TEX_T>::
save_obj(std::string const& mesh_filename) const
{
  std::string temp_fname = mesh_filename + ".tmp";
  igl::writeOBJ(temp_fname, V(), F(), N(), F(), T(), F());

  std::string mesh_dir = vul_file::dirname(mesh_filename);
  std::string basename = vul_file::basename(mesh_filename);
  std::string texture_basename = basename + ".png";
  std::string texture_filename = mesh_dir + "/" + texture_basename;
  dlib::save_png(texture_, texture_filename.c_str());
  // write obj material file
  std::string mtl_basename = basename + ".mtl";
  std::string mtl_fname = mesh_dir + "/" + mtl_basename;
  {
    std::ofstream ofs(mtl_fname.c_str());
    if (!ofs.good()) {
      std::cerr << "ERROR: unable to open mlt file for write: " << mtl_fname << std::endl;
      return;
    }
    ofs << "newmtl tex_mtl" << std::endl;
    ofs << "Ka 0.200000 0.200000 0.200000" << std::endl;
    ofs << "Kd 0.698039 0.698039 0.698039" << std::endl;
    ofs << "Ks 1.000000 1.000000 1.000000" << std::endl;
    ofs << "Tr 1.000000" << std::endl;
    ofs << "illum 2" << std::endl;
    ofs << "Ns 0.000000" << std::endl;
    ofs << "map_Kd " << texture_basename << std::endl;
  }
  // prepend necessary texture info to obj file
  std::ofstream ofs(mesh_filename.c_str());
  if (!ofs.good()) {
    std::cerr << "ERROR: unable to open mesh file for write: " << mesh_filename << std::endl;
    return;
  }
  ofs << "mtllib " << "./" + mtl_basename << std::endl;
  ofs << "usemtl tex_mtl" << std::endl;
  std::ifstream ifs(temp_fname.c_str());
  if (!ifs.good()) {
    std::cerr << "Unable to re-open temporary file " << temp_fname << std::endl;
    return;
  }
  std::string line;
  while(std::getline(ifs, line)) {
    ofs << line << std::endl;
  }
  std::remove(temp_fname.c_str());
}

#endif
