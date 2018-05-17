#pragma once

#include <string>
#include <vector>
#include "textured_triangle_mesh.h"
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <face3d_basic/face3d_util.h>

namespace face3d {

class head_mesh
{
public:
  using TEX_T = dlib::array2d<dlib::rgb_alpha_pixel>;

  head_mesh(std::string const& data_dir, bool noneck=false);

  std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes() const { return meshes_; }
  std::vector<face3d::textured_triangle_mesh<TEX_T> >& meshes() { return meshes_; }

  bool has_mouth() const { return meshes_.size() > 1; }

  textured_triangle_mesh<TEX_T> const& face_mesh() const { return meshes_[0];}
  textured_triangle_mesh<TEX_T> const& mouth_mesh() const;

  textured_triangle_mesh<TEX_T>& face_mesh() { return meshes_[0];}
  textured_triangle_mesh<TEX_T>& mouth_mesh();

  void apply_coefficients(vnl_matrix<double> const& subject_components,
                          vnl_matrix<double> const& expression_components,
                          vnl_vector<double> const& subject_coeffs,
                          vnl_vector<double> const& expression_coeffs);

  template<class T>
  void get_vertices(std::vector<T> &verts);

private:
  std::vector<face3d::textured_triangle_mesh<TEX_T> > meshes_;
  std::vector<vgl_point_3d<double> > base_verts_;

};


template<class T>
void head_mesh::get_vertices(std::vector<T> &verts)
{
  verts.clear();
  for (face3d::textured_triangle_mesh<TEX_T> const& mesh : meshes_) {
    std::vector<vgl_point_3d<double> > mesh_verts;
    mesh.get_vertices(mesh_verts);
    verts.insert(verts.end(), mesh_verts.begin(), mesh_verts.end());
  }
}

}
