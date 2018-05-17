#include "head_mesh.h"
#include <dlib/image_io.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/dir_nav.h>

#include <stdexcept>

face3d::head_mesh::head_mesh(std::string const& data_dir, bool noneck)
{
  // add face mesh
  std::string face_mesh_path = data_dir + "/mean_face_head.ply";
  std::string face_tex_path = data_dir + "/mean_face_head.png";
  if (noneck) {
    std::string face_mesh_path = data_dir + "/mean_face_head_noneck.ply";
  }
  if (dlib::file_exists(face_mesh_path) && dlib::file_exists(face_tex_path)) {
    meshes_.push_back(face3d::textured_triangle_mesh<TEX_T>(face_mesh_path, face_tex_path));
  }
  else {
    std::cerr << "Missing one or both of " << face_mesh_path << ", " << face_tex_path << std::endl;
    throw std::runtime_error("Missing data files");
  }
      
  // add mouth mesh if available
  std::string mouth_mesh_path = data_dir + "/mean_face_mouth.ply";
  std::string mouth_tex_path = data_dir + "/mean_face_mouth.png";
  if (dlib::file_exists(mouth_mesh_path)) {
    std::string mouth_mesh_path = data_dir + "/mean_face_mouth.ply";
    if (dlib::file_exists(mouth_tex_path)) {
      meshes_.push_back(face3d::textured_triangle_mesh<TEX_T>(mouth_mesh_path, mouth_tex_path));
    }
    else {
      std::cerr << "Missing file " << mouth_tex_path << std::endl;
      throw std::runtime_error("Mouth mesh exists, but not texture map");
    }
  }

  for (face3d::textured_triangle_mesh<TEX_T> const& mesh : meshes_) {
    std::vector<vgl_point_3d<double> > verts;
    mesh.get_vertices(verts);
    base_verts_.insert(base_verts_.end(), verts.begin(), verts.end());
  }
}

void face3d::head_mesh::
apply_coefficients(vnl_matrix<double> const& subject_components,
                   vnl_matrix<double> const& expression_components,
                   vnl_vector<double> const& subject_coeffs,
                   vnl_vector<double> const& expression_coeffs)
{
  if ( subject_components.cols() != base_verts_.size()*3 ) {
    std::cerr << "ERROR: subject_components.cols() == " << subject_components.cols() << std::endl;
    std::cerr << "       base_verts_.size()*3 = " << base_verts_.size()*3 << std::endl;
    std::cerr <<  "Values should match, but don't." << std::endl;
    throw std::logic_error("Subject components matrix size does not match number of combined mesh vertices");
  }
  if ( expression_components.cols() != base_verts_.size()*3 ) {
    std::cerr << "ERROR: expression_components.cols() == " << expression_components.cols() << std::endl;
    std::cerr << "       base_verts_.size()*3 = " << base_verts_.size()*3 << std::endl;
    std::cerr <<  "Values should match, but don't." << std::endl;
    throw std::logic_error("Expression components matrix size does not match number of combined mesh vertices");
  }

  std::vector<vgl_point_3d<double> > warped_verts;
  face3d_util::apply_coefficients(base_verts_,
                                  subject_components, expression_components,
                                  subject_coeffs, expression_coeffs,
                                  warped_verts);

  size_t vert_offset = 0;
  for (auto &mesh : meshes_) {
    auto begin_it = warped_verts.begin() + vert_offset;
    vert_offset += mesh.num_vertices();
    auto end_it = warped_verts.begin() + vert_offset;
    mesh.set_vertices(begin_it, end_it);
  }

}

face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T> const&
face3d::head_mesh::mouth_mesh() const 
{ 
  if(!has_mouth())
  { 
    throw std::logic_error("No mouth mesh");
  }
  return meshes_[1];
}

face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T>&
face3d::head_mesh::mouth_mesh()
{ 
  if(!has_mouth())
  { 
    throw std::logic_error("No mouth mesh");
  }
  return meshes_[1];
}
