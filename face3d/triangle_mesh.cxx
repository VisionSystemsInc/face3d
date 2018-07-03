#include "triangle_mesh.h"
#include <cstdio>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <igl/readPLY.h>
#include <igl/per_vertex_normals.h>
#include <igl/writePLY.h>
#include <igl/writeOBJ.h>

using namespace face3d;

face3d::triangle_mesh::
triangle_mesh(VTYPE const& V, FTYPE const& F):
  V_(V), F_(F), N_(V.rows(),3), T_(V.rows(),2)
{
  compute_normals();
  normalize_normals();
}

face3d::triangle_mesh::
triangle_mesh(VTYPE const& V, FTYPE const& F, NTYPE const& N):
  V_(V), F_(F), N_(N), T_(V.rows(),2)
{
  normalize_normals();
}

face3d::triangle_mesh::
triangle_mesh(VTYPE const& V, FTYPE const &F, NTYPE const& N, TTYPE const& T) :
  V_(V), F_(F), N_(N), T_(T)
{
  normalize_normals();
}

face3d::triangle_mesh::
triangle_mesh(std::string const& ply_filename)
{
  if(!igl::readPLY(ply_filename, V_, F_, N_, T_)) {
    throw std::runtime_error("Error reading PLY file");
  }
  if (N_.rows() != V_.rows()) {
    compute_normals();
  }
  normalize_normals();
}

void
face3d::triangle_mesh::
set_vertices(std::vector<vgl_point_3d<double> > const& verts)
{
  set_vertices(verts.begin(), verts.end());
}

void
face3d::triangle_mesh::
set_texture_coords(std::vector<vgl_point_2d<double> > const& uvs)
{
  set_texture_coords(uvs.begin(), uvs.end());
}

void
face3d::triangle_mesh::
compute_normals()
{
  igl::per_vertex_normals(V_, F_, igl::PerVertexNormalsWeightingType::PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT, N_);
}

void
face3d::triangle_mesh::
normalize_normals()
{
  const unsigned int nverts = N_.rows();
  for (int i=0; i<nverts; ++i) {
    N_.row(i).normalize();
  }
}

void face3d::triangle_mesh::
save_obj(std::string const& mesh_filename) const
{
  igl::writeOBJ(mesh_filename, V_, F_, N_, F_, T_, F_);
}

void face3d::triangle_mesh::
save_ply(std::string const& mesh_filename) const
{
  igl::writePLY(mesh_filename, V_, F_, N_, T_);
}
