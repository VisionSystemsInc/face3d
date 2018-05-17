#include "mesh_io.h"
#include <Eigen/Dense>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <dlib/image_io.h>

using namespace face3d;

bool mesh_io::read_mesh(std::string const& filename, triangle_mesh &mesh, mesh_io::mesh_format_t mesh_format)
{
  triangle_mesh::VTYPE V;
  triangle_mesh::FTYPE F;
  triangle_mesh::NTYPE N;

  switch (mesh_format) {
    case FORMAT_PLY:
      {
        Eigen::MatrixXd UV;
        igl::readPLY(filename, V, F, N, UV);
      }
      break;
    default:
      std::cerr << "Unhandled mesh_format value: " << mesh_format << std::endl;
      return false;
  }
  mesh = triangle_mesh(V,F,N);
  return true;
}

bool mesh_io::write_mesh(std::string const& filename, triangle_mesh const& mesh, mesh_io::mesh_format_t mesh_format)
{
  switch(mesh_format) {
    case FORMAT_PLY:
      igl::writePLY(filename, mesh.V(), mesh.F(), mesh.N(), mesh.T());
      break;
    default:
      std::cerr << "Unsupported output mesh format " << mesh_format << std::endl;
      return false;
  }
  return true;
}
