#pragma once

#include "triangle_mesh.h"
#include "textured_triangle_mesh.h"
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

namespace face3d {

namespace mesh_io {

enum mesh_format_t {FORMAT_OBJ, FORMAT_STL, FORMAT_PLY, FORMAT_OFF};

bool read_mesh(std::string const& filename, triangle_mesh &mesh, mesh_format_t mesh_format=FORMAT_PLY);

template<class TEX_T>
bool read_mesh(std::string const& filename, std::string const& tex_filename,
               textured_triangle_mesh<TEX_T> &mesh, mesh_format_t mesh_format=FORMAT_PLY);

bool write_mesh(std::string const& filename, triangle_mesh const& mesh, mesh_format_t mesh_format=FORMAT_PLY);

}

template <class TEX_T>
bool mesh_io::read_mesh(std::string const& filename, std::string const& tex_filename,
              textured_triangle_mesh<TEX_T> &mesh, mesh_io::mesh_format_t mesh_format)
{
  TEX_T tex;
  dlib::load_image(tex, tex_filename.c_str());
  std::cout << "loaded texture " << tex_filename << ": " << tex.nc() << "x" << tex.nr() << std::endl;
  std::cout << "tex[0,0] = " << (int)tex[0][0].red <<"," << (int)tex[0][0].green << "," << (int)tex[0][0].blue<< std::endl;
  triangle_mesh::VTYPE V;
  triangle_mesh::FTYPE F;
  triangle_mesh::VTYPE N;
  triangle_mesh::TTYPE UV;

  switch (mesh_format) {
    case FORMAT_OBJ:
      {
        triangle_mesh::FTYPE FTC;
        triangle_mesh::FTYPE FN;
        igl::readOBJ(filename, V, UV, N, F, FTC, FN);
        std::cout << "FTC: " << FTC.rows() << "x" << FTC.cols() << std::endl;
        std::cout << "FN: " << FN.rows() << "x" << FN.cols() << std::endl;
        std::cout << "FTC[0] = " << FTC.row(0) << std::endl;
      }
      break;
    case FORMAT_PLY:
      {
        igl::readPLY(filename, V, F, N, UV);
      }
      break;
    default:
      std::cerr << "Unhandled mesh_format value: " << mesh_format << std::endl;
      return false;
  }
  mesh = textured_triangle_mesh<TEX_T>(V,F,N,UV,tex);
  return true;
}

}
