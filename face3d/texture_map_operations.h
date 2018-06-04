#pragma once

#include <utility>
#include <vector>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/geometry.h>
#include "triangle_mesh.h"
#include "head_mesh.h"
#include "mesh_renderer.h"

namespace face3d {

void merge_textures(std::vector<dlib::array2d<dlib::rgb_alpha_pixel> > const& img_textures,
                    dlib::array2d<dlib::rgb_alpha_pixel> &merged);

void generate_symmetry_map(triangle_mesh::VTYPE const& V, triangle_mesh::FTYPE const& F,
                           triangle_mesh::TTYPE const& UV, int tex_nx, int tex_ny,
                           dlib::array2d<dlib::vector<float,2> > &sym_map);

void generate_face_symmetry_map(face3d::head_mesh const& base_mesh,
                                dlib::array2d<dlib::vector<float,2> > &sym_map);

void complete_texture(dlib::array2d<dlib::rgb_alpha_pixel> const& tex,
                      dlib::array2d<dlib::vector<float,2> > const& sym_map,
                      dlib::array2d<dlib::rgb_alpha_pixel> &tex_completed);

void mask_mesh_faces(triangle_mesh const& mesh, dlib::array2d<unsigned char> const& texmask,
                std::vector<size_t> &valid_faces);

template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
  void image_to_texture(IMG_T_IN const& img,
                        std::vector<textured_triangle_mesh<TEX_T> > const& meshes, CAM_T const& cam_params,
                        mesh_renderer &renderer,
                        IMG_T_OUT &tex_out)
{
  std::vector<IMG_T_OUT > textures_out;
  renderer.set_ambient_weight(1.0f);
  renderer.render_to_texture(meshes, img, cam_params, textures_out);
  if (textures_out.size() == 0) {
    std::cerr << "ERROR: render_to_texture returned array of 0 textures" << std::endl;
    throw std::runtime_error("unexpected return value from render_to_texture");
  }
  if (textures_out.size() > 1) {
    std::cerr << "WARNING: image_to_texture(): got multiple textures back from render_to_texture().  Using first only." << std::endl;
  }
  std::swap(textures_out[0], tex_out);
  return;
}

template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
void texture_to_image(IMG_T_IN const& tex,
                      textured_triangle_mesh<TEX_T> const& mesh, CAM_T const& cam_params,
                      mesh_renderer &renderer,
                      IMG_T_OUT &img_out)
{
  std::vector<textured_triangle_mesh<IMG_T_IN> > meshes
  { textured_triangle_mesh<IMG_T_IN>(mesh, tex) };

  renderer.set_ambient_weight(1.0f);
  renderer.render(meshes, cam_params, img_out);
  return;
}

}
