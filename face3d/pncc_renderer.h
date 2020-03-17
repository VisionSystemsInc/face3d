#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <dlib/cmd_line_parser.h>
#include <face3d_basic/cmd_line_util.h>
#include <exception>
#include <face3d_basic/io_utils.h>
#include <face3d/mesh_renderer.h>
#include <face3d/mesh_io.h>
#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d_basic/face3d_util.h>
#include <face3d/triangle_mesh.h>
#include <face3d/semantic_map.h>
#include <face3d_basic/cmd_line_util.h>
#include <face3d_basic/io_utils.h>
#include <face3d/mesh_renderer.h>
#include <face3d/mesh_io.h>
#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d_basic/face3d_util.h>
#include <face3d/triangle_mesh.h>
#include <face3d/semantic_map.h>
#include <face3d_basic/cmd_line_util.h>

#include <vgl/vgl_point_3d.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vul/vul_file.h>
#include <vil/vil_save.h>

#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
namespace face3d{
  template <class CAM_T>
  class pncc_and_offsets_renderer{
  public:
    pncc_and_offsets_renderer():debug_mode_(false), debug_dir_(""){}
    pncc_and_offsets_renderer(std::string debug_dir){
      this->debug_dir_ = debug_dir;
      this->use_flame_= true;
      this->noneck_ = false;
      this->use_3DMM_ = false;
    }
    pncc_and_offsets_renderer(std::string mesh_normalization_box,std::string debug_dir){

      this->debug_dir_ = debug_dir;
      this->use_flame_= false;
      this->noneck_ = false;
      this->use_3DMM_ = false;
      if (mesh_normalization_box=="3dmm"){
        this->use_3DMM_ = true;
      }else if(mesh_normalization_box=="flame"){
        this->use_flame_ = true;
      }else{
        std::cout<<"Invalid mesh box identifier \""<<mesh_normalization_box<<"\".Choices are: "<<" \"3dmm\", \"flame\". Defaulting to \"flame\""<<std::endl;
        this->use_flame_ = true;
      }

    }
    void render(face3d::triangle_mesh& base_mesh, face3d::triangle_mesh& articulated_mesh,
                CAM_T& camera_params, std::string output_dir, std::string basename, std::string extension);
  private:
    face3d::mesh_renderer renderer_;
    bool debug_mode_, noneck_, use_3DMM_, use_flame_;
    std::string debug_dir_;
  };
}


template <class CAM_T>
void face3d::pncc_and_offsets_renderer<CAM_T>::render(face3d::triangle_mesh& base_mesh,
                                                      face3d::triangle_mesh& articulated_mesh, CAM_T& camera_params,
                                                      std::string output_dir, std::string basename, std::string extension){

  int nx = camera_params.nx();
  int ny = camera_params.ny();

  face3d::triangle_mesh::VTYPE const& V_og = base_mesh.V();
  face3d::triangle_mesh::FTYPE const& F_og = base_mesh.F();

  face3d::triangle_mesh::VTYPE const& V = articulated_mesh.V();
  face3d::triangle_mesh::FTYPE const& F = articulated_mesh.F();

  if (V.size() != V_og.size()){
    throw std::runtime_error("Meshes should have the same number of vertices");
  }

  if (F.size() != F_og.size()){
    throw std::runtime_error("Meshes should have the same number of faces");
  }
  dlib::array2d<dlib::rgb_alpha_pixel> dummy_image(ny,nx);
  dlib::array2d<vgl_point_3d<float> > img3d(ny,nx);
  dlib::array2d<vgl_point_3d<float> > face_bc_img(ny,nx);
  dlib::array2d<GLint> face_idx_img(ny, nx);
  face3d::render_aux_out ancillary_buffers;
  ancillary_buffers.img3d_ = &img3d;
  ancillary_buffers.face_bc_ = &face_bc_img;
  ancillary_buffers.face_idx_ = &face_idx_img;

  std::vector<face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T> > renderer_input;

  face3d::head_mesh::TEX_T dummy_texture(1,1);
  dummy_texture[0][0] = dlib::rgb_alpha_pixel(255,0,0,255);
  face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T> input;
  if (articulated_mesh.UV().rows()!= articulated_mesh.V().rows()){
    face3d::triangle_mesh::TTYPE TE(articulated_mesh.V().rows(), 2);
    for (unsigned j=0; j<articulated_mesh.V().rows(); j++)
      TE.row(j) = Eigen::RowVector2d(0,0);
    input = face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T>(articulated_mesh.V(), articulated_mesh.F(), articulated_mesh.N(), TE, dummy_texture);
  }else{
    input = face3d::textured_triangle_mesh<face3d::head_mesh::TEX_T>(articulated_mesh, dummy_texture);
  }

  renderer_input.push_back(input);
  this->renderer_.render(renderer_input, camera_params, dummy_image, ancillary_buffers);

  const int min_face_idx = 0;
  const int max_face_idx = F.rows()-1;

  dlib::array2d<vgl_point_3d<float> > face_map(ny,nx);
  dlib::array2d<vgl_point_3d<float> > offsets(ny,nx);
  vil_image_view<float> depth(nx, ny);
  dlib::array2d<GLfloat>& depth_img = this->renderer_.get_last_depth();

  const vgl_point_3d<float> origin(0,0,0);
  const vgl_point_3d<float> bad_pt(NAN, NAN, NAN);
  for (unsigned int y=0; y<ny; ++y) {
    for (unsigned int x=0; x<nx; ++x) {
      // default point to encode background as
      Eigen::RowVector3d base_pt(NAN,NAN,NAN);
      int face_idx = static_cast<int>(face_idx_img[y][x]);
      if ( (face_idx >= min_face_idx) && (face_idx <= max_face_idx) ) {
        vgl_point_3d<float> face_bc(face_bc_img[y][x]);
        Eigen::Vector3i vert_indices = F.row(face_idx);
        base_pt =
          face_bc.x()*V_og.row(vert_indices(0)) +
          face_bc.y()*V_og.row(vert_indices(1)) +
          face_bc.z()*V_og.row(vert_indices(2));
      }
      else {
        img3d[y][x] = bad_pt;
      }
      vgl_point_3d<float> mean_face_pt(base_pt.x(), base_pt.y(), base_pt.z());
      face_map[y][x] = mean_face_pt;
      offsets[y][x] = origin + (img3d[y][x] - mean_face_pt);
      depth(x, y) = depth_img[y][x];
    }
  }
  vgl_box_3d<double> face_map_bbox, img3d_bbox, offsets_bbox;
  if (this->noneck_) {
    face_map_bbox = face3d::noneck_semantic_map_bbox;
    img3d_bbox = face3d::noneck_img3d_bbox;
    offsets_bbox = face3d::default_offsets_bbox;
  }
  else if (this->use_3DMM_) {
    face_map_bbox = face3d::MM_semantic_map_bbox;
    img3d_bbox = face3d::MM_img3d_bbox;
    offsets_bbox = face3d::MM_offsets_bbox;
  }
  else if (this->use_flame_) {
    face_map_bbox = face3d::flame_semantic_map_bbox;
    img3d_bbox = face3d::flame_img3d_bbox;
    offsets_bbox = face3d::flame_offsets_bbox;
  }
  else{
    face_map_bbox = face3d::default_semantic_map_bbox;
    img3d_bbox = face3d::default_img3d_bbox;
    offsets_bbox = face3d::default_offsets_bbox;
  }

  // convert to rgb byte images
  dlib::array2d<dlib::rgb_pixel> face_map_rgb(ny,nx);
  face3d::normalize_3d_image(face_map, face_map_bbox, face_map_rgb);
  dlib::array2d<dlib::rgb_pixel> img3d_rgb(ny,nx);
  face3d::normalize_3d_image(img3d, img3d_bbox, img3d_rgb);
  dlib::array2d<dlib::rgb_pixel> offsets_rgb(ny,nx);
  face3d::normalize_3d_image(offsets, offsets_bbox, offsets_rgb);

  // convert extension to lower case
  for (char &c : extension) { c = std::tolower(c);}
  // save in the desired image format
  std::string PNCC_output_fname =  output_dir + "/" + basename + "_pncc" + extension ;
  std::string img3d_output_fname =  output_dir + "/" + basename + "_img3d" + extension ;
  std::string offsets_output_fname =  output_dir + "/" + basename + "_offsets" + extension ;
  std::string depth_output_fname =  output_dir + "/" + basename + "_depth.tiff";

  vil_save(depth, depth_output_fname.c_str());
  if (extension == ".tiff") {
    face3d::io_utils::save_3d_tiff(face_map, PNCC_output_fname.c_str());
    face3d::io_utils::save_3d_tiff(img3d, img3d_output_fname.c_str());
    face3d::io_utils::save_3d_tiff(offsets, offsets_output_fname.c_str());

  }
  else {
    face3d::io_utils::save_image(face_map_rgb, PNCC_output_fname.c_str());
    face3d::io_utils::save_image(img3d_rgb, img3d_output_fname.c_str());
    face3d::io_utils::save_image(offsets_rgb, offsets_output_fname.c_str());

  }

  if (this->debug_dir_ != "") {
    // save rendering

    std::string render_fname = this->debug_dir_ + "/" + basename + "_render.png";
    dlib::save_png(dummy_image, render_fname.c_str());
    // save face_index image
    std::string face_idx_fname = this->debug_dir_ + "/" + basename + "_face_idx.tiff";
    face3d::io_utils::save_1d_tiff(face_idx_img, face_idx_fname.c_str());
    // save face_bc image
    std::string face_bc_fname = this->debug_dir_ + "/" + basename + "_face_bc.tiff";
    face3d::io_utils::save_3d_tiff(face_bc_img, face_bc_fname.c_str());
  }
  //std::cout << "wrote " << output_fname << std::endl;
}
