#pragma once
#include <string>
#include <vector>
#include <cmath>

#include "head_mesh.h"
#include "semantic_map.h"
#include "camera_estimation.h"

#include <vnl/vnl_matrix.h>

#include <face3d_basic/io_utils.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <igl/AABB.h>
#include "estimation_results.h"

#include <Eigen/Dense>

#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/cpp17/optional.hpp"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
namespace fs = boost::filesystem;

namespace face3d {

class media_coefficient_from_PNCC_and_offset_estimator_eos
{
public:
  media_coefficient_from_PNCC_and_offset_estimator_eos(head_mesh const& mesh,
                                                       std::string model_filename,
                                                       std::string blendshapes_filename,
                                                       bool debug_mode, std::string debug_dir,
                                                       double fixed_focal_len=-1.0);

  template<class CAM_T>
  estimation_results_t estimate_coefficients(std::string const& img_ids,
                                             dlib::array2d<vgl_point_3d<float> >  const& semantic_maps,
                                             dlib::array2d<vgl_vector_3d<float> >  const& offsets,
                                             subject_sighting_coefficients<CAM_T> &results,
                                             std::string image_fname,
                                             std::string matrix_fp);
  void
  pretty_print_eos_params(eos::fitting::ScaledOrthoProjectionParameters& params){
    std::cout<<"scale:"<<params.s<<std::endl;
    std::cout<<"translation:"<<params.tx<<" "<<params.ty<<std::endl;
    std::cout<<"Rotation:"<<std::endl;
    std::cout<<std::setw(3)<<params.R[0][0]<<" "<< params.R[1][0]<<" "<<params.R[2][0]<<std::endl;
    std::cout<<std::setw(3)<<params.R[0][1]<<" "<< params.R[1][1]<<" "<<params.R[2][1]<<std::endl;
    std::cout<<std::setw(3)<<params.R[0][2]<<" "<< params.R[1][2]<<" "<<params.R[2][2]<<std::endl;
  }
  Eigen::Matrix4f load_inverse_alignment_matrix(std::string fpath){
    if (fpath=="")
      std::cout<<"Could not find rigid body transformation at "<<fpath<<" ! This is required for fitting to work!"<<std::endl;
    std::ifstream input_fs(fpath);
    Eigen::Matrix4f output_mat;
    std::vector<float> raw_data;
    if (!input_fs){
      std::cout<<"Could not open "<<fpath<<std::endl;
      return output_mat;
    }
    while (input_fs){
      std::string line, cell;
      if (!std::getline(input_fs, line)){
        std::cout<<"Error getting line!"<<std::endl;
        break;
      }
      std::vector<std::string> line_items;
      std::stringstream line_stream(line);
      std::cout<<line<<std::endl;
      while (std::getline(line_stream, cell, ' ')){
        raw_data.push_back(std::stod(cell));
      }
    }
    assert(raw_data.size() == 16);
    float initializer[16];
    for (unsigned i=0; i < 16; i++){
      initializer[i] = raw_data[i];
    }
    output_mat = Eigen::Matrix4f(initializer);
    output_mat.resize(4,4);
    output_mat.transposeInPlace();
    std::cout<<output_mat<<std::endl;
    return output_mat;
  }
private:
  head_mesh base_mesh_;
  vnl_matrix<double> subject_pca_components_;
  vnl_matrix<double> expression_pca_components_;
  vnl_matrix<double> subject_pca_ranges_;
  vnl_matrix<double> expression_pca_ranges_;

  triangle_mesh mean_face_mesh_;
  igl::AABB<triangle_mesh::VTYPE, 3> mean_face_mesh_tree_;
  std:: string modelfile_, blendshapesfile_;
  bool debug_mode_;
  std::string debug_dir_;
  double fixed_focal_len_;
};
}

template<class CAM_T>
face3d::estimation_results_t face3d::media_coefficient_from_PNCC_and_offset_estimator_eos::
estimate_coefficients(std::string const& img_ids,
                      dlib::array2d<vgl_point_3d<float> >  const& semantic_maps,
                      dlib::array2d<vgl_vector_3d<float> >  const& offsets,
                      subject_sighting_coefficients<CAM_T> &results,
                      std::string image_fname, // image filepath for debugging purposes
                      std::string matrix_fp) // path to rigid body transformation that brings the Surrey mesh into the Basel space
{

  std::vector<CAM_T> all_cam_params;

  const int num_images = 1;
  int total_num_data = 0;
  eos::morphablemodel::MorphableModel morphable_model;
  try
    {
      morphable_model = eos::morphablemodel::load_model(this->modelfile_);
    } catch (const std::runtime_error& e)
    {
      std::cout << "Error loading the Morphable Model: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  const std::vector<eos::morphablemodel::Blendshape> blendshapes = eos::morphablemodel::load_blendshapes(this->blendshapesfile_);

  eos::morphablemodel::MorphableModel morphable_model_with_expressions(morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), eos::cpp17::nullopt,
                                                                       morphable_model.get_texture_coordinates());

  eos::core::LandmarkMapper landmark_mapper;
  // keep track of all used mesh vertices. This vector will remain empty if not in debug mode.
  std::vector<std::vector<int> > vertex_indices_dbg(num_images);
  estimation_results_t retval(1);

  // for each image
  // get the image locations of the mean face vertices via the PNCC
  std::map<int, vgl_point_2d<double> > vertex_projection_map;
  extract_vertex_projections(semantic_maps, mean_face_mesh_, vertex_projection_map, mean_face_mesh_tree_);
  retval.vertices_found_[0] = vertex_projection_map.size();
  std::cout << "Found " << retval.vertices_found_[0] << " out of " << mean_face_mesh_.num_vertices() << " vertex projections." << std::endl;
  Eigen::Matrix4f transformation_mat = this->load_inverse_alignment_matrix(matrix_fp);
  // fill in lists containing per-vertex information
  std::vector<vgl_point_2d<double> > vertex_projections;
  std::vector<int> vertex_indices;
  std::vector<vgl_vector_3d<double> > vertex_offsets;
  std::vector<vgl_point_3d<double> > vertex_locations;
  std::vector<Eigen::Vector2f> image_locations;
  std::vector<Eigen::Vector4f> head_mesh_locations;
  for (auto vp : vertex_projection_map) {
    vertex_projections.push_back(vp.second);
    vertex_indices.push_back(vp.first);
    vgl_point_3d<double> base_mesh_vert(mean_face_mesh_.vertex(vp.first));
    // TODO: interpolation
    int pix_r = static_cast<int>(std::round(vp.second.y()));
    int pix_c = static_cast<int>(std::round(vp.second.x()));
    vgl_vector_3d<float> offset_f = offsets[pix_r][pix_c];
    vgl_vector_3d<double> offset(offset_f.x(), offset_f.y(), offset_f.z());
    vgl_point_3d<double> mesh_coordinate = base_mesh_vert + offset;
    Eigen::Vector4f warped_coordinate(mesh_coordinate.x(), mesh_coordinate.y(), mesh_coordinate.z(), 1.0f);
    head_mesh_locations.emplace_back(transformation_mat * warped_coordinate); // warp to Surrey model space
    image_locations.emplace_back(Eigen::Vector2f(vp.second.x(), vp.second.y()));

    vertex_offsets.push_back(offset);
    vertex_locations.push_back(mesh_coordinate);
  }

  if (debug_mode_) {
    std::string vert_projections_fname = debug_dir_ + "/vertex_projections_" + std::to_string(0) + ".txt";
    face3d::io_utils::write_points(vertex_projections, vert_projections_fname);
  }
  const int min_corrs = 10;
  if (vertex_projections.size() < min_corrs) {
    std::cerr << "ERROR: found " << vertex_projections.size() << " correspondences, need " << min_corrs << std::endl;
    return retval;
  }

  // compute camera parameters for this image
  CAM_T cam_params;
  const int nx = semantic_maps.nc();
  const int ny = semantic_maps.nr();
  if (fixed_focal_len_ > 0) {
    vgl_point_2d<double> principal_pt(static_cast<double>(nx)/2, static_cast<double>(ny)/2);
    camera_estimation::compute_camera_params(vertex_projections, vertex_locations, nx, ny, fixed_focal_len_, principal_pt, cam_params);
  }
  else{
    camera_estimation::compute_camera_params(vertex_projections, vertex_locations, nx, ny, cam_params);
  }
  all_cam_params.push_back(cam_params);
  cam_params.pretty_print(std::cout);
  std::cout<<"OG Rotation"<<std::endl;
  std::cout<<cam_params.rotation().as_matrix()<<std::endl;

  const int num_data = vertex_projections.size();
  float lambda = 30.0f;
  std::vector<float> subject_coeffs_raw, expression_coeffs_raw;
  eos::fitting::ScaledOrthoProjectionParameters pose =
    eos::fitting::estimate_orthographic_projection_linear(image_locations, head_mesh_locations, true, ny);
  eos::fitting::RenderingParameters rendering_params(pose, nx, ny);
  std::cout<<"Estimated EOS rendering params"<<std::endl;
  this->pretty_print_eos_params(pose);
  Eigen::Matrix<float, 3, 4> affine_from_ortho =
    eos::fitting::get_3x4_affine_camera_matrix(rendering_params, nx, ny);
  Eigen::VectorXf result_shape = eos::fitting::fit_shape(affine_from_ortho,
                                                         morphable_model_with_expressions,
                                                         blendshapes,
                                                         image_locations,
                                                         vertex_indices, lambda,
                                                         eos::cpp17::nullopt,
                                                         subject_coeffs_raw,
                                                         expression_coeffs_raw);

  const int num_subject_components = subject_coeffs_raw.size();
  const int num_expression_components = expression_coeffs_raw.size();

  vnl_vector<double> subject_coeffs(num_subject_components, 0.0);

  for (int i=0; i<num_subject_components; ++i) {
    subject_coeffs[i] = subject_coeffs_raw[i];
  }

  std::vector<vnl_vector<double> > expression_coeffs;
  for (int n=0; n<num_images; ++n) {
    vnl_vector<double> these_expression_coeffs(num_expression_components,0.0);
    for (int i=0; i<num_expression_components; ++i) {
      these_expression_coeffs[i] = expression_coeffs_raw[i];
    }
    expression_coeffs.push_back(these_expression_coeffs);
  }
  std::vector<std::string> img_ids_vec(1); img_ids_vec[0] = img_ids;
  results = face3d::subject_sighting_coefficients<CAM_T>(subject_coeffs,
                                                         img_ids_vec,
                                                         expression_coeffs,
                                                         all_cam_params);
  retval.success_ = true;
  if (image_fname != ""){
    cv::Mat image = cv::imread(image_fname);
    cv::Mat outimg = image.clone();

    eos::core::Mesh  result_mesh_expressions = eos::morphablemodel::sample_to_mesh(
                                                                                   result_shape, morphable_model_with_expressions.get_color_model().get_mean(),
                                                                                   morphable_model_with_expressions.get_shape_model().get_triangle_list(),
                                                                                   morphable_model_with_expressions.get_color_model().get_triangle_list(), morphable_model_with_expressions.get_texture_coordinates(),
                                                                                   morphable_model_with_expressions.get_texture_triangle_indices());

    eos::core::Mesh warped_mesh(result_mesh_expressions);
    Eigen::Matrix4f transform_inverse = transformation_mat.inverse(); // need to warp back to Basel Face Model space
    for (auto & vertex : warped_mesh.vertices ){
      Eigen::Vector4f vertex_tmp(vertex.x(), vertex.y(), vertex.z(), 1.0f);
      vertex_tmp =  transform_inverse * vertex_tmp;
      vertex = Eigen::Vector3f(vertex_tmp.x(), vertex_tmp.y(), vertex_tmp.z());
    }
    bool export_expressions = true;
    const eos::core::Mesh result_mesh =  result_mesh_expressions;
    const eos::core::Image4u isomap =
      eos::render::extract_texture(result_mesh, affine_from_ortho, eos::core::from_mat(image), true);

    // Draw the fitted mesh as wireframe, and save the image:
    eos::render::draw_wireframe(outimg, result_mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                                eos::fitting::get_opencv_viewport(image.cols, image.rows));
    fs::path outputfile = this->debug_dir_ + "/output.png";
    cv::imwrite(outputfile.string(), outimg);
    //Draw the landmarks
    cv::Mat outimg_lnd = image.clone();
    for (auto&& lm : image_locations)
      {
        cv::rectangle(outimg_lnd, cv::Point2f(lm.x() - 2.0f, lm.y() - 2.0f),
                      cv::Point2f(lm.x() + 2.0f, lm.y() + 2.0f), {255, 0, 0});
      }
    fs::path outputfile_lnd = this->debug_dir_ + "/output_landmarks.png";
    cv::imwrite(outputfile_lnd.string(), outimg_lnd);
    // Save the mesh as textured obj:
    outputfile.replace_extension(".obj");
    eos::core::write_textured_obj(warped_mesh, outputfile.string());

    // And save the isomap:
    outputfile.replace_extension(".isomap.png");
    cv::imwrite(outputfile.string(), eos::core::to_mat(isomap));

    std::cout << "Finished fitting and wrote result mesh and isomap to files with basename "
              << outputfile.stem().stem() << "." << std::endl;
  }
  return retval;
}
