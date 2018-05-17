#pragma once
#include <string>
#include <vector>
#include <limits>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_homg_point_2d.h>
#include <face3d_basic/sighting_coefficients.h>
#include <face3d_basic/io_utils.h>
#include <face3d/head_mesh.h>
#include <face3d/mesh_renderer.h>
#include <igl/triangle/triangulate.h>

namespace face3d {

template <class CAM_T, class IMG_T>
class mesh_background_renderer
{
public:
  mesh_background_renderer(IMG_T const& img,
                           sighting_coefficients<CAM_T> const& coeffs,
                           head_mesh const& mesh,
                           vnl_matrix<double> const& subject_components,
                           vnl_matrix<double> const& expression_components,
                           mesh_renderer &renderer);

  mesh_background_renderer(std::vector<IMG_T> const& imgs,
                           std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                           head_mesh const& mesh,
                           vnl_matrix<double> const & subject_components,
                           vnl_matrix<double> const & expression_components,
                           mesh_renderer &renderer);

  template<class CAM_T_OUT>
  void render(sighting_coefficients<CAM_T_OUT> const& render_coeffs,
              dlib::array2d<dlib::rgb_pixel> &background);

  void set_debug_dir(std::string const& debug_dir) { debug_mode_ = true; debug_dir_ = debug_dir; }

private:
  void initialize();

  std::vector<dlib::array2d<dlib::rgb_pixel> > images_;
  std::vector<sighting_coefficients<CAM_T> > coeffs_;

  head_mesh mesh_;
  mesh_renderer &renderer_;
  vnl_matrix<double> const & subject_components_;
  vnl_matrix<double> const & expression_components_;
  // the following members are set by initialize()
  std::vector<Eigen::MatrixXd> UVs_;
  std::vector<Eigen::MatrixXd> Vs_;
  std::vector<Eigen::MatrixXi> Fs_;
  std::vector<std::vector<int> > visible_mesh_verts_;

  bool debug_mode_;
  std::string debug_dir_;

  // helper functions
  void composite_image(dlib::array2d<dlib::rgb_pixel> &dest,
                       dlib::array2d<dlib::rgb_pixel> const& src,
                       float weight);
  void assign_border_locations(Eigen::MatrixXd &V, int border_begin, int nx, int ny,
                               int num_boundary_verts_per_side, bool flip_x);

  void adjust_border_locations(Eigen::MatrixXd &V, Eigen::MatrixXi const& F,
                               int border_begin, int nx, int ny, int num_boundary_verts_per_side);
};


template<class CAM_T, class IMG_T>
mesh_background_renderer<CAM_T, IMG_T>::
mesh_background_renderer(IMG_T const& img,
                         sighting_coefficients<CAM_T> const& coeffs,
                         head_mesh const& mesh,
                         vnl_matrix<double> const& subject_components,
                         vnl_matrix<double> const& expression_components,
                         mesh_renderer &renderer) :
  images_(1), coeffs_(1, coeffs), mesh_(mesh), renderer_(renderer),
  subject_components_(subject_components), expression_components_(expression_components),
  UVs_(1), Vs_(1), Fs_(1), visible_mesh_verts_(1), debug_mode_(false)
{
  // deep copy image
  dlib::assign_image(images_[0], img);

  initialize();
}

template<class CAM_T, class IMG_T>
mesh_background_renderer<CAM_T, IMG_T>::
mesh_background_renderer(std::vector<IMG_T> const& imgs,
                         std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                         head_mesh const& mesh,
                         vnl_matrix<double> const& subject_components,
                         vnl_matrix<double> const& expression_components,
                         mesh_renderer &renderer) :
  images_(imgs.size()), coeffs_(coeffs), mesh_(mesh), renderer_(renderer),
  subject_components_(subject_components), expression_components_(expression_components),
  UVs_(imgs.size()), Vs_(imgs.size()), Fs_(imgs.size()), visible_mesh_verts_(imgs.size()),
  debug_mode_(false)
{
  const int num_imgs = imgs.size();
  if (coeffs.size() != num_imgs) {
    throw std::logic_error("Different number of images and coefficients passed to mesh_background_renderer");
  }
  // deep copy images
  for (int i=0; i<images_.size(); ++i) {
    dlib::assign_image(images_[i], imgs[i]);
  }
  initialize();
}

template<class CAM_T, class IMG_T>
void mesh_background_renderer<CAM_T, IMG_T>::
initialize()
{
  const int num_inputs = images_.size();
  for (int i=0; i < num_inputs; ++i) {
    // project visible mesh vertices into image
    mesh_.apply_coefficients(subject_components_, expression_components_,
                             coeffs_[i].subject_coeffs(), coeffs_[i].expression_coeffs());
    dlib::array2d<dlib::rgb_alpha_pixel> render_img;
    dlib::array2d<vgl_point_3d<float> > render_img3d;
    render_aux_out aux_out;
    aux_out.img3d_ = &render_img3d;
    renderer_.render(mesh_.meshes(), coeffs_[i].camera(), render_img, aux_out);

    auto camera = coeffs_[i].camera().to_camera();
    std::vector<vgl_point_2d<double> > visible_mesh_verts_2d;
    std::vector<vgl_point_3d<double> > mesh_verts_3d;
    visible_mesh_verts_[i].clear();
    mesh_.get_vertices(mesh_verts_3d);
    const int num_verts = mesh_verts_3d.size();
    for (int v=0; v<num_verts; ++v) {
      vgl_point_2d<double> vert2d(camera.project(mesh_verts_3d[v]));
      int c = int(vert2d.x());
      int r = int(vert2d.y());
      if ( (r < 0) || (r >= render_img3d.nr()) || (c < 0) || (c >= render_img3d.nc()) ) {
        // point out of image bounds
        continue;
      }
      vgl_point_3d<float> render_pt3df(render_img3d[r][c]);
      vgl_point_3d<double> render_pt3d(render_pt3df.x(), render_pt3df.y(), render_pt3df.z());
      const double sqr_dist_thresh = 2.0*2.0;
      double sqr_dist = (mesh_verts_3d[v] - render_pt3d).sqr_length();
      if (sqr_dist > sqr_dist_thresh) {
        // point not visible in image
        continue;
      }
      visible_mesh_verts_2d.push_back(vert2d);
      visible_mesh_verts_[i].push_back(v);
    }
    const int num_visible_verts = visible_mesh_verts_[i].size();

    // add border vertices
    const int nx = coeffs_[i].camera().nx();
    const int ny = coeffs_[i].camera().ny();

    const int num_boundary_verts_per_side = 100;
    const int num_boundary_verts = 4*num_boundary_verts_per_side;

    // triangulate
    Eigen::MatrixXd V2d(num_visible_verts + num_boundary_verts,2);
    Eigen::MatrixXi E(num_boundary_verts, 2);
    Eigen::MatrixXd H(1,2); // no holes in triangulation
    H(0,0) = 0; H(0,1) = 0;

    for (int v=0; v<num_visible_verts; ++v) {
      V2d(v,0) = visible_mesh_verts_2d[v].x();
      V2d(v,1) = visible_mesh_verts_2d[v].y();
    }

    assign_border_locations(V2d, num_visible_verts, nx, ny, num_boundary_verts_per_side, false);

    // traverse a closed loop around the edge vertices
    for (int v=0; v<num_boundary_verts; ++v) {
      E(v,0) = num_visible_verts + v;
      E(v,1) = num_visible_verts + ((v + 1) % num_boundary_verts);
    }

    Eigen::MatrixXd V_triangulated;
    std::string triangle_flags ="-Q";
    std::thread::id this_id = std::this_thread::get_id();
    if (debug_mode_)
      triangle_flags = "";
    {
      igl::triangle::triangulate(V2d,E,H, triangle_flags, V_triangulated, Fs_[i]);
    }
    if (V2d.rows() != V_triangulated.rows()) {
      std::cout << "V.rows = " << V2d.rows() << std::endl;
      std::cout << "V_triangulated.rows() = " << V_triangulated.rows() << std::endl;
      throw std::logic_error("Triangulation: Different number of input and output vertices");
    }

    adjust_border_locations(V2d, Fs_[i], num_visible_verts, nx, ny, num_boundary_verts_per_side
                            );

    // texture coordinates are (normalized) image coordinates
    const int num_background_verts = V_triangulated.rows();
    UVs_[i].resize(num_background_verts,2);
    for (int v=0; v<num_background_verts; ++v) {
      UVs_[i](v,0) = V2d(v,0) / nx;
      UVs_[i](v,1) = V2d(v,1) / ny;
    }

    // initialize Vs, setting z value to 0
    Vs_[i].resize(V2d.rows(),3);
    for (int v=0; v<V2d.rows(); ++v) {
      Vs_[i](v,0) = V2d(v,0);
      Vs_[i](v,1) = V2d(v,1);
      Vs_[i](v,2) = 0.0f;
    }
  }
}


template<class CAM_T, class IMG_T>
template<class CAM_T_OUT>
void mesh_background_renderer<CAM_T, IMG_T>::
render(sighting_coefficients<CAM_T_OUT> const& render_coeffs,
       dlib::array2d<dlib::rgb_pixel> &background)
{
  if (images_.size() != coeffs_.size()) {
    std::cerr << "ERROR: render(): images_.size() == " << images_.size() << ", coeffs_size() = " << coeffs_.size() << std::endl;
    throw std::logic_error("Different numbers of images and coefficients");
  }
  const int num_input_images = images_.size();

  auto render_camera = render_coeffs.camera().to_camera();
  vgl_rotation_3d<double> render_R = render_coeffs.camera().rotation();

  const int render_nx = render_coeffs.camera().nx();
  const int render_ny = render_coeffs.camera().ny();
  // initialize background
  background.set_size(render_ny, render_nx);
  dlib::assign_all_pixels(background, dlib::rgb_pixel(0,0,0));

  vnl_matrix_fixed<double,3,3> R_render = render_coeffs.camera().rotation().as_matrix();
  vnl_vector_fixed<double,3> render_camz_vnl = R_render.get_row(2);
  vgl_vector_3d<double> render_cam_z(render_camz_vnl[0], render_camz_vnl[1], render_camz_vnl[2]);
  // compute weights for each of the input images
  std::vector<bool> flip_image(num_input_images, false);
  std::vector<double> image_weights;
  double weight_sum = 0;
  const double softmax_weight = 80.0;
  for (int i=0; i<num_input_images; ++i) {
    CAM_T camera_params(coeffs_[i].camera());
    vnl_matrix_fixed<double,3,3> R_og = camera_params.rotation().as_matrix();
    if (std::signbit(R_og(2,0)) != std::signbit(R_render(2,0))) {
      // mirror the original image
      flip_image[i] = true;
      vnl_matrix_fixed<double,3,3> flip_x_3x3(0.0);
      flip_x_3x3.set_diagonal(vnl_vector_fixed<double,3>(-1,1,1));
      R_og *= flip_x_3x3;
    }
    vnl_vector_fixed<double,3> cam_z_vnl = R_og.get_row(2);
    vgl_vector_3d<double> cam_z(cam_z_vnl[0], cam_z_vnl[1], cam_z_vnl[2]);
    double dp = dot_product(cam_z, render_cam_z);

    if (dp < 1e-3) {
      dp = 1e-3;
    }
    double weight = std::exp(softmax_weight * dp);
    image_weights.push_back(weight);
    weight_sum += weight;
  }
  // add constant term to weight_sum to create fade to black if no nearby images
  // weight_sum += std::exp(softmax_weight*0.1);

  for (double &w : image_weights) {
    // normalize to complete softmax computation
    w /= weight_sum;
  }

  // project vertices using render camera
  mesh_.apply_coefficients(subject_components_, expression_components_,
                           render_coeffs.subject_coeffs(), render_coeffs.expression_coeffs());
  std::vector<vgl_point_3d<double> > mesh_verts_3d;
  mesh_.get_vertices(mesh_verts_3d);


  for (int i=0; i<num_input_images; ++i) {
    if (image_weights[i] <= 0) {
      // don't bother
      continue;
    }

    Eigen::MatrixXd V_render(Vs_[i]);
    const int num_visible_verts = visible_mesh_verts_[i].size();
    const int num_total_verts = V_render.rows();
    float zmin = std::numeric_limits<float>::max();
    float zmax = std::numeric_limits<float>::min();
    for (int v=0; v<num_visible_verts; ++v) {
      int vert_idx = visible_mesh_verts_[i][v];
      vgl_point_3d<double> vert3d(mesh_verts_3d[vert_idx]);
      if (flip_image[i]) {
        vert3d = vgl_point_3d<double>(-vert3d.x(), vert3d.y(), vert3d.z());
      }
      vgl_point_2d<double> vert2d(render_camera.project(vert3d));
      vgl_point_3d<double> vert3d_cam = render_R * vert3d;
      V_render(v,0) = vert2d.x();
      V_render(v,1) = vert2d.y();
      V_render(v,2) = vert3d_cam.z();
      if (vert3d_cam.z() > zmax) {
        zmax = vert3d_cam.z();
      }
      if (vert3d_cam.z() < zmin) {
        zmin = vert3d_cam.z();
      }
    }
    // assign boundary vertex locations
    const int num_boundary_verts_per_side = (num_total_verts - num_visible_verts)/4;
    assign_border_locations(V_render, num_visible_verts, render_nx, render_ny, num_boundary_verts_per_side, flip_image[i]);
    adjust_border_locations(V_render, Fs_[i], num_visible_verts, render_nx, render_ny, num_boundary_verts_per_side);
    // set z value of background vertices to maximum of mesh
    for (int v=num_visible_verts; v<V_render.rows(); ++v) {
      V_render(v,2) = zmax;
    }

    if (debug_mode_) {
      std::stringstream V_fname;
      V_fname << debug_dir_ << "/background_V_" << i << ".txt";
      io_utils::write_t(Vs_[i], V_fname.str());
      std::stringstream Vrender_fname;
      Vrender_fname << debug_dir_ << "/background_V_render_" << i << ".txt";
      io_utils::write_t(V_render, Vrender_fname.str());
      std::stringstream F_fname;
      F_fname << debug_dir_ << "/background_F_" << i << ".txt";
      io_utils::write_t(Fs_[i], F_fname.str());
    }

    // re-render the mesh
    dlib::array2d<dlib::rgb_pixel> img_warped;
    renderer_.render_2d(V_render, Fs_[i], UVs_[i], images_[i], render_nx, render_ny, zmin, zmax, img_warped);
    //std::cout<<"[face3d::mesh_background_renderer:render]: weight "<<image_weights[i]<<std::endl;
    composite_image(background, img_warped, image_weights[i]);
  }
}

template<class CAM_T, class IMG_T>
void mesh_background_renderer<CAM_T, IMG_T>::
composite_image(dlib::array2d<dlib::rgb_pixel> &dest,
                dlib::array2d<dlib::rgb_pixel> const& src,
                float weight)
{
  const int nx = dest.nc();
  const int ny = dest.nr();
  if ((src.nc() != nx) || (src.nr() != ny)) {
    throw std::logic_error("Sizes of dest and src images do not match");
  }
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      dlib::rgb_pixel &pix_out = dest[y][x];
      dlib::rgb_pixel pix_src = src[y][x];
      pix_out.red = static_cast<unsigned char>(std::min(255.0f, pix_out.red + weight*pix_src.red));
      pix_out.green = static_cast<unsigned char>(std::min(255.0f, pix_out.green + weight*pix_src.green));
      pix_out.blue = static_cast<unsigned char>(std::min(255.0f, pix_out.blue + weight*pix_src.blue));
    }
  }
  return;
}

template <class CAM_T, class IMG_T>
void mesh_background_renderer<CAM_T, IMG_T>::
assign_border_locations(Eigen::MatrixXd &V, int border_begin, int nx, int ny,
                        int num_boundary_verts_per_side, bool flip_x)
{
  // sanity check on V's size
  const int num_total_verts = V.rows();
  if (num_total_verts != (border_begin + 4*num_boundary_verts_per_side)) {
    std::cerr << "ERROR: num_total_verts = " << num_total_verts << std::endl;
    std::cerr << "       border_begin = " << border_begin <<std::endl;
    std::cerr << "       num_boundary_verts_per_side = " << num_boundary_verts_per_side << std::endl;
    throw std::logic_error("Vertex matrix of wrong size passed to assign_border_locations()");
  }
  // add boundary vertices
  // top
  int idx_off = border_begin;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    V(idx_off + v, 0) = static_cast<float>(v)/num_boundary_verts_per_side * nx;
    V(idx_off + v, 1) = 0;
  }
  // right
  idx_off = border_begin + num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    V(idx_off + v, 0) = nx;
    V(idx_off + v, 1) = static_cast<float>(v)/num_boundary_verts_per_side * ny;
  }
  // bottom
  idx_off = border_begin + 2*num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    V(idx_off + v, 0) = static_cast<float>(num_boundary_verts_per_side - v)/num_boundary_verts_per_side * nx;
    V(idx_off + v, 1) = ny;
  }
  // left
  idx_off = border_begin + 3*num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    V(idx_off + v, 0) = 0;
    V(idx_off + v, 1) = static_cast<float>(num_boundary_verts_per_side - v)/num_boundary_verts_per_side * ny;
  }
  if (flip_x) {
    for (int v=border_begin; v<num_total_verts; ++v) {
      V(v,0) = nx - V(v,0);
    }
  }
}

template <class CAM_T, class IMG_T>
void mesh_background_renderer<CAM_T, IMG_T>::
adjust_border_locations(Eigen::MatrixXd &V, Eigen::MatrixXi const& F,
                        int border_begin, int nx, int ny, int num_boundary_verts_per_side)
{
  // sanity check on V's size
  const int num_total_verts = V.rows();
  if (num_total_verts != (border_begin + 4*num_boundary_verts_per_side)) {
    std::cerr << "ERROR: num_total_verts = " << num_total_verts << std::endl;
    std::cerr << "       border_begin = " << border_begin <<std::endl;
    std::cerr << "       num_boundary_verts_per_side = " << num_boundary_verts_per_side << std::endl;
    throw std::logic_error("Vertex matrix of wrong size passed to adjust_border_locations()");
  }
  // sum total "forces" on each vertex
  Eigen::MatrixXd V_neighbor_means = Eigen::MatrixXd::Zero(num_total_verts, V.cols());
  std::vector<int> num_neighbors(num_total_verts, 0);

  const int num_faces = F.rows();
  for (int f=0; f<num_faces; ++f) {
    int i0 = F(f,0);
    int i1 = F(f,1);
    int i2 = F(f,2);
    auto v0 = V.row(i0);
    auto v1 = V.row(i1);
    auto v2 = V.row(i2);
    // 0 <-> 1
    if (i1 < border_begin) {
      V_neighbor_means.row(i0) += v1;
      ++num_neighbors[i0];
    }
    if (i0 < border_begin) {
      V_neighbor_means.row(i1) += v0;
      ++num_neighbors[i1];
    }
    // 1 <-> 2
    if (i2 < border_begin) {
      V_neighbor_means.row(i1) += v2;
      ++num_neighbors[i1];
    }
    if (i1 < border_begin) {
      V_neighbor_means.row(i2) += v1;
      ++num_neighbors[i2];
    }
    // 2 <-> 0
    if (i0 < border_begin) {
      V_neighbor_means.row(i2) += v0;
      ++num_neighbors[i2];
    }
    if (i2 < border_begin) {
      V_neighbor_means.row(i0) += v2;
      ++num_neighbors[i0];
    }
  }
  // adjust boundary positions according to mean neighbor positions
  // top
  const float anchor_weight = 0.01f;
  int idx_off = border_begin;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    // move in x direction only
    double mean_loc = (V(idx_off + v, 0) * anchor_weight + V_neighbor_means(idx_off + v,0)) / (anchor_weight + num_neighbors[idx_off + v]);
    V(idx_off + v, 0) = mean_loc;
  }
  // right
  idx_off = border_begin + num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    // move in y direction only
    double mean_loc = (V(idx_off + v, 1) * anchor_weight + V_neighbor_means(idx_off + v, 1)) / (anchor_weight + num_neighbors[idx_off + v]);
    V(idx_off + v, 1) = mean_loc;
  }
  // bottom
  idx_off = border_begin + 2*num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    // move in x direction only
    double mean_loc = (V(idx_off + v, 0) * anchor_weight + V_neighbor_means(idx_off + v, 0)) / (anchor_weight + num_neighbors[idx_off + v]);
    V(idx_off + v, 0) = mean_loc;
  }
  // left
  idx_off = border_begin + 3*num_boundary_verts_per_side;
  for (int v=0; v<num_boundary_verts_per_side; ++v) {
    // move in y direction only
    double mean_loc = (V(idx_off + v, 1) * anchor_weight + V_neighbor_means(idx_off + v, 1)) / (anchor_weight + num_neighbors[idx_off + v]);
    V(idx_off + v, 1) = mean_loc;
  }
}

}
