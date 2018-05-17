#pragma once

#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>

#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_box_3d.h>
#include <vil/vil_image_view.h>
#include <vul/vul_file.h>

#include "triangle_mesh.h"
#include <face3d_basic/io_utils.h>

#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>

#include <unordered_map>
#include <utility>

#ifdef FACE3D_USE_CUDA
#include "face3d_cuda_context.h"
#endif

namespace face3d
{

extern const vgl_box_3d<double> default_semantic_map_bbox;
extern const vgl_box_3d<double> default_img3d_bbox;
extern const vgl_box_3d<double> noneck_semantic_map_bbox;
extern const vgl_box_3d<double> noneck_img3d_bbox;
extern const vgl_box_3d<double> MM_semantic_map_bbox;
extern const vgl_box_3d<double> MM_img3d_bbox;
extern const vgl_box_3d<double> default_offsets_bbox;
extern const vgl_box_3d<double> MM_offsets_bbox;

void normalize_3d_image(vil_image_view<float> const& in, vgl_box_3d<double> const& bbox,
                        vil_image_view<unsigned char> &out);

void unnormalize_3d_image(vil_image_view<unsigned char> const& in, vgl_box_3d<double> const& bbox,
                              vil_image_view<float> &out);

template <class T>
void normalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                            dlib::array2d<dlib::rgb_pixel> &out);

template <class T>
void unnormalize_3d_image(dlib::array2d<dlib::rgb_pixel> const& in, vgl_box_3d<double> const& bbox,
                              dlib::array2d<T> &out);

// puts output in range -1,1
template <class T>
void normalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                        dlib::array2d<T> &out);

// Assumes the input image has range -1,1
template <class T>
void unnormalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                          dlib::array2d<T> &out);

template<class T>
void load_3d_image(dlib::array2d<T> &img3d,
                   vgl_box_3d<double> const& normalizing_bbox,
                   std::string const& filename);

template<class T>
void save_3d_image(dlib::array2d<T> const& img3d,
                   vgl_box_3d<double> const& normalizing_bbox,
                   std::string const& filename);

template<class T>
void localize_points(dlib::array2d<vgl_point_3d<float> > const& img,
                     dlib::array2d<unsigned char> const& mask,
                     triangle_mesh const& mesh,
                     std::vector<vgl_point_2d<T> > &locs,
                     const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree,
                     float dist_thresh=1.0);

template<class T>
void localize_points(dlib::array2d<vgl_point_3d<float> > const& img,
                     triangle_mesh const& mesh,
                     std::vector<vgl_point_2d<T> > &locs,
                     const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree,
                     float dist_thresh=1.0);

template<class T1, class T2>
void extract_vertex_projections( dlib::array2d<vgl_point_3d<T1> > const& semantic_map,
                                 triangle_mesh const& mesh,
                                 std::map<int, vgl_point_2d<T2> > &vertex_projection_map,
                                 const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree);

#ifdef FACE3D_USE_CUDA
void localize_points_cuda(MatrixXfRowMajor& query_points,
                          MatrixXfRowMajor& query_points_xy,
                          MatrixXfRowMajor& V,
                          MatrixXfRowMajor& locs_eigen,
                          MatrixXfRowMajor& dists,
                          int cuda_device);

#endif
};

// --- Template Definitions ---- //

#ifdef FACE3D_USE_CUDA
template<class T>
void face3d::
localize_points(dlib::array2d<vgl_point_3d<float> > const& img,
                dlib::array2d<unsigned char> const& mask,
                triangle_mesh const& mesh,
                std::vector<vgl_point_2d<T> > &locs,
                const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree,
                float dist_thresh) {
  const int cuda_device = face3d::get_cuda_device();
  std::cout << "Running the GPU version: cuda_device = " << cuda_device << std::endl;
  const int num_vertices = mesh.num_vertices();
  const int nx = img.nc();
  const int ny = img.nr();

  const vgl_point_2d<T> invalid_pt(NAN, NAN);
  locs.assign(num_vertices, invalid_pt);

  MatrixXfRowMajor query_points = MatrixXfRowMajor::Constant(nx*ny, 3, NAN);
  MatrixXfRowMajor query_points_xy = MatrixXfRowMajor::Constant(nx*ny, 2, NAN);
  int num_good_pixels = 0;
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      // only consider pixels with mask turned on
      if (mask[y][x] == dlib::on_pixel) {
        query_points_xy(num_good_pixels, 0) = x;
        query_points_xy(num_good_pixels, 1) = y;
        query_points(num_good_pixels, 0) = img[y][x].x();
        query_points(num_good_pixels, 1) = img[y][x].y();
        query_points(num_good_pixels, 2) = img[y][x].z();
        num_good_pixels++;
      }
    }
  }
  query_points.conservativeResize(num_good_pixels, 3);
  query_points_xy.conservativeResize(num_good_pixels, 2);

  double dist_sqrd_thresh = dist_thresh*dist_thresh;

  MatrixXdRowMajor V_double = mesh.V();
  MatrixXfRowMajor V = MatrixXfRowMajor::Zero(V_double.rows(), V_double.cols());

  for (int i = 0; i < V.rows(); i++) {
    for (int j = 0; j < V.cols(); j++) {
      V(i, j) = (float)V_double(i,j);
    }
  }

  MatrixXfRowMajor locs_eigen = MatrixXfRowMajor::Zero(num_vertices, 2);
  MatrixXfRowMajor dists(num_vertices, 1);

  // Localize vertex points on the image using the GPU
  localize_points_cuda(query_points, query_points_xy, V, locs_eigen, dists, cuda_device);

  int matched_points = 0;
  std::unordered_map<int, std::pair<int, double> > vertex_pixel_map;
  for (int i = 0; i < locs_eigen.rows(); i++) {
    if (dists(i, 0) < dist_sqrd_thresh) {
      locs[i] = vgl_point_2d<T>(locs_eigen(i,0), locs_eigen(i,1));
      vertex_pixel_map[i] = std::pair<int, double>(locs_eigen(i,0) * nx + locs_eigen(i,1),  dists(i,0));
      matched_points++;
    }
  }

  //Clean up double-pointers
  std::unordered_map<int, std::pair<int, double> > pixel_vertex_map;
  for (const auto &kvpair : vertex_pixel_map) {
    int vertex = kvpair.first;
    int pixel = kvpair.second.first;
    double distance = kvpair.second.second;
    if (pixel_vertex_map.find(pixel) == pixel_vertex_map.end()) {
      pixel_vertex_map[pixel] = std::pair<int, double>(vertex, distance);
    } else {
      std::pair<int, double> prev_vertex_and_distance = pixel_vertex_map[pixel];
      if (distance < prev_vertex_and_distance.second) {
        pixel_vertex_map[pixel] = std::pair<int, double>(vertex , distance);
      }
    }
  }

  int pixel, pixel_x, pixel_y, map_vertex;
  for (int i = 0; i < locs_eigen.rows(); i++) {
    pixel_x = locs_eigen(i, 0); pixel_y = locs_eigen(i, 1);
    pixel = pixel_x * nx + pixel_y;
    map_vertex = pixel_vertex_map[pixel].first;
    if (map_vertex != i) {
      locs[i] = invalid_pt;
      matched_points--;
    }
  }
  return;
}
#else
template<class T>
void face3d::
localize_points(dlib::array2d<vgl_point_3d<float> > const& img,
                dlib::array2d<unsigned char> const& mask,
                triangle_mesh const& mesh,
                std::vector<vgl_point_2d<T> > &locs,
                const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree,
                float dist_thresh)
{
  const int num_pts = mesh.num_vertices();
  const int nx = img.nc();
  const int ny = img.nr();
  std::chrono::steady_clock::time_point begin, end;

  // initialize 2d locations
  const vgl_point_2d<T> invalid_pt(NAN,NAN);
  locs.assign(num_pts, invalid_pt);

  std::vector<vgl_point_3d<T> > query_point_list;
  std::vector<vgl_point_2d<T> > query_points_2d;
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      vgl_point_3d<T> pt3d(img[y][x].x(), img[y][x].y(), img[y][x].z());
      // only consider pixels with mask turned on
      if (mask[y][x] == dlib::on_pixel) {
        query_point_list.push_back(pt3d);
        query_points_2d.push_back(vgl_point_2d<T>(x,y));
      }
    }
  }

  int num_good_pixels = query_point_list.size();
  Eigen::MatrixXd query_points(num_good_pixels,3);
  for (int i=0; i<num_good_pixels; ++i) {
      query_points(i,0) = query_point_list[i].x();
      query_points(i,1) = query_point_list[i].y();
      query_points(i,2) = query_point_list[i].z();
  }

  // search for best vertex at each pixel
  const T dist_sqrd_thresh = dist_thresh*dist_thresh;

  Eigen::MatrixXd query_dists(num_good_pixels,1);
  Eigen::MatrixXi face_indices(num_good_pixels,1);
  Eigen::MatrixXd closest_points(num_good_pixels,3);
  std::vector<T> best_sqr_dists(num_pts, dist_sqrd_thresh);

  mesh_tree.squared_distance(mesh.V(), mesh.F(),
                        query_points, query_dists, face_indices, closest_points);

  if (face_indices.rows() != num_good_pixels) {
    throw std::logic_error("squared_distance() returned matrix with unexpected number of rows");
  }
  int matched_points = 0;
  for (int i=0; i<num_good_pixels; ++i) {
    vgl_point_3d<T> closest_pt(closest_points(i,0), closest_points(i,1), closest_points(i,2));
    if (query_dists(i) <= dist_sqrd_thresh) {
      // get the points on the closest face
      Eigen::MatrixXd barycentric_coords;
      Eigen::MatrixXi vert_indices = mesh.F().row(face_indices(i));
      Eigen::MatrixXd query_point = query_points.row(i);

      // compute the barycentric coordinates of the query point wrt the mesh face
      igl::barycentric_coordinates(query_point,
                                   Eigen::MatrixXd(mesh.V().row(vert_indices(0))),
                                   Eigen::MatrixXd(mesh.V().row(vert_indices(1))),
                                   Eigen::MatrixXd(mesh.V().row(vert_indices(2))),
                                   barycentric_coords);
      // find closest vertex
      int vert_idx = -1;
      if (barycentric_coords(0) > barycentric_coords(1)) {
        // 0 or 2
        if (barycentric_coords(0) > barycentric_coords(2)) {
          vert_idx = vert_indices(0);
        }
        else {
          vert_idx = vert_indices(2);
        }
      }
      else {
        // 1 or 2
        if (barycentric_coords(1) > barycentric_coords(2)) {
          vert_idx = vert_indices(1);
        }
        else {
          vert_idx = vert_indices(2);
        }
      }
      vgl_point_3d<T> vert(mesh.V()(vert_idx,0),
                           mesh.V()(vert_idx,1),
                           mesh.V()(vert_idx,2));
      vgl_point_3d<T> pt3d(query_point(0), query_point(1), query_point(2));
      double vert_sqr_dist = (pt3d - vert).sqr_length();
      if (vert_sqr_dist <= best_sqr_dists[vert_idx]) {
        matched_points++;
        best_sqr_dists[vert_idx] = vert_sqr_dist;
        locs[vert_idx] = query_points_2d[i];
      }
    }
  }
}
#endif

template<class T>
void face3d::
localize_points(dlib::array2d<vgl_point_3d<float> > const& img,
                triangle_mesh const& mesh,
                std::vector<vgl_point_2d<T> > &locs,
                const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree,
                float dist_thresh) {
  dlib::array2d<unsigned char> mask(img.nr(), img.nc());
  dlib::assign_all_pixels(mask, dlib::on_pixel);
  localize_points(img, mask, mesh, locs, mesh_tree, dist_thresh);
}

template<class T1, class T2>
void face3d::extract_vertex_projections( dlib::array2d<vgl_point_3d<T1> > const& semantic_map,
                                         triangle_mesh const& mesh,
                                         std::map<int, vgl_point_2d<T2> > &vertex_projection_map,
                                         const igl::AABB<triangle_mesh::VTYPE, 3>& mesh_tree
                                         )
{
  // create mask of valid pixel locations
  const int nx = semantic_map.nc();
  const int ny = semantic_map.nr();
  dlib::array2d<unsigned char> mask(ny, nx);
  // invalid points are represented as (0,0,0)
  const vgl_point_3d<T1> invalid_loc(0,0,0);
  const T1 squared_len_thresh = 10.0*10.0;
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      if (std::isnan(semantic_map[y][x].x())) {
        mask[y][x] = dlib::off_pixel;
      }
      else {
        if ((semantic_map[y][x] - invalid_loc).sqr_length() < squared_len_thresh) {
          mask[y][x] = dlib::off_pixel;
        }
        else {
          mask[y][x] = dlib::on_pixel;
        }
      }
    }
  }
  // erode the mask to avoid problematic areas around edges
  dlib::array2d<unsigned char> mask_eroded(ny,nx);
  static const int strel_size = 3;
  unsigned char strel[strel_size][strel_size];
  for (int y=0; y<strel_size; ++y) {
    for (int x=0; x<strel_size; ++x) {
      strel[y][x] = dlib::on_pixel;
    }
  }
  dlib::binary_erosion(mask, mask_eroded, strel);

  const float dist_thresh = 4.0f;
  std::vector<vgl_point_2d<float> > vert_locs;
  localize_points(semantic_map, mask_eroded, mesh, vert_locs, mesh_tree, dist_thresh);
  if (vert_locs.size() != mesh.num_vertices()) {
    throw std::logic_error("localize_points() returned unexpected number of vertex locations");
  }
  for (int v=0; v<vert_locs.size(); ++v) {
    if (!(std::isnan(vert_locs[v].x()))) {
      vertex_projection_map[v] = vgl_point_2d<double>(vert_locs[v].x(),
                                                      vert_locs[v].y());
    }
  }
  return;
}

template <class T>
void face3d::load_3d_image(dlib::array2d<T> &image,
                           vgl_box_3d<double> const& normalizing_bbox,
                           std::string const& filename)
{
  std::string ext = vul_file::extension(filename);
  for (char &c : ext){ c = std::tolower(c); }
  if ((ext == ".tiff") || (ext == ".tif")) {
    // assume the image is floating point (unnormalized)
    face3d::io_utils::load_3d_tiff(image, filename);
  }
  else {
    // assume the image is encoded in a rgb byte image
    dlib::array2d<dlib::rgb_pixel> img_rgb;
    dlib::load_image(img_rgb, filename.c_str());
    face3d::unnormalize_3d_image(img_rgb, normalizing_bbox, image);
  }
}

template <class T>
void face3d::save_3d_image(dlib::array2d<T> const& image,
                           vgl_box_3d<double> const& normalizing_bbox,
                           std::string const& filename)
{
  std::string ext = vul_file::extension(filename);
  for (char &c : ext){ c = std::tolower(c); }
  if ((ext == ".tiff") || (ext == ".tif")) {
    // save the image as float, unnormalized
    face3d::io_utils::save_3d_tiff(image, filename);
  }
  else {
    // encode the image in a rgb byte image
    dlib::array2d<dlib::rgb_pixel> img_rgb;
    face3d::normalize_3d_image(image, normalizing_bbox, img_rgb);
    face3d::io_utils::save_image(img_rgb, filename);
  }
}

template <class T>
void face3d::
normalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                   dlib::array2d<dlib::rgb_pixel> &out)
{
  const int nx = in.nc();
  const int ny = in.nr();
  out.set_size(ny,nx);

  float min_x = bbox.min_x();
  float x_range = bbox.max_x() - bbox.min_x();
  float min_y = bbox.min_y();
  float y_range = bbox.max_y() - bbox.min_y();
  float min_z = bbox.min_z();
  float z_range = bbox.max_z() - bbox.min_z();

  for (unsigned int yi=0; yi<ny; ++yi) {
    for (unsigned int xi=0; xi<nx; ++xi) {
      T source_pt = in[yi][xi];
      if (std::isnan(source_pt.x()) || std::isnan(source_pt.y()) || std::isnan(source_pt.z())) {
        source_pt = T(0,0,0);
      }
      dlib::rgb_pixel out_pix;
      out_pix.red = static_cast<unsigned char>(std::max(0.0f,std::min(255.0f,(source_pt.x() - min_x)/x_range * 255)));
      out_pix.green = static_cast<unsigned char>(std::max(0.0f,std::min(255.0f,(source_pt.y() - min_y)/y_range * 255)));
      out_pix.blue= static_cast<unsigned char>(std::max(0.0f,std::min(255.0f,(source_pt.z() - min_z)/z_range * 255)));
      out[yi][xi] = out_pix;
    }
  }
  return;
}

// pixels in out are in range -1,1
template <class T>
void face3d::
normalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                   dlib::array2d<T> &out)
{
  const int nx = in.nc();
  const int ny = in.nr();
  out.set_size(ny,nx);

  float scale_x = 2.0/(bbox.max_x() - bbox.min_x());
  float off_x = -bbox.min_x()*scale_x - 1;
  float scale_y = 2.0/(bbox.max_y() - bbox.min_y());
  float off_y = -bbox.min_y()*scale_y - 1;
  float scale_z = 2.0/(bbox.max_z() - bbox.min_z());
  float off_z = -bbox.min_z()*scale_z - 1;

  for (unsigned int yi=0; yi<ny; ++yi) {
    for (unsigned int xi=0; xi<nx; ++xi) {
      T source_pt = in[yi][xi];
      if (std::isnan(source_pt.x()) || std::isnan(source_pt.y()) || std::isnan(source_pt.z())) {
        source_pt = T(0,0,0);
      }
      float x = scale_x * source_pt.x() + off_x;
      float y = scale_y * source_pt.y() + off_y;
      float z = scale_z * source_pt.z() + off_z;
      out[yi][xi] = T(x,y,z);
    }
  }
  return;
}

template <class T>
void face3d::
unnormalize_3d_image(dlib::array2d<T> const& in, vgl_box_3d<double> const& bbox,
                     dlib::array2d<T> &out)
{
  const int nx = in.nc();
  const int ny = in.nr();
  out.set_size(ny,nx);

  float scale_x = (bbox.max_x() - bbox.min_x())/2.0;
  float offset_x = bbox.centroid_x();
  float scale_y = (bbox.max_y() - bbox.min_y())/2.0;
  float offset_y = bbox.centroid_y();
  float scale_z = (bbox.max_z() - bbox.min_z())/2.0;
  float offset_z = bbox.centroid_z();

  for (unsigned int yi=0; yi<ny; ++yi) {
    for (unsigned int xi=0; xi<nx; ++xi) {
      T pix_in = in[yi][xi];
      float x = static_cast<float>(pix_in.x()*scale_x + offset_x);
      float y = static_cast<float>(pix_in.y()*scale_y + offset_y);
      float z = static_cast<float>(pix_in.z()*scale_z + offset_z);
      out[yi][xi] = T(x,y,z);
    }
  }
  return;
}

template <class T>
void face3d::
unnormalize_3d_image(dlib::array2d<dlib::rgb_pixel> const& in, vgl_box_3d<double> const& bbox,
                     dlib::array2d<T> &out)
{
  const int nx = in.nc();
  const int ny = in.nr();
  out.set_size(ny,nx);

  float min_x = bbox.min_x();
  float x_range = bbox.max_x() - bbox.min_x();
  float min_y = bbox.min_y();
  float y_range = bbox.max_y() - bbox.min_y();
  float min_z = bbox.min_z();
  float z_range = bbox.max_z() - bbox.min_z();

  for (unsigned int yi=0; yi<ny; ++yi) {
    for (unsigned int xi=0; xi<nx; ++xi) {
      dlib::rgb_pixel in_pix = in[yi][xi];
      float x = static_cast<float>(in_pix.red)/255 * x_range + min_x;
      float y = static_cast<float>(in_pix.green)/255 * y_range + min_y;
      float z = static_cast<float>(in_pix.blue)/255 * z_range + min_z;
      out[yi][xi] = T(x,y,z);
    }
  }
  return;
}

