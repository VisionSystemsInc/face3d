#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d/textured_triangle_mesh.h>
#include <face3d/head_mesh.h>
#include <face3d_basic/dlib_face_detector.h>
#include <face3d/media_coefficient_from_semantic_map_estimator.h>
#include <face3d/media_coefficient_from_PNCC_and_offset_estimator.h>
#include <face3d/media_jitterer.h>
#include <face3d/offset_correction.h>
#include <face3d/mesh_renderer.h>
#include <face3d/coeffs_to_pixmap.h>
#include <face3d/camera_estimation.h>
#include <face3d/texture_map_operations.h>
#include <face3d/pose_jitterer.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>

#include "pybind_util.h"

using namespace face3d;
namespace py = pybind11;

using MESH_TEX_T = dlib::array2d<dlib::rgb_alpha_pixel>;

vpgl_affine_camera<double> ortho_camera_parameters_to_camera(ortho_camera_parameters<double> const& params)
{
  return params.to_camera();
}

vpgl_perspective_camera<double> perspective_camera_parameters_to_camera(perspective_camera_parameters<double> const& params)
{
  vpgl_perspective_camera<double> pcam;
  params.to_camera(pcam);
  return pcam;
}

py::array_t<float>
wrap_coeffs_to_pixmap(face3d::head_mesh const& mesh,
                      vnl_vector<double> const& subject_coeffs, vnl_vector<double> const &expression_coeffs,
                      vnl_matrix<double> const& subject_components, vnl_matrix<double> const& expression_components)
{
  dlib::array2d<vgl_point_3d<float> > pixmap_dlib =
    face3d::coeffs_to_pixmap(mesh,
                             subject_coeffs, expression_coeffs,
                             subject_components, expression_components);
  py::array_t<float> pixmap_out;
  pybind_util::img_to_buffer(pixmap_dlib, pixmap_out);
  return pixmap_out;
}

py::array_t<float>
wrap_correct_offsets(py::array_t<float> &pncc, py::array_t<float> &offsets)
{
  dlib::array2d<vgl_point_3d<float> > pncc_dlib;
  pybind_util::img_from_buffer(pncc, pncc_dlib);
  dlib::array2d<vgl_vector_3d<float> > offsets_dlib;
  pybind_util::img_from_buffer(offsets, offsets_dlib);
  dlib::array2d<vgl_vector_3d<float> > offsets_out_dlib;
  py::array_t<float> offsets_out;
  std::cout << "correcting offsets" << std::endl;
  if(!face3d::correct_offsets(pncc_dlib, offsets_dlib, offsets_out_dlib)) {
    std::cerr << "WARNING: face3d::correct_offsets() returned error" << std::endl;
    pybind_util::img_to_buffer(offsets_dlib, offsets_out);
  }
  else {
    pybind_util::img_to_buffer(offsets_out_dlib, offsets_out);
  }
  return offsets_out;
}

py::array_t<float>
wrap_generate_face_symmetry_map( face3d::head_mesh const& base_mesh )
{
  dlib::array2d<dlib::vector<float,2> > sym_map_dlib;
  face3d::generate_face_symmetry_map(base_mesh, sym_map_dlib);
  py::array_t<float> sym_map_np;
  pybind_util::img_to_buffer(sym_map_dlib, sym_map_np);
  return sym_map_np;
}

std::vector<size_t>
wrap_mask_mesh_faces(triangle_mesh const& mesh, py::array_t<unsigned char> &mask)
{
  dlib::array2d<unsigned char> mask_dlib;
  pybind_util::img_from_buffer(mask, mask_dlib);
  std::vector<size_t> valid_faces;
  mask_mesh_faces(mesh, mask_dlib, valid_faces);
  return valid_faces;
}

void construct_triangle_mesh(face3d::triangle_mesh &mesh,
                             py::array_t<float> &V,
                             py::array_t<int> &F)
{
  // convert input matrices to Eigen
  triangle_mesh::VTYPE Ve;
  triangle_mesh::FTYPE Fe;
  pybind_util::matrix_from_buffer(V, Ve);
  pybind_util::matrix_from_buffer(F, Fe);
  // in-place constructor
  new (&mesh) triangle_mesh(Ve, Fe);
}

void construct_triangle_mesh_full(face3d::triangle_mesh &mesh,
                                  py::array_t<float> &V,
                                  py::array_t<int> &F,
                                  py::array_t<float> &N,
                                  py::array_t<float> &UV)
{
  // convert input matrices to Eigen
  triangle_mesh::VTYPE Ve;
  triangle_mesh::FTYPE Fe;
  triangle_mesh::NTYPE Ne;
  triangle_mesh::TTYPE Te;
  pybind_util::matrix_from_buffer(V, Ve);
  pybind_util::matrix_from_buffer(F, Fe);
  pybind_util::matrix_from_buffer(N, Ne);
  pybind_util::matrix_from_buffer(UV, Te);
  // in-place constructor
  new (&mesh) triangle_mesh(Ve, Fe, Ne, Te);
}

void construct_textured_triangle_mesh(face3d::textured_triangle_mesh<MESH_TEX_T> &tmesh,
                                      face3d::triangle_mesh const& mesh,
                                      py::array_t<typename face3d_img_util::img_traits<MESH_TEX_T>::dtype> &tex)
{
  MESH_TEX_T tex_dlib;
  pybind_util::img_from_buffer(tex, tex_dlib);
  new (&tmesh) textured_triangle_mesh<MESH_TEX_T>(mesh, tex_dlib);
}

py::array_t<double>
wrap_triangle_mesh_V(face3d::triangle_mesh const& mesh)
{
  triangle_mesh::VTYPE const&V = mesh.V();
  py::array_t<double> V_py;
  pybind_util::matrix_to_buffer(V, V_py);
  return V_py;
}

py::array_t<double>
wrap_triangle_mesh_UV(face3d::triangle_mesh const& mesh)
{
  triangle_mesh::UVTYPE const&UV = mesh.UV();
  py::array_t<double> UV_py;
  pybind_util::matrix_to_buffer(UV, UV_py);
  return UV_py;
}

py::array_t<double>
wrap_triangle_mesh_N(face3d::triangle_mesh const& mesh)
{
  triangle_mesh::NTYPE const&N = mesh.N();
  py::array_t<double> N_py;
  pybind_util::matrix_to_buffer(N, N_py);
  return N_py;
}


py::array_t<int>
wrap_triangle_mesh_F(face3d::triangle_mesh const& mesh)
{
  triangle_mesh::FTYPE const&F = mesh.F();
  py::array_t<int> F_py;
  pybind_util::matrix_to_buffer(F, F_py);
  return F_py;
}

template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
py::array_t<typename face3d_img_util::img_traits<IMG_T_OUT>::dtype>
wrap_image_to_texture(py::array_t<typename face3d_img_util::img_traits<IMG_T_IN>::dtype> &img,
                      face3d::textured_triangle_mesh<TEX_T> const& mesh, CAM_T const& cam_params,
                      mesh_renderer &renderer)
{
  IMG_T_IN img_dlib;
  pybind_util::img_from_buffer(img, img_dlib);
  IMG_T_OUT tex_out;
  std::vector<face3d::textured_triangle_mesh<TEX_T> > meshes {mesh};
  face3d::image_to_texture(img_dlib, meshes, cam_params, renderer, tex_out);
  py::array_t<typename face3d_img_util::img_traits<IMG_T_OUT>::dtype> tex_out_np;
  pybind_util::img_to_buffer(tex_out, tex_out_np);
  return tex_out_np;
}

template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
py::array_t<typename face3d_img_util::img_traits<IMG_T_OUT>::dtype>
wrap_texture_to_image(py::array_t<typename face3d_img_util::img_traits<IMG_T_IN>::dtype> &tex,
                      face3d::textured_triangle_mesh<TEX_T> const& mesh,
                      CAM_T const& cam_params,
                      mesh_renderer &renderer)
{
  IMG_T_IN tex_dlib;
  pybind_util::img_from_buffer(tex, tex_dlib);
  IMG_T_OUT img_out_dlib;
  face3d::texture_to_image(tex_dlib, mesh, cam_params, renderer, img_out_dlib);
  py::array_t<typename face3d_img_util::img_traits<IMG_T_OUT>::dtype> img_out;
  pybind_util::img_to_buffer(img_out_dlib, img_out);
  return img_out;
}

template<class CAM_T>
CAM_T wrap_compute_camera_params_from_pncc_and_offsets(py::array_t<float> &pncc, py::array_t<float> &offsets)
{
  dlib::array2d<vgl_point_3d<float> > pncc_dlib;
  pybind_util::img_from_buffer(pncc, pncc_dlib);
  dlib::array2d<vgl_vector_3d<float> > offsets_dlib;
  pybind_util::img_from_buffer(offsets, offsets_dlib);
  CAM_T cam_out;
  if(!face3d::camera_estimation::
     compute_camera_params_from_pncc_and_offsets(pncc_dlib, offsets_dlib, cam_out)) {
    throw std::runtime_error("compute_camera_params() failed");
  }
  return cam_out;
}

template<class CAM_T>
CAM_T wrap_compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                                 std::vector<vgl_point_3d<double> > const& pts3d,
                                 int nx, int ny)
{
  CAM_T cam_out;
  face3d::camera_estimation::compute_camera_params(pts2d, pts3d, nx, ny, cam_out);
  return cam_out;
}



template<class CAM_T>
std::tuple<face3d::subject_sighting_coefficients<CAM_T>, face3d::estimation_results_t>
wrap_estimate_coefficients_from_pncc_and_offsets( face3d::media_coefficient_from_PNCC_and_offset_estimator &estimator,
                                                  std::vector<std::string> const& img_ids,
                                                  std::vector<py::array_t<float> > &pnccs,
                                                  std::vector<py::array_t<float> > &offsets)
{
  const int num_imgs = img_ids.size();
  if (pnccs.size() != num_imgs) {
    throw std::logic_error("Different number of image IDs and pnccs passed to estimate_coefficients");
  }
  if (offsets.size() != num_imgs) {
    throw std::logic_error("Different number of image IDs and offsets passed to estimate_coefficients");
  }
  std::vector<dlib::array2d<vgl_point_3d<float> > > pnccs_dlib(num_imgs);
  std::vector<dlib::array2d<vgl_vector_3d<float> > > offsets_dlib(num_imgs);
  for (int i=0; i<num_imgs; ++i) {
    pybind_util::img_from_buffer(pnccs[i], pnccs_dlib[i]);
    pybind_util::img_from_buffer(offsets[i], offsets_dlib[i]);
  }
  subject_sighting_coefficients<CAM_T> coeffs;
  face3d::estimation_results_t retval = estimator.estimate_coefficients(img_ids, pnccs_dlib, offsets_dlib, coeffs);
  if (!retval) {
    std::cerr << "WARNING: media_coefficient_from_PNCC_and_offset_estimator::estimate_coefficients() returned error" <<std::endl;
  }
  return std::make_tuple(coeffs, retval);
}

template<class CAM_T>
std::tuple<face3d::subject_sighting_coefficients<CAM_T>, face3d::estimation_results_t>
wrap_estimate_coefficients_from_pncc( face3d::media_coefficient_from_semantic_map_estimator &estimator,
                                                  std::vector<std::string> const& img_ids,
                                                  std::vector<py::array_t<float> > &pnccs)
{
  const int num_imgs = img_ids.size();
  if (pnccs.size() != num_imgs) {
    throw std::logic_error("Different number of image IDs and pnccs passed to estimate_coefficients");
  }
  std::vector<dlib::array2d<vgl_point_3d<float> > > pnccs_dlib(num_imgs);
  for (int i=0; i<num_imgs; ++i) {
    pybind_util::img_from_buffer(pnccs[i], pnccs_dlib[i]);
  }
  subject_sighting_coefficients<CAM_T> coeffs;
  face3d::estimation_results_t retval = estimator.estimate_coefficients(img_ids, pnccs_dlib, coeffs);
  if (!retval) {
    std::cerr << "WARNING: media_coefficient_from_PNCC_estimator::estimate_coefficients() returned error" <<std::endl;
  }
  return std::make_tuple(coeffs, retval);
}

py::array_t<unsigned char>
wrap_render_2d(face3d::mesh_renderer &renderer,
               py::array_t<double> &V, py::array_t<int> &F,
               py::array_t<double> &UV, py::array_t<unsigned char> &tex,
               int nx, int ny, float zmin, float zmax)
{
  dlib::array2d<dlib::rgb_alpha_pixel> tex_dlib;
  pybind_util::img_from_buffer(tex, tex_dlib);
  dlib::array2d<dlib::rgb_alpha_pixel> img_out_dlib;

  triangle_mesh::VTYPE V_eig;
  triangle_mesh::FTYPE F_eig;
  triangle_mesh::TTYPE UV_eig;
  pybind_util::matrix_from_buffer(V, V_eig);
  pybind_util::matrix_from_buffer(F, F_eig);
  pybind_util::matrix_from_buffer(UV, UV_eig);
  renderer.render_2d(V_eig, F_eig, UV_eig, tex_dlib, nx, ny, zmin, zmax, img_out_dlib);

  py::array_t<unsigned char> py_img_out;
  pybind_util::img_to_buffer(img_out_dlib, py_img_out);
  return py_img_out;
}



template<class CAM_T, class TEX_T>
py::array_t<unsigned char>
wrap_render(face3d::mesh_renderer &renderer,
            std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
            CAM_T const& cam_params)
{
  dlib::array2d<dlib::rgb_alpha_pixel> img;
  renderer.render(meshes, cam_params, img);
  py::array_t<unsigned char> py_img;
  pybind_util::img_to_buffer(img, py_img);
  return py_img;
}

template<class CAM_T, class TEX_T>
py::array_t<float>
wrap_render_3d(face3d::mesh_renderer &renderer,
               std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
               CAM_T const& cam_params)
{
  dlib::array2d<vgl_point_3d<float> > img3d;
  face3d::render_aux_out aux_out;
  aux_out.img3d_ = &img3d;
  dlib::array2d<dlib::rgb_alpha_pixel> img;
  renderer.render(meshes, cam_params, img, aux_out);
  py::array_t<float> py_img3d;
  pybind_util::img_to_buffer(img3d, py_img3d);
  return py_img3d;
}

template<class CAM_T, class TEX_T>
py::array_t<float>
wrap_render_normals(face3d::mesh_renderer &renderer,
                    std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
                    CAM_T const& cam_params)
{
  dlib::array2d<vgl_vector_3d<float> > normals;
  face3d::render_aux_out aux_out;
  aux_out.normals_ = &normals;
  dlib::array2d<dlib::rgb_alpha_pixel> img;
  renderer.render(meshes, cam_params, img, aux_out);
  py::array_t<float> py_normals;
  pybind_util::img_to_buffer(normals, py_normals);
  return py_normals;
}


template<class CAM_T, class TEX_T>
py::array_t<float>
wrap_render_uv(face3d::mesh_renderer &renderer,
               std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
               CAM_T const& cam_params)
{
  dlib::array2d<vgl_point_3d<float> > img_uv;
  face3d::render_aux_out aux_out;
  aux_out.uv_ = &img_uv;
  dlib::array2d<dlib::rgb_alpha_pixel> img;
  renderer.render(meshes, cam_params, img, aux_out);
  py::array_t<float> py_img_uv;
  pybind_util::img_to_buffer(img_uv, py_img_uv);
  return py_img_uv;
}


template<class TEX_T>
void wrap_set_texture(face3d::textured_triangle_mesh<TEX_T> &mesh,
                      py::array_t<unsigned char> &tex)
{
  py::buffer_info info = tex.request();
  if (info.format != py::format_descriptor<unsigned char>::format()) {
    throw std::runtime_error("Incompatible buffer type for conversion to dlib image");
  }
  if (info.ndim != 3) {
    throw std::runtime_error("Expecting array with shape RxCxP");
  }
  const size_t num_planes = info.shape[2];
  if (num_planes == 3) {
    dlib::array2d<dlib::rgb_pixel> tex_dlib;
    pybind_util::img_from_buffer(tex, tex_dlib);
    mesh.set_texture(tex_dlib);
  }
  else if (num_planes == 4) {
    dlib::array2d<dlib::rgb_alpha_pixel> tex_dlib;
    pybind_util::img_from_buffer(tex, tex_dlib);
    mesh.set_texture(tex_dlib);
  }
  else {
    throw std::runtime_error("Unexpected number of planes in texture image");
  }
  return;
}

template<class TEX_T>
py::array_t<unsigned char> wrap_get_texture(face3d::textured_triangle_mesh<TEX_T> &mesh)
{
  TEX_T const& tex = mesh.texture();
  py::array_t<unsigned char> py_tex_out;
  pybind_util::img_to_buffer(tex, py_tex_out);
  return py_tex_out;
}

template<class CAM_T>
void construct_media_jitterer(face3d::media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel>> &jitterer,
                              std::vector<py::array_t<unsigned char> > &imgs,
                              face3d::subject_sighting_coefficients<CAM_T> const& coeffs,
                              face3d::head_mesh const& base_mesh,
                              vnl_matrix<double> const& subject_pca_components,
                              vnl_matrix<double> const& expression_pca_components,
                              face3d::mesh_renderer &renderer,
                              std::string const& debug_dir)
{
  const int num_images = imgs.size();
  // convert input images to dlib array2d's
  std::vector<dlib::array2d<dlib::rgb_pixel> > imgs_dlib(num_images);
  for (int i=0; i<num_images; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  // extract sighting coeffs
  std::vector<face3d::sighting_coefficients<CAM_T> > sighting_coeffs =
    coeffs.all_sightings();

  // in-place constructor
  new (&jitterer) media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel>>(imgs_dlib, sighting_coeffs, base_mesh,
                                        subject_pca_components, expression_pca_components,
                                        renderer, debug_dir);
}

template<class CAM_T>
void construct_media_jitterer_wsymm(face3d::media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel>> &jitterer,
                                    std::vector<py::array_t<unsigned char> > &imgs,
                                    face3d::subject_sighting_coefficients<CAM_T> const& coeffs,
                                    face3d::head_mesh const& base_mesh,
                                    vnl_matrix<double> const& subject_pca_components,
                                    vnl_matrix<double> const& expression_pca_components,
                                    face3d::mesh_renderer &renderer,
                                    py::array_t<float> &face_symmetry_map,
                                    std::string const& debug_dir)
{
  const int num_images = imgs.size();
  // convert input images to dlib array2d's
  std::vector<dlib::array2d<dlib::rgb_pixel> > imgs_dlib(num_images);
  for (int i=0; i<num_images; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  // extract sighting coeffs
  std::vector<face3d::sighting_coefficients<CAM_T> > sighting_coeffs =
    coeffs.all_sightings();

  // convert face symmetry map to dlib array2d
  dlib::array2d<dlib::vector<float,2> > face_symmetry_map_dlib;
  pybind_util::img_from_buffer(face_symmetry_map, face_symmetry_map_dlib);

  // in-place constructor
  new (&jitterer) media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel>>(imgs_dlib, sighting_coeffs, base_mesh,
                                        subject_pca_components, expression_pca_components,
                                        renderer, face_symmetry_map_dlib, debug_dir);
}

template<class CAM_T, class CAM_OUT_T>
py::array_t<unsigned char>
wrap_jitter_render(face3d::media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel>> &jitterer,
                   CAM_OUT_T const& render_cam,
                   vnl_vector<double> const& subject_coeffs,
                   vnl_vector<double> const& expression_coeffs,
                   vnl_matrix<double> const& subject_pca_components,
                   vnl_matrix<double> const& expression_pca_components)
{
  dlib::array2d<dlib::rgb_alpha_pixel> img_out_dlib;
  jitterer.render(render_cam, subject_coeffs, expression_coeffs,
                  subject_pca_components, expression_pca_components,
                  img_out_dlib);

  py::array_t<unsigned char> img_out;
  pybind_util::img_to_buffer(img_out_dlib,img_out);

  return img_out;
}


std::vector<py::array_t<unsigned char> >
wrap_pose_jitterer_jitter_images(face3d::pose_jitterer_uniform& jitterer_uniform,
                                 std::vector<py::array_t<unsigned char> > &imgs,
                                 face3d::subject_sighting_coefficients<CAM_T> & coeffs, int num_jitters)
{
  // handle input
  unsigned num_images_in = imgs.size();
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_dlib(num_images_in);
  for (int i=0; i<num_images_in; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_out_dlib;

  jitterer_uniform.jitter_images(imgs_dlib, coeffs, num_jitters, imgs_out_dlib);
  unsigned num_images_out = imgs_out_dlib.size();
  std::vector<py::array_t<unsigned char> > imgs_out(num_images_out);
  for (int i=0; i<num_images_out; ++i) {
    pybind_util::img_to_buffer(imgs_out_dlib[i], imgs_out[i]);
  }
  return imgs_out;
}

py::array_t<unsigned char>
wrap_pose_jitterer_random_jitter(face3d::pose_jitterer_uniform& jitterer_uniform,
                                 std::vector<py::array_t<unsigned char> > &imgs,
                                 face3d::subject_sighting_coefficients<CAM_T> & coeffs, int num_jitters)
{
  // handle input
  unsigned num_images_in = imgs.size();
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_dlib(num_images_in);
  for (int i=0; i<num_images_in; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  dlib::matrix<dlib::rgb_pixel>  img_out_dlib;
  py::array_t<unsigned char> img_out;
  jitterer_uniform.get_random_jitter(imgs_dlib, coeffs, num_jitters, img_out_dlib);
  pybind_util::img_to_buffer(img_out_dlib, img_out);

  return img_out;
}


void wrap_pose_jitterer(py::module &m, std::string pyname)
{
  py::class_<face3d::pose_jitterer_uniform>(m, pyname.c_str())
    .def(py::init<std::string, int, int, std::string, float, float, unsigned>())
    .def("jitter_images", wrap_pose_jitterer_jitter_images)
    .def("get_random_jitter", wrap_pose_jitterer_random_jitter);
}


std::vector<py::array_t<unsigned char> >
wrap_profile_jitterer_jitter_images(face3d::pose_jitterer_profile& jitterer_profile,
                                 std::vector<py::array_t<unsigned char> > &imgs,
                                 face3d::subject_sighting_coefficients<CAM_T> & coeffs, int num_jitters)
{
  // handle input
  unsigned num_images_in = imgs.size();
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_dlib(num_images_in);
  for (int i=0; i<num_images_in; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_out_dlib;

  jitterer_profile.jitter_images(imgs_dlib, coeffs, num_jitters, imgs_out_dlib);
  unsigned num_images_out = imgs_out_dlib.size();
  std::vector<py::array_t<unsigned char> > imgs_out(num_images_out);
  for (int i=0; i<num_images_out; ++i) {
    pybind_util::img_to_buffer(imgs_out_dlib[i], imgs_out[i]);
  }
  return imgs_out;
}

py::array_t<unsigned char>
wrap_profile_jitterer_random_jitter(face3d::pose_jitterer_profile& jitterer_profile,
                                 std::vector<py::array_t<unsigned char> > &imgs,
                                 face3d::subject_sighting_coefficients<CAM_T> & coeffs, int num_jitters)
{
  // handle input
  unsigned num_images_in = imgs.size();
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_dlib(num_images_in);
  for (int i=0; i<num_images_in; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  dlib::matrix<dlib::rgb_pixel>  img_out_dlib;
  py::array_t<unsigned char> img_out;
  jitterer_profile.get_random_jitter(imgs_dlib, coeffs, num_jitters, img_out_dlib);
  pybind_util::img_to_buffer(img_out_dlib, img_out);

  return img_out;
}

std::vector<py::array_t<unsigned char> >
wrap_profile_jitterer_multiple_random_jitters(face3d::pose_jitterer_profile& jitterer_profile,
                                              std::vector<py::array_t<unsigned char> > &imgs,
                                              face3d::subject_sighting_coefficients<CAM_T> & coeffs, int num_jitters)
{
  // handle input
  unsigned num_images_in = imgs.size();
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_dlib(num_images_in);
  for (int i=0; i<num_images_in; ++i) {
    pybind_util::img_from_buffer(imgs[i], imgs_dlib[i]);
  }
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_out_dlib;

  jitterer_profile.multiple_random_jitters(imgs_dlib, coeffs, num_jitters, imgs_out_dlib);
  //    jitterer_profile.jitter_images(imgs_dlib, coeffs, num_jitters, imgs_out_dlib);
  unsigned num_images_out = imgs_out_dlib.size();
  std::vector<py::array_t<unsigned char> > imgs_out(num_images_out);
  for (int i=0; i<num_images_out; ++i) {
    pybind_util::img_to_buffer(imgs_out_dlib[i], imgs_out[i]);
  }
  return imgs_out;
}

void wrap_profile_jitterer(py::module &m, std::string pyname)
{
  py::class_<face3d::pose_jitterer_profile>(m, pyname.c_str())
    .def(py::init<std::string, int, int, std::string, float, float, unsigned>())
    .def("jitter_images", wrap_profile_jitterer_jitter_images)
    .def("get_random_jitter", wrap_profile_jitterer_random_jitter)
    .def("multiple_random_jitters", wrap_profile_jitterer_multiple_random_jitters);
}



template<class CAM_T>
void wrap_media_jitterer(py::module &m, std::string pyname)
{
  py::class_<face3d::media_jitterer<CAM_T, dlib::array2d<dlib::rgb_pixel> > >(m, pyname.c_str())
    .def("__init__", construct_media_jitterer<CAM_T>)
    .def("__init__", construct_media_jitterer_wsymm<CAM_T>)
    .def("render", wrap_jitter_render<CAM_T, face3d::ortho_camera_parameters<double> >)
    .def("render", wrap_jitter_render<CAM_T, face3d::perspective_camera_parameters<double> >);
}

template<class CAM_T>
void wrap_subject_sighting_coefficients(py::module &m, std::string pyname)
{
  py::class_<face3d::subject_sighting_coefficients<CAM_T> >(m, pyname.c_str())
    // constructor: subject coeffs, image filenames, expression coeffs, camera params
    .def(py::init<
         vnl_vector<double>,
         std::vector<std::string>,
         std::vector<vnl_vector<double> >,
         std::vector<CAM_T>
         >())
    .def(py::init<std::string>())
    .def(py::init<>())
    .def_property_readonly("num_sightings", &face3d::subject_sighting_coefficients<CAM_T>::num_sightings)
    .def("subject_coeffs", &face3d::subject_sighting_coefficients<CAM_T>::subject_coeffs, py::return_value_policy::copy)
    .def("camera", &face3d::subject_sighting_coefficients<CAM_T>::camera, py::return_value_policy::copy)
    .def("image_filename", &subject_sighting_coefficients<CAM_T>::image_filename)
    .def("expression_coeffs", &subject_sighting_coefficients<CAM_T>::expression_coeffs, py::return_value_policy::copy)
    .def("save",&subject_sighting_coefficients<CAM_T>::save)
    .def("__repr__",[](face3d::subject_sighting_coefficients<CAM_T> const& c){
         std::stringstream ss;
         c.write(ss);
         return ss.str();
         });
}


PYBIND11_MODULE(face3d, m)
{
  m.doc() = "face3d python bindings";

  py::module::import("vxl");
  py::class_<face3d::ortho_camera_parameters<double> > (m, "ortho_camera_parameters")
    // constructor: scale, offset, rotation, nx, ny
    .def(py::init<double, vgl_vector_2d<double>, vgl_rotation_3d<double>, int, int>())
    .def_property_readonly("scale", &face3d::ortho_camera_parameters<double>::scale)
    .def_property_readonly("offset", &face3d::ortho_camera_parameters<double>::offset, py::return_value_policy::copy)
    .def_property_readonly("rotation", &face3d::ortho_camera_parameters<double>::rotation, py::return_value_policy::copy)
    .def_property_readonly("nx", &face3d::ortho_camera_parameters<double>::nx)
    .def_property_readonly("ny", &face3d::ortho_camera_parameters<double>::ny)
    .def("to_camera", &ortho_camera_parameters_to_camera);

  py::class_<face3d::perspective_camera_parameters<double> > (m, "perspective_camera_parameters")
    // constructor: focal_len, principal_pt, rotation, translation, nx, ny
    .def(py::init<double, vgl_point_2d<double>, vgl_rotation_3d<double>, vgl_vector_3d<double>, int, int>())
    .def_property_readonly("focal_len", &face3d::perspective_camera_parameters<double>::focal_len)
    .def_property_readonly("principal_point", &face3d::perspective_camera_parameters<double>::principal_point, py::return_value_policy::copy)
    .def_property_readonly("rotation", &face3d::perspective_camera_parameters<double>::rotation, py::return_value_policy::copy)
    .def_property_readonly("translation", &face3d::perspective_camera_parameters<double>::translation, py::return_value_policy::copy)
    .def_property_readonly("nx", &face3d::perspective_camera_parameters<double>::nx)
    .def_property_readonly("ny", &face3d::perspective_camera_parameters<double>::ny)
    .def("to_camera", &perspective_camera_parameters_to_camera);

  wrap_subject_sighting_coefficients<face3d::ortho_camera_parameters<double> >(m, "subject_ortho_sighting_coefficients");
  wrap_subject_sighting_coefficients<face3d::perspective_camera_parameters<double> >(m, "subject_perspective_sighting_coefficients");


  py::class_<face3d::triangle_mesh>(m,"triangle_mesh")
    .def(py::init<std::string>())
    .def(py::init<face3d::triangle_mesh&>())
    .def("__init__", construct_triangle_mesh)
    .def("__init__", construct_triangle_mesh_full)
    .def_property_readonly("num_faces", &face3d::triangle_mesh::num_faces)
    .def_property_readonly("num_vertices", &face3d::triangle_mesh::num_vertices)
    .def("save_ply", &face3d::triangle_mesh::save_ply)
    .def("V", &wrap_triangle_mesh_V)
    .def("F", &wrap_triangle_mesh_F)
    .def("UV", &wrap_triangle_mesh_UV)
    .def("N", &wrap_triangle_mesh_N)
    .def("vertex", &face3d::triangle_mesh::vertex)
    .def("vertex_tex", &face3d::triangle_mesh::vertex_tex)
    .def("face", &face3d::triangle_mesh::face)
    .def("set_texture_coord", &face3d::triangle_mesh::set_texture_coord);

  py::class_<face3d::textured_triangle_mesh<MESH_TEX_T>, face3d::triangle_mesh >(m, "textured_triangle_mesh")
    .def(py::init<std::string, std::string>())
    .def(py::init<face3d::textured_triangle_mesh<MESH_TEX_T>& >())
    .def("__init__", construct_textured_triangle_mesh)
    .def_property_readonly("num_faces", &face3d::textured_triangle_mesh<MESH_TEX_T>::num_faces)
    .def_property_readonly("num_vertices", &face3d::textured_triangle_mesh<MESH_TEX_T>::num_vertices)
    .def("set_texture", &wrap_set_texture<MESH_TEX_T>)
    .def("texture", &wrap_get_texture<MESH_TEX_T>);

  py::class_<face3d::head_mesh>(m, "head_mesh")
    .def(py::init<std::string>())
    .def(py::init<face3d::head_mesh&>())
    .def("apply_coefficients", &face3d::head_mesh::apply_coefficients)
    .def("face_mesh", (textured_triangle_mesh<face3d::head_mesh::TEX_T>& (face3d::head_mesh::*)()) &face3d::head_mesh::face_mesh, "Get a reference to the face mesh", py::return_value_policy::reference)
    .def("meshes", (std::vector<textured_triangle_mesh<face3d::head_mesh::TEX_T> >& (face3d::head_mesh::*)()) &face3d::head_mesh::meshes, "Get all meshes");

  py::class_<face3d::dlib_face_detector>(m, "dlib_face_detector")
    .def(py::init<std::string>());


  py::class_<face3d::estimation_results_t>(m, "estimation_results_t")
    .def_readonly("success", &face3d::estimation_results_t::success_)
    //.def_readonly("image_success", &face3d::estimation_results_t::image_success_)
    .def_readonly("vertices_found", &face3d::estimation_results_t::vertices_found_);

  py::class_<face3d::media_coefficient_from_PNCC_and_offset_estimator>(m, "media_coefficient_from_PNCC_and_offset_estimator")
    .def(py::init<face3d::head_mesh, vnl_matrix<double> const&, vnl_matrix<double> const&, vnl_matrix<double> const&, vnl_matrix<double> const&, bool, std::string>())
    .def("estimate_coefficients_ortho", &wrap_estimate_coefficients_from_pncc_and_offsets<face3d::ortho_camera_parameters<double> >)
    .def("estimate_coefficients_perspective", &wrap_estimate_coefficients_from_pncc_and_offsets<face3d::perspective_camera_parameters<double> >);

  py::class_<face3d::media_coefficient_from_semantic_map_estimator>(m, "media_coefficient_from_PNCC_estimator")
    .def(py::init<face3d::head_mesh, vnl_matrix<double> const&, vnl_matrix<double> const&, vnl_matrix<double> const&, vnl_matrix<double> const&, bool, std::string>())
    .def("estimate_coefficients_ortho", &wrap_estimate_coefficients_from_pncc<face3d::ortho_camera_parameters<double> >)
    .def("estimate_coefficients_perspective", &wrap_estimate_coefficients_from_pncc<face3d::perspective_camera_parameters<double> >);

  py::class_<face3d::mesh_renderer>(m, "mesh_renderer")
    .def(py::init<>())
    .def(py::init<unsigned>())
    .def("render", &wrap_render<ortho_camera_parameters<double>, MESH_TEX_T >)
    .def("render", &wrap_render<perspective_camera_parameters<double>, MESH_TEX_T>)
    .def("render_3d", &wrap_render_3d<ortho_camera_parameters<double>, MESH_TEX_T>)
    .def("render_3d", &wrap_render_3d<perspective_camera_parameters<double>, MESH_TEX_T>)
    .def("render_uv", &wrap_render_uv<ortho_camera_parameters<double>, MESH_TEX_T>)
    .def("render_uv", &wrap_render_uv<perspective_camera_parameters<double>, MESH_TEX_T>)
    .def("render_normals", &wrap_render_normals<ortho_camera_parameters<double>, MESH_TEX_T>)
    .def("render_normals", &wrap_render_normals<perspective_camera_parameters<double>, MESH_TEX_T>)
    .def("render_2d", &wrap_render_2d)
    .def("set_ambient_weight", &face3d::mesh_renderer::set_ambient_weight)
    .def("set_light_dir", &face3d::mesh_renderer::set_light_dir);

  m.def("correct_offsets", &wrap_correct_offsets);

  m.def("compute_camera_params_from_pncc_and_offsets_ortho", &wrap_compute_camera_params_from_pncc_and_offsets<ortho_camera_parameters<double> >);
  m.def("compute_camera_params_from_pncc_and_offsets_perspective", &wrap_compute_camera_params_from_pncc_and_offsets<perspective_camera_parameters<double> >);

  m.def("compute_camera_params_ortho", &wrap_compute_camera_params<ortho_camera_parameters<double> >);
  m.def("compute_camera_params_perspective", &wrap_compute_camera_params<perspective_camera_parameters<double> >);

  wrap_media_jitterer<face3d::ortho_camera_parameters<double> >(m, "media_jitterer_ortho");
  wrap_media_jitterer<face3d::perspective_camera_parameters<double> >(m, "media_jitterer_perspective");
  wrap_pose_jitterer(m, "pose_jitterer_uniform");
  wrap_profile_jitterer(m, "pose_jitterer_profile");

  m.def("coeffs_to_pixmap", &wrap_coeffs_to_pixmap);

  m.def("texture_to_image", &wrap_texture_to_image<face3d::perspective_camera_parameters<double>, dlib::array2d<dlib::rgb_alpha_pixel>, dlib::array2d<dlib::rgb_alpha_pixel>, MESH_TEX_T >);
  m.def("texture_to_image", &wrap_texture_to_image<face3d::ortho_camera_parameters<double>, dlib::array2d<dlib::rgb_alpha_pixel>, dlib::array2d<dlib::rgb_alpha_pixel>, MESH_TEX_T >);
  m.def("texture_to_image_float", &wrap_texture_to_image<face3d::perspective_camera_parameters<double>, vil_image_view<float>, vil_image_view<float>, MESH_TEX_T >);
  m.def("texture_to_image_float", &wrap_texture_to_image<face3d::ortho_camera_parameters<double>, vil_image_view<float>, vil_image_view<float>, MESH_TEX_T >);
  m.def("image_to_texture", &wrap_image_to_texture<face3d::perspective_camera_parameters<double>, dlib::array2d<dlib::rgb_pixel>, dlib::array2d<dlib::rgb_alpha_pixel>, MESH_TEX_T >);
  m.def("image_to_texture", &wrap_image_to_texture<face3d::ortho_camera_parameters<double>, dlib::array2d<dlib::rgb_pixel>, dlib::array2d<dlib::rgb_alpha_pixel>, MESH_TEX_T >);
  m.def("image_to_texture_float", &wrap_image_to_texture<face3d::perspective_camera_parameters<double>, vil_image_view<float>, vil_image_view<float>, MESH_TEX_T >);
  m.def("image_to_texture_float", &wrap_image_to_texture<face3d::ortho_camera_parameters<double>, vil_image_view<float>, vil_image_view<float>, MESH_TEX_T >);
  m.def("generate_face_symmetry_map", &wrap_generate_face_symmetry_map);
  m.def("mask_mesh_faces", &wrap_mask_mesh_faces);

#ifdef FACE3D_USE_CUDA
  m.def("set_cuda_device", &face3d::set_cuda_device);
  m.def("get_cuda_device", &face3d::get_cuda_device);
#endif
}
