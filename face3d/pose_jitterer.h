#pragma once
#include <face3d/head_mesh.h>
#include <cstddef>
#include <face3d/media_jitterer.h>
#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d_basic/io_utils.h>
using CAM_T = face3d::perspective_camera_parameters<double>;
namespace face3d{
vnl_vector_fixed<double, 3> quaternion_to_euler_angles(vnl_quaternion<double> const & q,  std::string order="XYZ");
vnl_quaternion<double> euler_angles_to_quaternion(double theta1, double theta2, double theta3, std::string order="XYZ");
vgl_vector_3d<double> axis_from_string(char axis);

class  pose_jitterer_base {

public:
  pose_jitterer_base(std::string pvr_data_dir, int num_subject_components=20, int num_expression_components=10, std::string debug_dir="", unsigned gl_device_id=0):
    base_mesh_(face3d::head_mesh(pvr_data_dir)), renderer_(0), gl_device_id_(gl_device_id){
    subject_components_.reset(new vnl_matrix<double>(face3d::io_utils::read_numpy_matrix(pvr_data_dir + "/pca_components_subject.npy").get_n_rows(0, num_subject_components)));
    expression_components_.reset(new vnl_matrix<double>(face3d::io_utils::read_numpy_matrix(pvr_data_dir + "/pca_components_expression.npy").get_n_rows(0, num_expression_components)));
    face3d::generate_face_symmetry_map(base_mesh_, sym_map_dlib_);
    debug_dir_ = debug_dir;
    rnd=dlib::rand();
    rnd.set_seed(dlib::cast_to_string(std::rand()));
  }
  pose_jitterer_base(pose_jitterer_base const & other):base_mesh_(other.base_mesh_){
    this->renderer_.reset();
    this->expression_components_ = other.expression_components_;
    this->subject_components_ = other.subject_components_;
    this->expression_component_in_ranges_ = other.expression_component_in_ranges_;
    this->subject_component_ranges_ = other.subject_component_ranges_;
    this->jitterer_ = other.jitterer_;
    this->gl_device_id_ = other.gl_device_id_;
    this->debug_dir_ = other.debug_dir_;
    this->sym_map_dlib_ = dlib::array2d<dlib::vector<float, 2> >(other.sym_map_dlib_.nr(), other.sym_map_dlib_.nc());
    rnd=other.rnd;
    rnd.set_seed(dlib::cast_to_string(std::rand()));
    for (unsigned i = 0; i < this->sym_map_dlib_.nr(); i++ )
      for (unsigned j = 0; j < this->sym_map_dlib_.nc(); j++ )
        this->sym_map_dlib_[i][j] = other.sym_map_dlib_[i][j]; //deep copy
  }

  pose_jitterer_base(pose_jitterer_base && other):base_mesh_(std::move(other.base_mesh_)){
    this->renderer_.reset();
    this->expression_components_ = (other.expression_components_);
    this->subject_components_ = (other.subject_components_);
    this->expression_component_in_ranges_ = std::move(other.expression_component_in_ranges_);
    this->subject_component_ranges_ = std::move(other.subject_component_ranges_);
    this->gl_device_id_ = other.gl_device_id_;
    this->jitterer_ = std::move(other.jitterer_);
    this->debug_dir_ = std::move(other.debug_dir_);
    this->sym_map_dlib_ = std::move(other.sym_map_dlib_);
    rnd=other.rnd;
  }
  ~pose_jitterer_base(){
    if (this->debug_dir_!=""){
        std::thread::id this_id = std::this_thread::get_id();
        std::cout<<" thread ID "<<std::hex<<this_id<<" killed the pose_jitterer: "<<this<<std::dec<<std::endl;
    }
  }

  virtual bool jitter_images(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                             face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs,
                             int num_jitters,
                             std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images)=0;

  bool get_random_jitter(dlib::matrix<dlib::rgb_pixel> const & input_image,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                 dlib::rand rnd=dlib::rand()){
    std::vector<dlib::matrix<dlib::rgb_pixel> > input_array_vec(1); input_array_vec.push_back(input_image);
    return this->get_random_jitter(input_array_vec, est_coeffs, num_jitters, output_image, rnd);
  }



  virtual bool get_random_jitter(std::vector<dlib::matrix<dlib::rgb_pixel> > const & input_images,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                 dlib::rand rnd=dlib::rand())=0;



protected:
  unsigned gl_device_id_;
  void init_renderer(){
    if (!renderer_){
      renderer_.reset(new face3d::mesh_renderer(this->gl_device_id_));
    }
  }
  CAM_T get_camera(CAM_T const& cam_og, double yaw, double pitch, double roll, unsigned nx, unsigned ny);

protected:
  face3d::head_mesh base_mesh_;
  dlib::array2d<dlib::vector<float,2> > sym_map_dlib_;
  std::shared_ptr<face3d::mesh_renderer> renderer_;
  std::shared_ptr<face3d::media_jitterer<CAM_T, dlib::matrix<dlib::rgb_pixel> > > jitterer_;
  std::string debug_dir_;
  std::shared_ptr<vnl_matrix<double> > subject_components_;
  std::shared_ptr<vnl_matrix<double> > expression_components_;
  std::shared_ptr<vnl_matrix<double> > subject_component_ranges_;
  std::shared_ptr<vnl_matrix<double> > expression_component_in_ranges_;
  dlib::rand rnd;
};

class pose_jitterer_uniform: public pose_jitterer_base{

public:
  pose_jitterer_uniform(std::string pvr_data_dir, int num_subject_components=20, int num_expression_components=10,
                        std::string debug_dir="", float min_yaw=-90.0f, float max_yaw=90.0f,
                        unsigned gl_device_id=0)
    : pose_jitterer_base(pvr_data_dir, num_subject_components, num_expression_components, debug_dir, gl_device_id)
  {
    min_yaw_ = min_yaw;
    max_yaw_ = max_yaw;
    if (min_yaw > max_yaw)
      std::swap(min_yaw_, max_yaw_);
  }

  pose_jitterer_uniform(pose_jitterer_uniform const & other): pose_jitterer_base(other){
    min_yaw_ = other.min_yaw_;
    max_yaw_ = other.max_yaw_;
  }

   pose_jitterer_uniform(pose_jitterer_uniform && other): pose_jitterer_base(other){
    min_yaw_ = other.min_yaw_;
    max_yaw_ = other.max_yaw_;

  }


  virtual bool jitter_images(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                             face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                             int num_jitters,
                             std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images);

  virtual bool get_random_jitter(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                 dlib::rand rnd=dlib::rand());


   virtual bool get_random_jitter(dlib::matrix<dlib::rgb_pixel> const & input_image,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                  dlib::rand rnd=dlib::rand()){
     std::vector<dlib::matrix<dlib::rgb_pixel> > input_array_vec; input_array_vec.push_back(input_image);
     return this->get_random_jitter(input_array_vec, est_coeffs, num_jitters, output_image, rnd);
   }

   virtual bool multiple_random_jitters(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images,
                                 dlib::rand rnd=dlib::rand());

  void get_cameras(face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs,
                   std::vector<CAM_T> & cameras, int num_jitters, bool random=false);
private:
   float min_yaw_, max_yaw_;
};

  class pose_jitterer_profile: public pose_jitterer_base{

public:
    pose_jitterer_profile(std::string pvr_data_dir, int num_subject_components=20, int num_expression_components=10, std::string debug_dir="", float min_yaw=30.0f, float max_yaw=90.0f,
                          unsigned gl_device_id=0)
      : pose_jitterer_base(pvr_data_dir, num_subject_components, num_expression_components, debug_dir, gl_device_id)
  {
    min_yaw_ = fabs(min_yaw);
    max_yaw_ = fabs(max_yaw);
    if (min_yaw > max_yaw)
      std::swap(min_yaw_, max_yaw_);
  }

  pose_jitterer_profile(pose_jitterer_profile const & other): pose_jitterer_base(other){
    min_yaw_ = other.min_yaw_;
    max_yaw_ = other.max_yaw_;

  }

   pose_jitterer_profile(pose_jitterer_profile && other): pose_jitterer_base(other){
    min_yaw_ = other.min_yaw_;
    max_yaw_ = other.max_yaw_;

  }

    void set_min_max_yaw(float min_yaw, float max_yaw){
       min_yaw_ = fabs(min_yaw);
       max_yaw_ = fabs(max_yaw);
       if (min_yaw > max_yaw)
         std::swap(min_yaw, max_yaw);
    }

    virtual bool jitter_images(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                             face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                             int num_jitters,
                             std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images);

    virtual bool get_random_jitter(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                 dlib::rand rnd=dlib::rand());
    virtual bool get_random_jitter(dlib::matrix<dlib::rgb_pixel> const & input_image,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 dlib::matrix<dlib::rgb_pixel> & output_image,
                                 dlib::rand rnd=dlib::rand()){
      std::vector<dlib::matrix<dlib::rgb_pixel> > input_array_vec; input_array_vec.push_back(input_image);
      return this->get_random_jitter(input_array_vec, est_coeffs, num_jitters, output_image, rnd);
   }

    virtual bool multiple_random_jitters(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                 face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                 int num_jitters,
                                 std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images,
                                 dlib::rand rnd=dlib::rand());

  void get_cameras(face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs,
                   std::vector<CAM_T> & cameras, int num_jitters, bool random=false);

  private:
    float min_yaw_, max_yaw_;


};

}
