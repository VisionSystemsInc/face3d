#include <face3d/pose_jitterer.h>
vgl_vector_3d<double> face3d::axis_from_string(char axis_string){

  if (axis_string == 'X')
    return vgl_vector_3d<double>(1,0,0);
  else if (axis_string == 'Y')
    return vgl_vector_3d<double>(0,1,0);
  else if (axis_string == 'Z')
    return vgl_vector_3d<double>(0,0,1);
  else{
    std::cout<< "Expecting one of [X,Y,Z], got "<< axis_string<<std::endl;
    return vgl_vector_3d<double>(0,0,0);
  }

}
vnl_vector_fixed<double, 3> face3d::quaternion_to_euler_angles(vnl_quaternion<double> const & q, std::string order){
    std::vector<std::string> valid_orders = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
    if (std::find(valid_orders.begin(), valid_orders.end(), order) == std::end(valid_orders)){
      std::cout<<"order "<<order<<" is invalid"<<std::endl;
      order="XYZ";
  }
  double p0, p1, p2, p3; std::vector<double> qv = {q.x(), q.y(), q.z(), q.r()};
  p0 = qv[3];

  p1 = qv[0];
  if (order[0] == 'Y'){
    p1 = qv[1];
  }else if (order[0] == 'Z'){
    p1 = qv[2];
  }

  p2 = qv[1];
  if (order[1] == 'X'){
    p2 = qv[0];
  }else if (order[1] == 'Z'){
    p2 = qv[2];
  }

  p3 = qv[2];
  if (order[2] == 'X'){
    p3 = qv[0];
  }else if (order[2] == 'Y'){
    p3 = qv[1];
  }
  vgl_vector_3d<double>  e1 = face3d::axis_from_string(order[0]);
  vgl_vector_3d<double>  e2 = face3d::axis_from_string(order[1]);
  vgl_vector_3d<double>  e3 = face3d::axis_from_string(order[2]);

  double e = dot_product(cross_product(e3, e2),e1) > 0 ? 1 : -1;
  double alpha = atan2(e*2*(p2*p3 + e*p0*p1), p0*p0 - p1*p1 - p2*p2 + p3*p3);
  double beta =  asin(-e*2*(p1*p3 - e*p0*p2));
  double gamma = atan2(e*2*(p1*p2 + e*p0*p3), p0*p0 + p1*p1 - p2*p2 - p3*p3);
  return vnl_vector_fixed<double, 3>(alpha, beta, gamma);

}

vnl_quaternion<double> face3d::euler_angles_to_quaternion(double theta1, double theta2, double theta3, std::string order){
    std::vector<std::string> valid_orders = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"};
    if (std::find(valid_orders.begin(), valid_orders.end(), order) == std::end(valid_orders)){
      std::cout<<"order "<<order<<" is invalid"<<std::endl;
      order="XYZ";
  }

   vgl_vector_3d<double>  e1 = face3d::axis_from_string(order[0]);
   vgl_vector_3d<double>  e2 = face3d::axis_from_string(order[1]);
   vgl_vector_3d<double>  e3 = face3d::axis_from_string(order[2]);

   vgl_vector_3d<double> imaginary1(e1 * sin(theta1/2.0));
   vgl_vector_3d<double> imaginary2(e2 * sin(theta2/2.0));
   vgl_vector_3d<double> imaginary3(e3 * sin(theta3/2.0));

   double r1= cos(theta1/2.0);
   double r2= cos(theta2/2.0);
   double r3= cos(theta3/2.0);

   vnl_quaternion<double> q1(imaginary1.x(), imaginary1.y(), imaginary1.z(), r1);
   vnl_quaternion<double> q2(imaginary2.x(), imaginary2.y(), imaginary2.z(), r2);
   vnl_quaternion<double> q3(imaginary3.x(), imaginary3.y(), imaginary3.z(), r3);

   return q1 * q2 * q3;
 }
CAM_T face3d::pose_jitterer_base::get_camera(CAM_T const& cam_og, double yaw, double pitch, double roll, unsigned nx, unsigned ny){

    //prepare single index vector
    vnl_quaternion<double> pose_as_quaternion = face3d::euler_angles_to_quaternion(yaw, pitch, roll ,"YXZ");
    vgl_rotation_3d<double> head_rot = vgl_rotation_3d<double>(pose_as_quaternion);
    vgl_rotation_3d<double> flip_rot = vgl_rotation_3d<double>(0, vnl_math::pi, vnl_math::pi); // x,y,z
    //    std::cout<<"flip rotation "<<flip_rot.as_matrix()<<std::endl;
    vgl_rotation_3d<double> cam_R = flip_rot * head_rot;

    vgl_vector_3d<double> rot_off(0, 10, -50);

    vgl_vector_3d<double> T_off = (head_rot * rot_off) - rot_off;
    vgl_vector_3d<double> T_cam = cam_og.translation() + flip_rot * T_off;
    CAM_T cam_params(cam_og);
    cam_params.set_rotation(cam_R);
    cam_params.set_translation(T_cam);
    cam_params.set_image_size(nx, ny);
    return cam_params;
  }

void face3d::pose_jitterer_uniform::get_cameras(face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs,
                                                std::vector<CAM_T> & cameras, int num_jitters, bool random){
  float yaw_rad_min = vnl_math::pi/180 * min_yaw_;
  float yaw_rad_max = vnl_math::pi/180 * max_yaw_;
  dlib::matrix<double> yaws;
  if (!random){
    yaws = dlib::linspace(yaw_rad_min, yaw_rad_max, num_jitters );
  }else{

    yaws.set_size(1, num_jitters);
    for (unsigned i=0; i < num_jitters; i++){
      float yaw_rnd = yaw_rad_min + (this->rnd.get_random_float() * ((float) fabs(yaw_rad_max - yaw_rad_min) - 0.01));
      yaws(i) = yaw_rnd;
    }
  }
    vgl_rotation_3d<double> flip_rot = vgl_rotation_3d<double>(0, vnl_math::pi, vnl_math::pi); // x,y,z
    CAM_T cam_og = est_coeffs.camera(0);
    vgl_rotation_3d<double> cam_rot = est_coeffs.camera(0).rotation(); // x,y,z
    vgl_rotation_3d<double> R_im = flip_rot * cam_rot;
    vnl_vector_fixed<double, 3> euler_angles = face3d::quaternion_to_euler_angles(R_im.as_quaternion() ,"YXZ");
    double yaw = euler_angles[0];
    double pitch = euler_angles[1];
    double roll = euler_angles[2];
    for (unsigned i=0; i < num_jitters; i++){
      CAM_T new_cam = this->get_camera(cam_og, yaws(i), pitch, roll, cam_og.nx(), cam_og.ny());
      //      std::cout<< new_cam<<std::endl;
      cameras.push_back(new_cam);
    }
    return;
}

bool face3d::pose_jitterer_uniform::jitter_images(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                  face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs, int num_jitters,
                                                  std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images){


  std::vector<CAM_T> cam_vec;
  this->get_cameras(est_coeffs, cam_vec, num_jitters, false);
  this->init_renderer();
  vnl_vector<double> subject_coeffs = est_coeffs.subject_coeffs();
  vnl_vector<double> expression_coeffs = est_coeffs.expression_coeffs(0);
  std::thread::id this_id = std::this_thread::get_id();
  jitterer_.reset(new face3d::media_jitterer<CAM_T, dlib::matrix<dlib::rgb_pixel> >(input_images, est_coeffs.all_sightings(), this->base_mesh_,
                                                    *this->subject_components_.get(), *this->expression_components_.get(),
                                                                                    *this->renderer_, this->sym_map_dlib_, this->debug_dir_));
  //  return false;
  for (CAM_T & cam : cam_vec){
    dlib::matrix<dlib::rgb_pixel> last_rendered_no_alpha;
    bool success = this->jitterer_->render(cam, subject_coeffs, expression_coeffs,
                            *this->subject_components_.get(), *this-> expression_components_.get(),
                            last_rendered_no_alpha);
    if (success)
      output_images.push_back(std::move(last_rendered_no_alpha));
  }
  return  output_images.size() == num_jitters;;
}

bool face3d::pose_jitterer_uniform::multiple_random_jitters(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                      face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                                      int num_jitters,
                                                      std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images,
                                                      dlib::rand rnd){

  std::vector<CAM_T> cam_vec;
  this->get_cameras(est_coeffs, cam_vec, num_jitters, true);
  this->init_renderer();

  vnl_vector<double> subject_coeffs = est_coeffs.subject_coeffs();
  vnl_vector<double> expression_coeffs = est_coeffs.expression_coeffs(0);
 jitterer_.reset(new face3d::media_jitterer<CAM_T, dlib::matrix<dlib::rgb_pixel> >(input_images, est_coeffs.all_sightings(), this->base_mesh_,
                                                    *this->subject_components_.get(), *this->expression_components_.get(),
                                                                                    *this->renderer_, this->sym_map_dlib_, this->debug_dir_));

 for (CAM_T & cam : cam_vec){
    dlib::matrix<dlib::rgb_pixel> last_rendered_no_alpha;
    bool success = this->jitterer_->render(cam, subject_coeffs, expression_coeffs,
                            *this->subject_components_.get(), *this-> expression_components_.get(),
                            last_rendered_no_alpha);
    if (success)
      output_images.push_back(std::move(last_rendered_no_alpha));
  }
  return  output_images.size() == num_jitters;;


}
bool face3d::pose_jitterer_uniform::get_random_jitter(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                      face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                                      int num_jitters,
                                                      dlib::matrix<dlib::rgb_pixel> & output_image,
                                                      dlib::rand rnd){

  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_out_dlib;
  bool success = this->multiple_random_jitters(input_images, est_coeffs, 1, imgs_out_dlib);
  if (success){
    output_image = imgs_out_dlib[0];
    return true;
  }else{
    return false;
  }

}


void face3d::pose_jitterer_profile::get_cameras(face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs,
                                                std::vector<CAM_T> & cameras, int num_jitters, bool random){
    vgl_rotation_3d<double> flip_rot = vgl_rotation_3d<double>(0, vnl_math::pi, vnl_math::pi); // x,y,z
    CAM_T cam_og = est_coeffs.camera(0);
    vgl_rotation_3d<double> cam_rot = est_coeffs.camera(0).rotation(); // x,y,z
    vgl_rotation_3d<double> R_im = flip_rot * cam_rot;
    vnl_vector_fixed<double, 3> euler_angles = face3d::quaternion_to_euler_angles(R_im.as_quaternion() ,"YXZ");
    double yaw = euler_angles[0];
    double pitch = euler_angles[1];
    double roll = euler_angles[2];
    float yaw_deg = yaw * 180/ vnl_math::pi;
    dlib::matrix<double> yaws;
    float yaw_rad_min = yaw > 0 ? std::max(min_yaw_, yaw_deg)  : std::min(- max_yaw_, yaw_deg) ; // push to profile
    float yaw_rad_max = yaw > 0 ? std::max(max_yaw_, yaw_deg)  : std::min(- min_yaw_, yaw_deg) ;
    yaw_rad_min *= vnl_math::pi/180 ;
    yaw_rad_max *= vnl_math::pi/180 ;

    if (!random){
      yaws = dlib::linspace( yaw_rad_min, yaw_rad_max, num_jitters );
    }else{
      yaws.set_size(1, num_jitters);
      for (unsigned i=0; i < num_jitters; i++){
        float yaw_rnd = yaw_rad_min + (this->rnd.get_random_float() * ((float) fabs(yaw_rad_max - yaw_rad_min) - 0.01));
        yaws(i) = yaw_rnd;
         if (this->debug_dir_!=""){
           std::cout<<"[face3d::pose_jitterer_profile::get_cameras()]: selected yaw "<<180/vnl_math::pi * yaw_rnd<<" curr: "<<yaw_deg<<" min: "<<180 / vnl_math::pi * yaw_rad_min <<" max:"<<180 / vnl_math::pi * yaw_rad_max<<std::endl;
         }
      }
    }
    //std::cout<<"[face3d::pose_jitterer_profile]: yaw range:"<<yaws<<std::endl;
    for (unsigned i=0; i < num_jitters; i++){

      CAM_T new_cam = this->get_camera(cam_og, yaws(i), pitch, roll, cam_og.nx(), cam_og.ny());
      cameras.push_back(new_cam);
    }
    return;
}

bool face3d::pose_jitterer_profile::jitter_images(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                  face3d::subject_sighting_coefficients<CAM_T> const& est_coeffs, int num_jitters,
                                                  std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images){


  std::vector<CAM_T> cam_vec;
  this->get_cameras(est_coeffs, cam_vec, num_jitters, false);
  this->init_renderer();
  vnl_vector<double> subject_coeffs = est_coeffs.subject_coeffs();
  vnl_vector<double> expression_coeffs = est_coeffs.expression_coeffs(0);
  std::thread::id this_id = std::this_thread::get_id();
  jitterer_.reset(new face3d::media_jitterer<CAM_T, dlib::matrix<dlib::rgb_pixel> >(input_images, est_coeffs.all_sightings(), this->base_mesh_,
                                                                                    *this->subject_components_.get(), *this->expression_components_.get(),
                                                                                    *this->renderer_, this->sym_map_dlib_, this->debug_dir_, false));
  //  return false;
  for (CAM_T & cam : cam_vec){
    dlib::matrix<dlib::rgb_pixel> last_rendered_no_alpha;
    bool success = this->jitterer_->render(cam, subject_coeffs, expression_coeffs,
                                           *this->subject_components_.get(), *this->expression_components_.get(),
                                           last_rendered_no_alpha);
    if (success)
      output_images.push_back(std::move(last_rendered_no_alpha));
    else{
      std::cout<<"[face3d::pose_jitterer::jitter_images]: could not render from camera "<<cam<<std::endl;
    }
  }
  return  output_images.size() == num_jitters;
}
bool face3d::pose_jitterer_profile::multiple_random_jitters(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                      face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                                      int num_jitters,
                                                      std::vector<dlib::matrix<dlib::rgb_pixel> >& output_images,
                                                      dlib::rand rnd){

  std::vector<CAM_T> cam_vec;
  this->get_cameras(est_coeffs, cam_vec, num_jitters, true);
  this->init_renderer();

  vnl_vector<double> subject_coeffs = est_coeffs.subject_coeffs();
  vnl_vector<double> expression_coeffs = est_coeffs.expression_coeffs(0);
  jitterer_.reset(new face3d::media_jitterer<CAM_T, dlib::matrix<dlib::rgb_pixel> >(input_images, est_coeffs.all_sightings(), this->base_mesh_,
                                                                                    *this->subject_components_.get(), *this->expression_components_.get(),
                                                                                    *this->renderer_, this->sym_map_dlib_, this->debug_dir_, false));

  for (CAM_T & cam : cam_vec){
    dlib::matrix<dlib::rgb_pixel> last_rendered_no_alpha;
    bool success = this->jitterer_->render(cam, subject_coeffs, expression_coeffs,
                                           *this->subject_components_.get(), *this-> expression_components_.get(),
                                           last_rendered_no_alpha);
    if (success)
      output_images.push_back(std::move(last_rendered_no_alpha));
    else{
      std::cout<<"[face3d::pose_jitterer::multiple_random_jitters]: could not render from camera "<<cam<<std::endl;
    }
  }
  return  output_images.size() == num_jitters;

}

bool face3d::pose_jitterer_profile::get_random_jitter(std::vector<dlib::matrix<dlib::rgb_pixel> >const & input_images,
                                                            face3d::subject_sighting_coefficients<CAM_T>const& est_coeffs,
                                                            int num_jitters,
                                                            dlib::matrix<dlib::rgb_pixel> & output_image,
                                                            dlib::rand rnd){
  std::vector<dlib::matrix<dlib::rgb_pixel> > imgs_out_dlib;
  bool success = this->multiple_random_jitters(input_images, est_coeffs, 1, imgs_out_dlib);
  if (success){
    output_image = imgs_out_dlib[0];
    return true;
  }else{
    return false;
  }
}
