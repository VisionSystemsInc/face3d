#ifndef subject_sighting_coefficients_h_included_
#define subject_sighting_coefficients_h_included_
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vnl/vnl_vector.h>

#include "sighting_coefficients.h"

namespace face3d {

template <class CAM_T>
class subject_sighting_coefficients
{
  public:
    subject_sighting_coefficients(){}

    subject_sighting_coefficients(std::vector<std::string> const& img_ids, int num_subj_coeffs, int num_expr_coeffs);

    subject_sighting_coefficients(vnl_vector<double> const& subj_coeffs,
                                 std::vector<std::string> const& image_filenames,
                                 std::vector<vnl_vector<double> > const& expression_coeffs,
                                 std::vector<CAM_T> const& camera_params);

    subject_sighting_coefficients(std::string const& filename);

    bool save(std::string const& filename) const;

    bool write(std::ostream &ofs) const;
    bool read(std::istream &ifs);

    int num_sightings() const {return image_fnames_.size();}
    vnl_vector<double> const& subject_coeffs() const {return subj_coeffs_;}
    vnl_vector<double> const& expression_coeffs(int idx) const {return expression_coeffs_[idx];}
    std::string image_filename(int idx) const {return image_fnames_[idx];}
    CAM_T const& camera(int idx) const {return cam_params_[idx];}

    vnl_vector<double> pack() const;
    void unpack(vnl_vector<double> const& x);

    void set_subject_coeffs(vnl_vector<double> const& subj_coeffs) { subj_coeffs_ = subj_coeffs;}

    // return the single-image sighting coefficients for image idx
    sighting_coefficients<CAM_T> sighting(int idx) const;
    std::vector<sighting_coefficients<CAM_T>> all_sightings() const;

  protected:
      vnl_vector<double> subj_coeffs_;
      std::vector<std::string> image_fnames_;
      std::vector<vnl_vector<double> > expression_coeffs_;
      std::vector<CAM_T> cam_params_;

};

//------------- Member Definitions ------------//


template<class CAM_T>
subject_sighting_coefficients<CAM_T>::subject_sighting_coefficients(std::vector<std::string> const& img_ids, int num_subj_coeffs, int num_expr_coeffs):
  image_fnames_(img_ids),
  subj_coeffs_(num_subj_coeffs,0.0),
  expression_coeffs_(img_ids.size(), vnl_vector<double>(num_expr_coeffs,0.0)),
  cam_params_(img_ids.size())
{
}

template<class CAM_T>
subject_sighting_coefficients<CAM_T>::
subject_sighting_coefficients(vnl_vector<double> const& subj_coeffs,
                              std::vector<std::string> const& image_filenames,
                              std::vector<vnl_vector<double> > const& expression_coeffs,
                              std::vector<CAM_T> const& camera_params)
  : subj_coeffs_(subj_coeffs),
  image_fnames_(image_filenames), expression_coeffs_(expression_coeffs), cam_params_(camera_params)
{
  // sanity check on inputs
  const int num_sightings = image_filenames.size();
  if (expression_coeffs.size() != num_sightings) {
    throw std::runtime_error("Number of expression coefficient vectors does not match number of images.");
  }
  if (camera_params.size() != num_sightings) {
    throw std::runtime_error("Number of cameras does not match number of images.");
  }
}

template<class CAM_T>
subject_sighting_coefficients<CAM_T>::
subject_sighting_coefficients(std::string const& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs) {
    std::cerr << "ERROR opening " << filename << " for read" << std::endl;
    throw std::runtime_error("Error opening input file");
  }
  if(!read(ifs)) {
    throw std::runtime_error("Error reading input from file");
  }
}

template<class CAM_T>
bool subject_sighting_coefficients<CAM_T>::
save(std::string const& filename) const
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()) {
    std::cerr << "ERROR opening " << filename << " for write." << std::endl;
    return false;
  }
  return write(ofs);
}

template<class CAM_T>
bool subject_sighting_coefficients<CAM_T>::
write(std::ostream &ofs) const
{
  if (!ofs) {
    std::cerr << "ERROR: bad ofstream" << std::endl;
    return false;
  }
  const int num_obs = this->num_sightings();
  ofs << subj_coeffs_ << std::endl;
  for (int c=0; c<num_obs; ++c) {
    ofs << image_fnames_[c] << "     " << cam_params_[c] << "     " << expression_coeffs_[c] << std::endl;
  }
  return true;
}

template<class CAM_T>
bool subject_sighting_coefficients<CAM_T>::
read(std::istream &ifs)
{
  if (!ifs) {
    throw std::runtime_error("invalid input stream");
  }
  std::string subj_coeffs_str;
  std::getline(ifs, subj_coeffs_str);
  std::stringstream subj_coeffs_ss(subj_coeffs_str);
  subj_coeffs_ss >> subj_coeffs_;

  std::vector<std::string> image_filenames;
  std::vector<vnl_vector<double> > expression_coeffs;
  std::vector<CAM_T> camera_params;

  image_fnames_.clear();
  expression_coeffs_.clear();
  cam_params_.clear();

  for (std::string sighting_str; std::getline(ifs, sighting_str);) {
    std::stringstream sighting_ss(sighting_str);
    std::string image_fname;
    if(!(sighting_ss >> image_fname)) {
      std::cerr << "ERROR reading image filename" << std::endl;
      return false;
    }
    CAM_T this_cam;
    if(!(sighting_ss >> this_cam)) {
      std::cerr << "Error reading camera parameters for image " << image_fname << std::endl;
      return false;
    }
    vnl_vector<double> this_expression(0);
    // this will read to EOF since size of this_expression is unknown
    sighting_ss >> this_expression;

    image_fnames_.push_back(image_fname);
    cam_params_.push_back(this_cam);
    expression_coeffs_.push_back(this_expression);
  }
  return true;
}

template<class CAM_T>
vnl_vector<double> subject_sighting_coefficients<CAM_T>::
pack() const
{
  const int num_images = expression_coeffs_.size();
  const int num_expression_coeffs = num_images > 0? expression_coeffs_[0].size() : 0;
  const int num_subject_coeffs = subj_coeffs_.size();

  const int num_parameters = num_subject_coeffs + num_images*(CAM_T::num_params() + num_expression_coeffs);
  vnl_vector<double> x(num_parameters);
  for (int i=0; i<num_subject_coeffs; ++i) {
    x[i] = subj_coeffs_[i];
  }
  for (int n=0; n<num_images; ++n) {
    int offset = num_subject_coeffs + n*(CAM_T::num_params() + num_expression_coeffs);
    vnl_vector<double> cam_packed = cam_params_[n].pack();
    for (int i=0; i<CAM_T::num_params(); ++i) {
      x[offset + i] = cam_packed[i];
    }
    offset += CAM_T::num_params();
    for (int i=0; i<num_expression_coeffs; ++i) {
      x[offset + i] = expression_coeffs_[n][i];
    }
  }
  return x;
}

template<class CAM_T>
void subject_sighting_coefficients<CAM_T>::
unpack(vnl_vector<double> const& x)
{
  const int num_images = expression_coeffs_.size();
  const int num_expression_coeffs = num_images > 0? expression_coeffs_[0].size() : 0;
  const int num_subject_coeffs = subj_coeffs_.size();
  const int num_parameters = num_subject_coeffs + num_images*(CAM_T::num_params() + num_expression_coeffs);
  if (x.size() != num_parameters) {
    throw std::runtime_error("unpack() got argument of wrong size");
  }
  for (int i=0; i<num_subject_coeffs; ++i) {
    subj_coeffs_[i] = x[i];
  }
  for (int n=0; n<num_images; ++n) {
    int offset = num_subject_coeffs + n*(CAM_T::num_params() + num_expression_coeffs);
    vnl_vector<double> cam_packed(CAM_T::num_params());
    for (int i=0; i<CAM_T::num_params(); ++i) {
      cam_packed[i] = x[offset+i];
    }
    cam_params_[n].unpack(cam_packed);
    offset += CAM_T::num_params();
    for (int i=0; i<num_expression_coeffs; ++i) {
      expression_coeffs_[n][i] = x[offset + i];
    }
  }
}

// return the single-image sighting coefficients for image idx
template<class CAM_T>
sighting_coefficients<CAM_T> subject_sighting_coefficients<CAM_T>::
sighting(int idx) const
{
  if ((idx < 0) || (idx >= expression_coeffs_.size())) {
    throw std::logic_error("invalid index passed to subject_sighting_coefficients::sighting_coefficients()");
  }
  return sighting_coefficients<CAM_T>(subj_coeffs_, expression_coeffs_[idx], cam_params_[idx]);
}

template<class CAM_T>
std::vector<sighting_coefficients<CAM_T>> subject_sighting_coefficients<CAM_T>::
all_sightings() const
{
  const int num_sightings = expression_coeffs_.size();
  std::vector<sighting_coefficients<CAM_T> > s;
  s.reserve(num_sightings);
  for(int i=0; i<num_sightings; ++i) {
    s.push_back(sighting_coefficients<CAM_T>(subj_coeffs_,
                                             expression_coeffs_[i],
                                             cam_params_[i]));
  }
  return s;
}


} // namespace face3d
#endif
