#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vnl/vnl_vector.h>

namespace face3d {

template <class CAM_T>
class sighting_coefficients
{
  public:
    sighting_coefficients(){}

    sighting_coefficients(int num_subj_coeffs, int num_expr_coeffs);

    sighting_coefficients(vnl_vector<double> const& subj_coeffs,
                          vnl_vector<double> const& expression_coeffs,
                          CAM_T const& camera_params);

    bool save(std::string const& filename) const;
    bool write(std::ostream &ofs) const;
    bool read(std::istream &ifs);

    vnl_vector<double> const& subject_coeffs() const {return subj_coeffs_;}
    vnl_vector<double> const& expression_coeffs() const {return expr_coeffs_;}
    CAM_T const& camera() const {return cam_params_;}

  private:
      vnl_vector<double> subj_coeffs_;
      vnl_vector<double> expr_coeffs_;
      CAM_T cam_params_;
};


//--------- Member Definitions -------//

template<class CAM_T>
sighting_coefficients<CAM_T>::
sighting_coefficients(int num_subj_coeffs, int num_expr_coeffs)
  : subj_coeffs_(num_subj_coeffs,0.0), expr_coeffs_(num_expr_coeffs,0.0), cam_params_()
{
}

template<class CAM_T>
sighting_coefficients<CAM_T>::
sighting_coefficients(vnl_vector<double> const& subj_coeffs,
                      vnl_vector<double> const& expression_coeffs,
                      CAM_T const& camera_params)
  : subj_coeffs_(subj_coeffs), expr_coeffs_(expression_coeffs), cam_params_(camera_params)
{
}

template<class CAM_T>
bool sighting_coefficients<CAM_T>::
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
bool sighting_coefficients<CAM_T>::
write(std::ostream &ofs) const
{
  if (!ofs) {
    std::cerr << "ERROR: bad ofstream" << std::endl;
    return false;
  }
  ofs << subj_coeffs_ << std::endl;
  ofs << expr_coeffs_ << std::endl;
  ofs << cam_params_ << std::endl;

  return true;
}

template<class CAM_T>
bool sighting_coefficients<CAM_T>::
read(std::istream &ifs)
{
  if (!ifs) {
    std::cerr << "ERROR: bad ifstream" << std::endl;
    return false;
  }
  std::string subj_coeffs_str;
  std::getline(ifs, subj_coeffs_str);
  std::stringstream subj_coeffs_ss(subj_coeffs_str);
  subj_coeffs_ss >> subj_coeffs_;

  std::string expr_coeffs_str;
  std::getline(ifs, expr_coeffs_str);
  std::stringstream expr_coeffs_ss(expr_coeffs_str);
  expr_coeffs_ss >> expr_coeffs_;

  std::string cam_params_str;
  std::getline(ifs, cam_params_str);
  std::stringstream cam_params_ss(cam_params_str);
  cam_params_ss >> cam_params_;

  return true;
}


} // namespace face3d
