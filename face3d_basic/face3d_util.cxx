#include "face3d_util.h"
#include <vector>
#include <vgl/vgl_vector_3d.h>
#include <vgl/vgl_point_3d.h>


void face3d_util::apply_coefficients(std::vector<vgl_point_3d<double> >  const& verts,
                                     vnl_matrix<double> const& subject_components,
                                     vnl_matrix<double> const& expression_components,
                                     vnl_vector<double> const& subject_coeffs,
                                     vnl_vector<double> const& expression_coeffs,
                                     std::vector<vgl_point_3d<double> > &warped_verts)
{
  const int num_subject_coeffs = subject_coeffs.size();
  const int num_expression_coeffs = expression_coeffs.size();
  if (num_subject_coeffs > subject_components.rows()) {
    throw std::logic_error("Number of subject coefficients exceeds number of components");
  }
  if (num_expression_coeffs > expression_components.rows()) {
    throw std::logic_error("Number of expression coefficients exceeds number of components");
  }
  warped_verts.clear();
  const int num_verts = verts.size();
  vnl_vector<double> offsets_flat(num_verts*3, 0.0);
  if (subject_coeffs.size() > 0) {
    // offsets_flat += subject_components.get_n_rows(0,num_subject_coeffs).transpose()*subject_coeffs;
    offsets_flat += subject_coeffs * subject_components.get_n_rows(0,num_subject_coeffs);
  }
  if (expression_coeffs.size() > 0) {
    //offsets_flat += expression_components.get_n_rows(0,num_expression_coeffs).transpose()*expression_coeffs;
    offsets_flat += expression_coeffs * expression_components.get_n_rows(0,num_expression_coeffs);
  }
  if (offsets_flat.size() != num_verts*3) {
    throw std::runtime_error("Unexpected vector size");
  }
  for (int v=0; v<num_verts; ++v) {
    int off = v*3;
    vgl_vector_3d<double> offset(offsets_flat[off+0],
                                 offsets_flat[off+1],
                                 offsets_flat[off+2]);
    warped_verts.push_back(verts[v] + offset);
  }
  return;
}
