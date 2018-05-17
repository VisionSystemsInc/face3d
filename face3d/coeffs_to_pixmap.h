#pragma once

#include "head_mesh.h"
#include <dlib/array2d.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vgl/vgl_point_3d.h>

namespace face3d {

dlib::array2d<vgl_point_3d<float> >
coeffs_to_pixmap(head_mesh const& mesh,
                 vnl_vector<double> const& subject_coeffs, vnl_vector<double> const &expression_coeffs,
                 vnl_matrix<double> const& subject_components, vnl_matrix<double> const& expression_components);


}
