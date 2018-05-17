#pragma once

#include <dlib/array2d.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_vector_3d.h>

namespace face3d {

bool correct_offsets(dlib::array2d<vgl_point_3d<float> > const& PNCC,
                     dlib::array2d<vgl_vector_3d<float> > const& offsets,
                     dlib::array2d<vgl_vector_3d<float> > &offsets_out);


}
