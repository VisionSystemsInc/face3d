#ifndef face3d_util_h_included_
#define face3d_util_h_included_

#include <vil/vil_image_view.h>
#include <vgl/vgl_box_2d.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>

#include <face3d_basic/dlib_object_detector.h>
#include <dlib/array2d.h>

namespace face3d_util {

  void apply_coefficients(std::vector<vgl_point_3d<double> >  const& verts,
                          vnl_matrix<double> const& subject_components,
                          vnl_matrix<double> const& expression_components,
                          vnl_vector<double> const& subject_coeffs,
                          vnl_vector<double> const& expression_coeffs,
                          std::vector<vgl_point_3d<double> > &warped_verts);


  template<class T>
  bool get_biggest_face_detection(dlib::array2d<T> const& img, face3d::dlib_object_detector &detector,
                                  vgl_box_2d<double> &bbox_out)
  {
    std::vector<vgl_box_2d<double> > det_boxes;
    bool status = detector.detect(img, det_boxes);
    if(!status || (det_boxes.size()==0)) {
      return false;
    }
    bbox_out = det_boxes[0];
    double max_area = det_boxes[0].volume();
    for (int i=1; i<det_boxes.size(); ++i) {
      double this_area = det_boxes[i].volume();
      if (this_area > max_area) {
        max_area = this_area;
        bbox_out = det_boxes[i];
      }
    }
    return true;
  }

}

#endif
