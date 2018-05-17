#ifndef dlib_object_detector_h_included_
#define dlib_object_detector_h_included_

#include <vector>
#include <string>
#include <map>
#include <dlib/image_processing.h>
#include <dlib/image_processing/object_detector.h>
#include <dlib/image_processing/scan_fhog_pyramid.h>
#include <dlib/image_transforms/image_pyramid.h>
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>

namespace face3d {

class dlib_object_detector
{
public:
  typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > detector_type;

  //: constructor - load model from disk
  dlib_object_detector(std::string model_filename);

  //: constuctor - pass in model directly
  dlib_object_detector(detector_type detector);

  //: detect the object (native dlib rectangle interface)
  template <typename pixel_type>
    bool detect(dlib::array2d<pixel_type> const& img,
                std::vector<dlib::rectangle> &det_boxes) {
      det_boxes = detector_(img);
      return true;
    }

  //: detect the object (vxl box_2d interface)
  template <typename pixel_type>
    bool detect(dlib::array2d<pixel_type> const& img, std::vector<vgl_box_2d<double> > &det_boxes) {
      // detect object in input image
      std::vector<dlib::rectangle> dets = detector_(img);
      det_boxes.clear();
      for (std::vector<dlib::rectangle>::const_iterator dit = dets.begin(); dit != dets.end(); ++dit) {
        det_boxes.push_back(vgl_box_2d<double>(dit->left(),dit->right(), dit->top(), dit->bottom()));
      }
      return true;
    }

private:
    detector_type detector_;
};

}

#endif
