#ifndef object_detector_h_included_
#define object_detector_h_included_

#include <vector>
#include <string>
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>


namespace face3d {

//: Interface class for detecting objects in an image
class object_detector
{
public:
  virtual ~object_detector(){}

  //: set the operating image - (convert to internal rep. if necessary)
  virtual bool set_image(vil_image_view<unsigned char> const& image) = 0;

  //: return all faces detected in the image
  virtual bool detect(std::vector<vgl_box_2d<double> > &object_boxes) = 0;

};

}

#endif
