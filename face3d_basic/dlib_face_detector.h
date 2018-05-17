#ifndef dlib_face_detector_h_included_
#define dlib_face_detector_h_included_

#include <string>
#include <map>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_box_2d.h>

#include "dlib_object_detector.h"

namespace face3d {

class dlib_face_detector : public dlib_object_detector
{
public:

    dlib_face_detector();

    dlib_face_detector(std::string model_filename);
};

}

#endif
