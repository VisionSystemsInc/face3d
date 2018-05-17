#include "dlib_face_detector.h"

#include <string>
#include <vector>

#include <dlib/image_processing.h>
#include <dlib/geometry/rectangle.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <vil/vil_convert.h>

#include "image_conversion.h"

using std::string;
using std::vector;

using namespace face3d;

dlib_face_detector::dlib_face_detector()
  : dlib_object_detector(dlib::get_frontal_face_detector())
{
}

dlib_face_detector::dlib_face_detector(std::string model_filename)
  : dlib_object_detector(model_filename)
{
}
