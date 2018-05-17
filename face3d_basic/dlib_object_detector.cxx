#include "dlib_object_detector.h"

#include <string>

#include <dlib/image_processing.h>
#include <dlib/geometry/rectangle.h>
#include <dlib/image_processing/object_detector.h>

#include <vil/vil_convert.h>

#include "image_conversion.h"

using std::string;

using namespace face3d;

dlib_object_detector::dlib_object_detector(string model_filename)
{
  dlib::deserialize(model_filename) >> detector_;
}

dlib_object_detector::dlib_object_detector(detector_type detector)
  : detector_(detector)
{
}
