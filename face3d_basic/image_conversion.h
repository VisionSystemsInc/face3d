#ifndef image_conversion_h_included_
#define image_conversion_h_included_

#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/matrix.h>
#include <vil/vil_image_view.h>
#include <vil/vil_convert.h>
#include <vgl/vgl_point_3d.h>
#include "face3d_img_util.h"

namespace face3d
{
  class image_conversion {
    public:

      // convert a vil_image_view to a dlib::array2d
      template <class VIL_T, class DLIB_T>
      static bool convert_vil_image_to_dlib_array(vil_image_view<VIL_T> const& img_in,
                                                  DLIB_T &img_out)
      {
        typedef typename dlib::image_traits<DLIB_T>::pixel_type pixel_type;
        // avoid converting grayscale image to color
        if ((dlib::pixel_traits<pixel_type>::num >= 3) && (img_in.nplanes() < 3)) {
          std::cerr << "ERROR: image_conversion::convert_vil_image_to_dlib_array(): Input image does not have 3 or 4 planes (has " << img_in.nplanes() << ")" << std::endl;
          return false;
        }
        vil_image_view<VIL_T> img_in_t = img_in;
        // convert to grayscale if necessary
        if ((dlib::pixel_traits<pixel_type>::num == 1) && (img_in.nplanes() >= 3)) {
          vil_convert_planes_to_grey(img_in, img_in_t);
        }
        // convert the vil image to dlib's representation
        const unsigned nx = img_in.ni();
        const unsigned ny = img_in.nj();
        img_out.set_size(ny, nx);  // dlib indexes rows, col
        // convert vil_image to dlib array2d
        for (unsigned y=0; y<ny; ++y) {
          for (unsigned x=0; x<nx; ++x) {
            convert_pixel_of(img_in_t, x, y, img_out);
          }
        }
        return true;
      }

      template <class VIL_T, class DLIB_T>
      static bool convert_dlib_array_to_vil_image(DLIB_T const& img_in,
                                                  vil_image_view<VIL_T> & img_out)
      {
        const int nx = img_in.nc();
        const int ny = img_in.nr();
        const int nplanes = face3d_img_util::img_traits<DLIB_T>::nplanes;
        img_out.set_size(nx, ny, nplanes);

        // convert the pixel values
        for (unsigned y=0; y<ny; ++y) {
          for (unsigned x=0; x<nx; ++x) {
            convert_pixel_of(img_in, x, y, img_out);
          }
        }
        return true;
      }

    private:

      static inline void convert_pixel_of(dlib::array2d<dlib::rgb_alpha_pixel> const& img_in, int x, int y, vil_image_view<unsigned char> &img_out)
      {
        dlib::rgb_alpha_pixel in = img_in[y][x];
        img_out(x,y,0) = in.red;
        img_out(x,y,1) = in.green;
        img_out(x,y,2) = in.blue;
        img_out(x,y,3) = in.alpha;
      }

      static inline void convert_pixel_of(dlib::array2d<dlib::rgb_pixel> const& img_in, int x, int y, vil_image_view<unsigned char> &img_out)
      {
        dlib::rgb_pixel in = img_in[y][x];
        img_out(x,y,0) = in.red;
        img_out(x,y,1) = in.green;
        img_out(x,y,2) = in.blue;
      }

      static inline void convert_pixel_of(dlib::array2d<vgl_point_3d<float> > const& img_in, int x, int y, vil_image_view<float> &img_out)
      {
        vgl_point_3d<float> in( img_in[y][x] );
        img_out(x,y,0) = in.x();
        img_out(x,y,1) = in.y();
        img_out(x,y,2) = in.z();
      }

      static inline void convert_pixel_of(dlib::array2d<vgl_vector_3d<float> > const& img_in, int x, int y, vil_image_view<float> &img_out)
      {
        vgl_vector_3d<float> in( img_in[y][x] );
        img_out(x,y,0) = in.x();
        img_out(x,y,1) = in.y();
        img_out(x,y,2) = in.z();
      }

      template<class T1, class T2>
      static inline void convert_pixel_of(dlib::array2d<T1> const& img_in, int x, int y, vil_image_view<T2> &img_out)
      {
        T1 in = img_in[y][x];
        img_out(x,y) = in;
      }

      template <class T1, class T2>
      static inline void convert_pixel_of(vil_image_view<T1> const& img_in, int x, int y, dlib::matrix<T2> &m)
      {
        convert_pixel(img_in, x, y, m(y,x));
      }

      template <class T1, class T2>
      static inline void convert_pixel_of(vil_image_view<T1> const& img_in, int x, int y, dlib::array2d<T2> &a)
      {
        convert_pixel(img_in, x, y, a[y][x]);
      }

      static inline void convert_pixel(vil_image_view<unsigned char> const& img_in, int x, int y, dlib::rgb_pixel& dlib_pixel)
      {
        dlib_pixel = dlib::rgb_pixel(img_in(x,y,0),
                                     img_in(x,y,1),
                                     img_in(x,y,2));
      }

      template <class T>
      static inline void convert_pixel(vil_image_view<T> const& img_in, int x, int y, dlib::bgr_pixel& dlib_pixel)
      {
        dlib_pixel = dlib::bgr_pixel(img_in(x,y,2),
                                     img_in(x,y,1),
                                     img_in(x,y,0));
      }

      template <class T>
      static inline void convert_pixel(vil_image_view<T> const& img_in, int x, int y, T& dlib_pixel)
      {
        dlib_pixel = img_in(x,y);
      }

};

}

#endif
