#pragma once
#include <dlib/pixel.h>
#include <dlib/array2d.h>
#include <dlib/image_transforms.h>
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_vector_3d.h>
#include <dlib/geometry.h>
namespace face3d_img_util {

  template <class T>
  struct pixel_traits;

  template <>
  struct pixel_traits<dlib::rgb_pixel>
  {
    static const int nplanes = 3;
    using dtype = unsigned char;
  };
  template <>
  struct pixel_traits<dlib::rgb_alpha_pixel>
  {
    static const int nplanes = 4;
    using dtype = unsigned char;
  };
  template<class T, long N>
  struct pixel_traits<dlib::vector<T,N> >
  {
    static const int nplanes = N;
    using dtype = T;
  };
  template <>
  struct pixel_traits<vgl_point_3d<float> >
  {
    static const int nplanes = 3;
    using dtype = float;
  };
  template <>
  struct pixel_traits<vgl_vector_3d<float> >
  {
    static const int nplanes = 3;
    using dtype = float;
  };
  template <>
  struct pixel_traits<float>
  {
    static const int nplanes = 1;
    using dtype = float;
  };
  template <>
  struct pixel_traits<int>
  {
    static const int nplanes = 1;
    using dtype = int;
  };
  template <>
  struct pixel_traits<unsigned char>
  {
    static const int nplanes = 1;
    using dtype = unsigned char;
  };

  template <class T>
  struct img_traits;

  //template <>
  template <class PIX_T>
  struct img_traits<dlib::array2d<PIX_T> >
  {
    using pixel_type = PIX_T;
    static const int nplanes = pixel_traits<PIX_T>::nplanes;
    using dtype = typename pixel_traits<PIX_T>::dtype;
  };

  //template <>
  template <class PIX_T>
  struct img_traits<dlib::matrix<PIX_T> >
  {
    using pixel_type = PIX_T;
    static const int nplanes = pixel_traits<PIX_T>::nplanes;
    using dtype = typename pixel_traits<PIX_T>::dtype;
  };

  //template <>
  template <class PIX_T>
  struct img_traits<vil_image_view<PIX_T> >
  {
    // nplanes is unknown at compile time for a vil_image_view
    using dtype = typename pixel_traits<PIX_T>::dtype;
  };


  void assign_pixel(dlib::rgb_pixel& pixel, unsigned char* pix_ptr, size_t plane_stride);
  void assign_pixel(dlib::rgb_alpha_pixel& pixel, unsigned char* pix_ptr, size_t plane_stride);
  void assign_pixel(vgl_point_3d<float>& pixel, float* pix_ptr, size_t plane_stride);
  void assign_pixel(vgl_vector_3d<float>& pixel, float* pix_ptr, size_t plane_stride);

  template<class T>
  void assign_pixel(dlib::vector<T,2>& pixel, T* pix_ptr, size_t plane_stride)
  {
    pixel.x() =  *(pix_ptr + 0*plane_stride);
    pixel.y() =  *(pix_ptr + 1*plane_stride);
  }
  template<class T>
  void assign_pixel(dlib::vector<T,3>& pixel, T* pix_ptr, size_t plane_stride)
  {
    pixel.x() =  *(pix_ptr + 0*plane_stride);
    pixel.y() =  *(pix_ptr + 1*plane_stride);
    pixel.z() =  *(pix_ptr + 2*plane_stride);
  }
  // implementation for trivial cases where pixel type and ptr type match.
  template<class T>
  void assign_pixel(T& pixel, T* pix_ptr, size_t plane_stride)
  {
    pixel = *pix_ptr;
  }

  template<class T1, class T2>
  void assign_pixel(dlib::array2d<T1> &img, int r, int c, T2* data_p, size_t plane_stride)
  {
    if (img.nr() <= r || img.nc()<= c){
      std::cerr<<"[face3d_img_util]::assign_pixel: output image size is ["<<img.nr()<< " , "<< img.nc()<<"] and pixel at ("<<r<<" , "<<c<<") exceeds bounds"<<std::endl;
      throw std::range_error("[face3d_img_util]::assign_pixel failed");
    }

    assign_pixel(img[r][c], data_p, plane_stride);
  }

  template<class MATRIX_T, class T2>
  void assign_pixel(MATRIX_T &img, int r, int c, T2* data_p, size_t plane_stride)
  {
    if (img.nr() <= r || img.nc()<= c){
      std::cerr<<"[face3d_img_util]::assign_pixel: output image size is ["<<img.nr()<< " , "<< img.nc()<<"] and pixel at ("<<r<<" , "<<c<<") exceeds bounds"<<std::endl;
      throw std::range_error("[face3d_img_util]::assign_pixel failed");
    }

    assign_pixel(img(r,c), data_p, plane_stride);
  }

  template <class PIXEL_T>
  PIXEL_T& get_pixel(dlib::array2d<PIXEL_T> & image, int r, int c){
    return image[r][c];
  };
  template <class PIXEL_T>
  PIXEL_T& get_pixel(dlib::matrix<PIXEL_T> & image, int r, int c){
    return image(r,c);
  };


  template<class T>
  const void* get_data_ptr(dlib::array2d<T> const& obj)
  {
    return static_cast<const void*>(&obj[0][0]);
  }

  template<class T>
  const void* get_data_ptr(dlib::matrix<T> const& obj)
  {
    return static_cast<const void*>(&obj(0,0));
  }

  template<class IMG_T>
  void set_color_img_size(IMG_T& img, unsigned nr, unsigned nc){
    img.set_size(nr, nc);
  }

  // the "assign_image" function is meant to mimic dlib's "assign_image", except that
  // it will work with objects of type vil_image_view as well.
  template<class PIX_T_SRC, class PIX_T_DEST>
  void assign_image(dlib::array2d<PIX_T_DEST> &dest, dlib::array2d<PIX_T_SRC> const &src)
  {
    dlib::assign_image(dest, src);
  }

  template<class PIX_T_SRC, class PIX_T_DEST>
  void assign_image(dlib::matrix<PIX_T_DEST> &dest, dlib::matrix<PIX_T_SRC> const &src)
  {
    dlib::assign_image(dest, src);
  }

  template <class T>
  void assign_image(vil_image_view<T> &dest, vil_image_view<T> const& src)
  {
    dest.deep_copy(src);
  }

}
