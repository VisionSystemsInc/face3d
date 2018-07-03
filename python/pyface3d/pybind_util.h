#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <dlib/pixel.h>
#include <dlib/array2d.h>
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_vector_3d.h>
#include <face3d_basic/face3d_img_util.h>

namespace pybind_util{

  template <class BUFF_T, class EIG_MAT_T>
  void matrix_from_buffer(pybind11::array_t<BUFF_T>& buff, EIG_MAT_T &mat)
  {
    pybind11::buffer_info info = buff.request();
    if (info.ndim != 2) {
      throw std::runtime_error("Expecting a buffer with shape RxC");
    }
    const size_t nrows = info.shape[0];
    const size_t ncols = info.shape[1];
    mat.resize(nrows, ncols);
    // deep copy of data, performing any needed type conversions
    const BUFF_T *data_ptr = static_cast<BUFF_T*>(info.ptr);
    const size_t row_stride = info.strides[0] / sizeof(BUFF_T);
    const size_t col_stride = info.strides[1] / sizeof(BUFF_T);
    for (size_t r=0; r<nrows; ++r) {
      const BUFF_T *row_ptr = data_ptr + r*row_stride; 
      for (size_t c=0; c<ncols; ++c) {
        mat(r,c) = *(row_ptr + c*col_stride);
      }
    }
  }

  template <class BUFF_T, class EIG_MAT_T>
  void matrix_to_buffer(EIG_MAT_T const &mat, pybind11::array_t<BUFF_T>& buff)
  {
    const size_t num_rows = mat.rows();
    const size_t num_cols = mat.cols();

    std::vector<size_t> shape = { num_rows, num_cols};
    std::vector<size_t> strides = {num_cols*sizeof(BUFF_T), sizeof(BUFF_T)};

    buff = pybind11::array_t<BUFF_T>(shape, strides);
    pybind11::buffer_info info = buff.request();
    BUFF_T *ptr = static_cast<BUFF_T*>(info.ptr);

    // make sure the buffer actually has shape and strides requested
    if (info.ndim != 2) {
      throw std::runtime_error("Constructed pybind array_t does not have 2 dimensions");
    }
    if ((info.shape[0] != shape[0]) || (info.shape[1] != shape[1])) {
      throw std::runtime_error("Constructed pybind array_t does not have requested shape");
    }
    if ((info.strides[0] != strides[0]) || (info.strides[1] != strides[1])) {
      throw std::runtime_error("Constructed pybind array_t does not have requested strides");
    }

    // make a deep copy
    for (int r=0; r<num_rows; ++r) {
      for (int c=0; c<num_cols; ++c) {
        *ptr++ = mat(r,c);
      }
    }
  }

  template <class pixeltype>
  using pixel_traits =face3d_img_util::pixel_traits<pixeltype>;

  template <class DLIB_T>
  void img_from_buffer(pybind11::array_t<typename pixel_traits<typename DLIB_T::type>::dtype>& py_img, DLIB_T &dlib_img)
  {
    using pixeltype = typename DLIB_T::type;
    using dtype = typename pixel_traits<pixeltype>::dtype;

    pybind11::buffer_info info = py_img.request();
    if (info.format != pybind11::format_descriptor<dtype>::format()) {
      throw std::runtime_error("Incompatible buffer type for conversion to dlib image");
    }
    int plane_stride = 0;
    if (pixel_traits<pixeltype>::nplanes == 1) {
      if (info.ndim != 2) {
        throw std::runtime_error("Expecting a buffer with shape RxC");
      }
    }
    else {
      if ((info.ndim != 2) && (info.ndim != 3)) {
        throw std::runtime_error("Expecting a buffer with shape RxC or RxCxP");
      }
      if (info.ndim == 3) {
        const size_t num_planes = info.shape[2];
        const size_t pix_size = sizeof(pixeltype)/sizeof(dtype);
        if (num_planes != pix_size) {
          throw std::runtime_error("Mismatch between size of pixel type and number of buffer planes");
        }
        plane_stride = info.strides[2]/sizeof(dtype);
      }
    }
    const size_t num_rows = info.shape[0];
    const size_t num_cols = info.shape[1];
    dlib_img.set_size(num_rows,num_cols);
    int row_stride = info.strides[0]/sizeof(dtype);
    int col_stride = info.strides[1]/sizeof(dtype);
    for (size_t r=0; r<num_rows; ++r) {
      for (size_t c=0; c<num_cols; ++c) {
        size_t pix_offset = r*row_stride + c*col_stride;
        face3d_img_util::assign_pixel(dlib_img, r, c, static_cast<dtype*>(info.ptr) + pix_offset, plane_stride);
      }
    }
  }

  template <class T>
  void img_from_buffer(pybind11::array_t<typename pixel_traits<T>::dtype>& py_img, vil_image_view<T> &img)
  {
    using dtype = typename pixel_traits<T>::dtype;

    pybind11::buffer_info info = py_img.request();
    if (info.format != pybind11::format_descriptor<dtype>::format()) {
      throw std::runtime_error("Incompatible buffer type for conversion to vil image");
    }
    int plane_stride = 0;
    size_t num_planes = 1;
    if (info.ndim ==  3) {
      num_planes = info.shape[2];
      plane_stride = info.strides[2]/sizeof(dtype);
    }
    else if (info.ndim <  2) {
      throw std::runtime_error("Expecting a buffer with shape RxC or RxCxP");
    }
    const size_t num_rows = info.shape[0];
    const size_t num_cols = info.shape[1];
    // need to reinitialize to guarantee interleaved planes
    img = vil_image_view<T>(num_cols, num_rows, 1, num_planes);
    int row_stride = info.strides[0]/sizeof(dtype);
    int col_stride = info.strides[1]/sizeof(dtype);
    for (size_t r=0; r<num_rows; ++r) {
      for (size_t c=0; c<num_cols; ++c) {
        size_t pix_offset = r*row_stride + c*col_stride;
        for (int p=0; p<num_planes; ++p) {
          img(c,r,p) = *(static_cast<dtype*>(info.ptr) + pix_offset + plane_stride*p);
        }
      }
    }
  }

  template<class DLIB_T>
  void img_to_buffer(DLIB_T const& dlib_img,
                     pybind11::array_t<typename pixel_traits<typename DLIB_T::type>::dtype> &py_img)
  {
    using pixeltype = typename DLIB_T::type;
    using dtype = typename pixel_traits<pixeltype>::dtype;

    const size_t num_rows = dlib_img.nr();
    const size_t num_cols = dlib_img.nc();

    const size_t num_planes = sizeof(pixeltype) / sizeof(dtype);
    std::vector<size_t> shape = { num_rows, num_cols, num_planes};
    std::vector<size_t> strides = { num_cols*num_planes*sizeof(dtype), num_planes*sizeof(dtype), sizeof(dtype) };

    if (num_planes == 1) {
      // no need for 3rd dimension
      shape.resize(2);
      strides.resize(2);
    }

    const dtype* data_ptr = reinterpret_cast<const dtype*>(face3d_img_util::get_data_ptr(dlib_img));

    py_img = pybind11::array_t<dtype>(shape, strides, data_ptr);
  }

  template<class T>
  void img_to_buffer(vil_image_view<T> const& img,
                     pybind11::array_t<typename pixel_traits<T>::dtype> &py_img)
  {
    using dtype = typename pixel_traits<T>::dtype;

    const size_t num_rows = img.nj();
    const size_t num_cols = img.ni();

    const size_t num_planes = img.nplanes();
    std::vector<size_t> shape = { num_rows, num_cols, num_planes};
    std::vector<size_t> strides = { img.jstep()*sizeof(dtype), img.istep()*sizeof(dtype), img.planestep()*sizeof(dtype) };

    if (num_planes == 1) {
      // no need for 3rd dimension
      shape.resize(2);
      strides.resize(2);
    }

    const dtype* data_ptr = reinterpret_cast<const dtype*>(img.top_left_ptr());

    py_img = pybind11::array_t<dtype>(shape, strides, data_ptr);
  }

}
