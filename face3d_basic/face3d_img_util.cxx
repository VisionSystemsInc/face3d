#include <face3d_basic/face3d_img_util.h>
namespace face3d_img_util{
  void assign_pixel(dlib::rgb_pixel& pixel, unsigned char* pix_ptr, size_t plane_stride)
  {
    unsigned char red =   *(pix_ptr + 0*plane_stride);
    unsigned char green = *(pix_ptr + 1*plane_stride);
    unsigned char blue =  *(pix_ptr + 2*plane_stride);
    dlib::assign_pixel(pixel, dlib::rgb_pixel(red,green,blue));
  }

  void assign_pixel(dlib::rgb_alpha_pixel& pixel, unsigned char* pix_ptr, size_t plane_stride)
  {
    unsigned char red =   *(pix_ptr + 0*plane_stride);
    unsigned char green = *(pix_ptr + 1*plane_stride);
    unsigned char blue =  *(pix_ptr + 2*plane_stride);
    unsigned char alpha = *(pix_ptr + 3*plane_stride);
    dlib::assign_pixel(pixel, dlib::rgb_alpha_pixel(red,green,blue,alpha));
  }

  void assign_pixel(vgl_point_3d<float>& pixel, float* pix_ptr, size_t plane_stride)
  {
    float x = *(pix_ptr + 0*plane_stride);
    float y = *(pix_ptr + 1*plane_stride);
    float z = *(pix_ptr + 2*plane_stride);
    pixel = vgl_point_3d<float>(x,y,z);
  }

  void assign_pixel(vgl_vector_3d<float>& pixel, float* pix_ptr, size_t plane_stride)
  {
    float x = *(pix_ptr + 0*plane_stride);
    float y = *(pix_ptr + 1*plane_stride);
    float z = *(pix_ptr + 2*plane_stride);
    pixel = vgl_vector_3d<float>(x,y,z);
  }


  template <>
  void assign_pixel<vil_image_view<float>, unsigned char>(vil_image_view<float> & img, int r, int c, unsigned char* data_p, size_t plane_stride){
    if (img.nplanes() != 3 || img.nplanes()!= 4)
      throw std::range_error("[face3d_img_util]::assign_pixel: VIL output image needs to have 3 or 4 planes");
    for (unsigned i=0; i < img.nplanes(); i++)
      img(c, r, i) = *(data_p + i * plane_stride);
  }

  template<>
  void set_color_img_size<vil_image_view<float> >(vil_image_view<float>& img, unsigned nr, unsigned nc){
    img.set_size(nc, nr, 3);
  }



}
