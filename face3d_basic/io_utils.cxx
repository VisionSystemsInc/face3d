#include "io_utils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vil/vil_image_view.h>

using namespace face3d;

vnl_matrix<double> io_utils::read_matrix(std::string const& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    throw std::runtime_error("ERROR opening matrix file for read");
  }
  vnl_matrix<double> matrix;
  ifs >> matrix;
  return matrix;
}

void io_utils::write_matrix(vnl_matrix<double> const&m, std::string const& filename)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()){
    std::cerr << "ERROR opening " << filename << " for write" <<std::endl;
    return;
  }
  for (int r=0; r<m.rows(); ++r) {
    for (int c=0; c<m.cols(); ++c) {
      ofs << m[r][c] << " ";
    }
    ofs << std::endl;
  }
}

void io_utils::write_vector(vnl_vector<double> const&v, std::string const& filename)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()){
    std::cerr << "ERROR opening " << filename << " for write" <<std::endl;
    return;
  }
  for (int i=0; i<v.size(); ++i) {
    ofs << v[i] << " ";
  }
  ofs << std::endl;
}

void io_utils::write_strings(std::vector<std::string> const& strings, std::string const& filename)
{
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()){
    std::cerr << "ERROR opening " << filename << " for write" <<std::endl;
    return;
  }
  for (int i=0; i<strings.size(); ++i) {
    ofs << strings[i] << std::endl;
  }
}

vnl_vector<double> io_utils::read_vector(std::string const& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    throw std::runtime_error("ERROR opening vector file for read");
  }
  vnl_vector<double> vector;
  ifs >> vector;
  return vector;
}

std::vector<std::string> io_utils::read_strings(std::string const& filename)
{
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    throw std::runtime_error("ERROR opening strings file for read");
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) {
    lines.push_back(line);
  }
  return lines;
}

bool io_utils::read_strings(std::string const& filename, std::vector<std::vector<std::string> > &strings)
{
  strings.clear();
  std::ifstream ifs(filename.c_str());
  if (!ifs.good()) {
    std::cerr << "ERROR opening " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> line_strings;
    std::stringstream ss(line);
    std::string s;
    while(ss >> s) {
      line_strings.push_back(s);
    }
    strings.push_back(line_strings);
  }
  return true;
}

bool io_utils::read_string(std::string const& filename, std::string &str)
{
  std::ifstream ifs(filename);
  if(!ifs.good()) {
    std::cerr << "ERROR opening " << filename << std::endl;
    return false;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  str = ss.str();
  return true;
}

bool io_utils::write_camera(const vpgl_proj_camera<double> *cam, std::string const& filename)
{
  if (const vpgl_perspective_camera<double>* pcam = dynamic_cast<const vpgl_perspective_camera<double>*>(cam)) {
    return write_t(*pcam, filename);
  }
  if (const vpgl_affine_camera<double>* acam = dynamic_cast<const vpgl_affine_camera<double>*>(cam)) {
    return write_t(*acam, filename);
  }
  return write_t(*cam, filename);
}


vpgl_perspective_camera<double> io_utils::read_camera(std::string const& filename)
{
  vpgl_perspective_camera<double> cam;
  if (!read_t(filename, cam)) {
    throw std::runtime_error("error reading camera file");
  }
  return cam;
}

vpgl_affine_camera<double> io_utils::read_affine_camera(std::string const& filename)
{
  vpgl_affine_camera<double> cam;
  if (!read_t(filename, cam)) {
    throw std::runtime_error("error reading camera file");
  }
  return cam;
}


bool io_utils::write_ply( std::vector<vgl_point_3d<double> > const& points,
                         std::vector<vil_rgb<unsigned char> > const& colors,
                         std::string const& filename)
{
  // open file for write
  std::ofstream ofs(filename.c_str());
  if (!ofs.good()) {
    std::cerr << "ERROR opening file " << filename << " for write." << std::endl;
    return false;
  }

  // write PLY header
  ofs << "ply" << std::endl << "format ascii 1.0" << std::endl;
  ofs << "element vertex " << points.size() << std::endl;
  ofs << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
  ofs << "property uchar red" << std::endl << "property uchar green" << std::endl << "property uchar blue" << std::endl;
  ofs << "end_header" << std::endl;

  // write points
  for (int i=0; i<points.size(); ++i) {
    ofs << points[i].x() << " " << points[i].y() << " " << points[i].z() << " ";
    ofs << (int)colors[i].R() << " " << (int)colors[i].G() << " " << (int)colors[i].B() << std::endl;
  }
  return true;
}

vnl_matrix<double> io_utils::read_numpy_matrix(std::string const&filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "Error opening file " << filename << " for read." << std::endl;
    throw std::runtime_error("Error opening file");
  }
  // read and verify magic string to verify file type
  const char magic[] = "\x93NUMPY";
  char magic_in_buff[sizeof(magic)];
  ifs.read(magic_in_buff, sizeof(magic)-1); // -1 because no null terminator in file
  if (std::strncmp(magic, magic_in_buff, sizeof(magic)-1)) {
    std::cerr << "Read incorrect magic string from matrix file" << std::endl;
    throw std::runtime_error("Invalid file type");
  }
  // file format version
  unsigned char version_major, version_minor;
  ifs.read(reinterpret_cast<char*>(&version_major), sizeof(version_major));
  ifs.read(reinterpret_cast<char*>(&version_minor), sizeof(version_minor));
  if ((version_major != 1) || (version_minor != 0)) {
    std::cerr << "ERROR: unsupported numpy file format version " << static_cast<int>(version_major)  << "." << static_cast<int>(version_minor) << std::endl;
    // add more versions to this list as they are supported.
    std::cerr << "Supported versions = [1.0,]" << std::endl;
    throw std::runtime_error("Unsupported numpy file format version");
  }
  // header
  char headerlen_bytes[2];
  ifs.read(headerlen_bytes, sizeof(headerlen_bytes));
  size_t headerlen = 0;
  // headerlen value is in little-endian format
  headerlen |= static_cast<size_t>(headerlen_bytes[0]);
  headerlen |= (static_cast<size_t>(headerlen_bytes[1]) << 8);

  std::vector<char> header_buff(headerlen,0);
  ifs.read(&(header_buff[0]), headerlen);
  std::string header(header_buff.begin(), header_buff.end());

  //std::cout << "header :" << header << std::endl;

  // test native endianess
  union { uint32_t i; char c[4]; } endianess_test = {0x01020304};
  bool big_endian = endianess_test.c[0] == 1;

  // get data type from header
  const char descr_key_str[] {"'descr':"};
  size_t descr_key_idx = header.find(descr_key_str);
  size_t descr_val_idx_begin = header.find("'", descr_key_idx + sizeof(descr_key_str)) + 1;
  size_t descr_val_idx_end = header.find("'", descr_val_idx_begin);
  if (descr_val_idx_end - descr_val_idx_begin < 3) {
    std::cerr << "Error: expecting format description string of at least length 3, got " << descr_val_idx_end << " - " << descr_val_idx_begin << std::endl;
    throw std::runtime_error("Unsupported format description");
  }
  std::string descr_val(header.begin() + descr_val_idx_begin, header.begin() + descr_val_idx_end);
  //std::cout << "descr_val = " << descr_val << std::endl;
  bool endianess_mismatch = false;
  if (descr_val[0] == '<') {
    // little endian data
    if (big_endian) {
      endianess_mismatch = true;
    }
  }
  else if (descr_val[0] == '>') {
    // big endian data
    if (!big_endian) {
      endianess_mismatch = true;
    }
  }
  else {
    std::cerr << "Expecting first character of format description to be '<' or '>'" << std::endl;
    std::cerr <<    "format description = " << descr_val << std::endl;
    throw std::runtime_error("Unsupported format description");
  }
  char value_type = descr_val[1];
  if (value_type != 'f') {
    std::cerr << "Error: Only float matrices are supported: got value type char " << value_type << std::endl;
    throw std::runtime_error("Unsupported data type");
  }
  std::stringstream value_size_ss(std::string(descr_val.begin() + 2, descr_val.end()));
  size_t value_size = 0;
  value_size_ss >> value_size;
  //std::cout << "value_size = " << value_size << std::endl;
  if ((value_size != sizeof(float)) && (value_size != sizeof(double))) {
    std::cerr << "Error: only floats of size " << sizeof(float) << " or " << sizeof(double) << " are supported. got " << value_size << std::endl;
    throw std::runtime_error("unsupported float size");
  }

  // get order description from header
  bool fortran_order = false;
  const char order_key_str[] {"'fortran_order':"};
  size_t order_key_idx = header.find(order_key_str);
  size_t order_val_idx_begin = order_key_idx + sizeof(order_key_str);
  size_t order_val_idx_end = header.find(",", order_val_idx_begin);
  std::stringstream order_val_ss(std::string(header.begin()+order_val_idx_begin, header.begin()+order_val_idx_end));
  std::string order_val;
  order_val_ss >> order_val;
  if (order_val == "True") {
    fortran_order = true;
  }
  else if (order_val == "False") {
    fortran_order = false;
  }
  else {
    std::cerr << "Error: Got unexpected string for 'fortran_order' key." << std::endl;
    std::cerr << "  Expecting 'True' or 'False', got " << order_val << std::endl;
    throw std::runtime_error("Invalid value for key fortran_order");
  }

  // get matrix shape from header
  const char shape_key_str[] {"'shape':"};
  size_t shape_key_idx = header.find(shape_key_str);
  size_t shape_val_idx_begin = header.find("(", shape_key_idx + sizeof(shape_key_str)) + 1;
  size_t shape_val_idx_end = header.find(")", shape_val_idx_begin);
  std::string shape_val(header.begin() + shape_val_idx_begin, header.begin() + shape_val_idx_end);
  //std::cout << "shape_val = " << shape_val << std::endl;
  std::stringstream shape_ss(std::string(shape_val.begin(), shape_val.end()));
  std::vector<int> matrix_dims;
  std::string dim_str;
  while(std::getline(shape_ss, dim_str, ',')) {
    std::stringstream dim_ss(dim_str);
    int dim=0;
    dim_ss >> dim;
    matrix_dims.push_back(dim);
  }
  if (matrix_dims.size() != 2) {
    std::cerr << "Expecting 2 matrix dimensions, but got " << matrix_dims.size() << std::endl;
    std::cerr << "  Matrix dims = ";
    for (auto d : matrix_dims) {
      std::cerr << d << ", ";
    }
    std::cerr << std::endl;
    throw std::runtime_error("Invalid matrix dimensions");
  }
  //std::cout << "shape = " << matrix_dims[0] << " x " << matrix_dims[1] << std::endl;

  vnl_matrix<double> matrix(matrix_dims[0], matrix_dims[1]);
  std::vector<char> value_bytes(value_size*matrix_dims[0]*matrix_dims[1]);
  // read data
  ifs.read(&value_bytes[0], value_bytes.size());
  size_t value_bytes_offset = 0;
  if (fortran_order) {
    for (int c=0; c<matrix_dims[1]; ++c) {
      for (int r=0; r<matrix_dims[0]; ++r) {
        if (endianess_mismatch) {
          // swap byte order
          std::reverse(value_bytes.begin()+value_bytes_offset, value_bytes.begin()+value_bytes_offset+value_size);
        }
        if (value_size == sizeof(float)) {
          matrix(r,c) = *reinterpret_cast<float*>(&value_bytes[value_bytes_offset]);
        }
        else if (value_size == sizeof(double)) {
          matrix(r,c) = *reinterpret_cast<double*>(&value_bytes[value_bytes_offset]);
        }
        else {
          throw std::runtime_error("Unhandled data size");
        }
        value_bytes_offset += value_size;
      }
    }
  }
  else {
    for (int r=0; r<matrix_dims[0]; ++r) {
      for (int c=0; c<matrix_dims[1]; ++c) {
        if (endianess_mismatch) {
          // swap byte order
          std::reverse(value_bytes.begin()+value_bytes_offset, value_bytes.begin()+value_bytes_offset+value_size);
        }
        if (value_size == sizeof(float)) {
          matrix(r,c) = *reinterpret_cast<float*>(&value_bytes[value_bytes_offset]);
        }
        else if (value_size == sizeof(double)) {
          matrix(r,c) = *reinterpret_cast<double*>(&value_bytes[value_bytes_offset]);
        }
        else {
          throw std::runtime_error("Unhandled data size");
        }
        value_bytes_offset += value_size;
      }
    }
  }
  return matrix;
}
