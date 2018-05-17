#include <face3d_basic/io_utils.h>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename_to_read>" << std::endl;
    return -1;
  }
  std::string fname_in(argv[1]);

  vnl_matrix<double> m = face3d::io_utils::read_numpy_matrix(fname_in);

  std::cout << "read matrix: " << std::endl;
  std::cout << "size = " << m.rows() << " x " << m.cols() << std::endl;
  std::cout << "m[0,0] = " << m(0,0) << std::endl;
  std::cout << "m[0,-1] = " << m(0,m.cols()-1) << std::endl;
  std::cout << "m[1,0] = " << m(1,0) << std::endl;
  std::cout << "m[1,-1] = " << m(1,m.cols()-1) << std::endl;
  //std::cout << m << std::endl;

  return 0;
}


