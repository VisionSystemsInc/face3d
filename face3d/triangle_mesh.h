#ifndef face3d_triangle_mesh_h_included_
#define face3d_triangle_mesh_h_included_

#include <vector>
#include <Eigen/Dense>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>
#include <vgl/vgl_point_3d.h>
#include <igl/per_vertex_normals.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_point_2d.h>

namespace face3d
{
  typedef Eigen::Matrix<int , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXiRowMajor;
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRowMajor;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;

  class triangle_mesh
  {
    public:
      ///*
      // vertex coordinates
      using VTYPE = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
      // face indices
      using FTYPE = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
      // texture coordinates
      using TTYPE = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
      // texture coordinates (alias for TTYPE)
      using UVTYPE = TTYPE;
      // normals
      using NTYPE = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
      //*/

      triangle_mesh(){}
      triangle_mesh(VTYPE const& V, FTYPE const& F);
      triangle_mesh(VTYPE const& V, FTYPE const& F, NTYPE const& N);
      triangle_mesh(VTYPE const& V, FTYPE const &F, NTYPE const& N, TTYPE const& T);

      // explicit to prevent argument conversion from string to mesh
      explicit triangle_mesh(std::string const& ply_filename);

      void set_vertices(std::vector<vgl_point_3d<double> > const& verts);
      void set_texture_coords(std::vector<vgl_point_2d<double> > const& uvs);

      template <class IT_TYPE>
        void set_vertices(IT_TYPE verts_begin, IT_TYPE verts_end);
      template <class IT_TYPE>
        void set_texture_coords(IT_TYPE uv_begin, IT_TYPE uv_end);

      template<class PT_T>
        void get_vertices(std::vector<vgl_point_3d<PT_T> > &verts) const;

      VTYPE const& V() const {return V_;}
      FTYPE const& F() const {return F_;}
      TTYPE const& T() const {return T_;}
      UVTYPE const& UV() const {return T();} // alias for T()
      NTYPE const& N() const {return N_;}

      // get a single vertex
      vgl_point_3d<double> vertex(int i) const { return vgl_point_3d<double>(V_(i,0),V_(i,1),V_(i,2)); }
      // get a single texture coordinate
      vgl_point_2d<double> vertex_tex(int i) const { return vgl_point_2d<double>(T_(i,0), T_(i,1)); }
      // get a single vertex normal
      vgl_point_3d<double> vertex_normal(int i) const { return vgl_point_3d<double>(N_(i,0), N_(i,1), N_(i,2)); }
      // set a single texture coordinate
      void set_texture_coord(int i, vgl_point_2d<double> const& uv) { T_(i,0) = uv.x(); T_(i,1) = uv.y();}
      // get the vertex index for a face
      std::array<int,3> face(int i) const { return std::array<int,3>{ {F_(i,0), F_(i,1), F_(i,2)} }; }

      int num_faces() const { return F_.rows(); }
      int num_vertices() const { return V_.rows(); }

      void save_obj(std::string const& mesh_filename) const;
      void save_ply(std::string const& mesh_filename) const;

    private:
      VTYPE V_;
      FTYPE F_;
      NTYPE N_;
      TTYPE T_;

      void compute_normals();
      void normalize_normals();
  };

}

template <class IT_TYPE>
 void face3d::triangle_mesh::set_vertices(IT_TYPE verts_begin, IT_TYPE verts_end)
{
  const int num_verts = V_.rows();
  if (std::distance(verts_begin, verts_end) != num_verts) {
    throw std::logic_error("ERROR: changing number of vertices");
  }
  int v = 0;
  for (auto it = verts_begin; it != verts_end; ++it, ++v) {
    V_(v,0) = it->x();
    V_(v,1) = it->y();
    V_(v,2) = it->z();
  }
  compute_normals();
}

template <class IT_TYPE>
 void face3d::triangle_mesh::set_texture_coords(IT_TYPE uv_begin, IT_TYPE uv_end)
{
  const int num_verts = T_.rows();
  if (std::distance(uv_begin, uv_end) != num_verts) {
    throw std::logic_error("ERROR: changing number of vertices");
  }
  int v = 0;
  for (auto it = uv_begin; it != uv_end; ++it, ++v) {
    T_(v,0) = it->x();
    T_(v,1) = it->y();
  }
}

template <class PT_T>
void face3d::triangle_mesh::get_vertices(std::vector<vgl_point_3d<PT_T> > &verts) const
{
  verts.clear();
  const int nverts = V_.rows();
  for (int i=0; i<nverts; ++i) {
    verts.push_back(vgl_point_3d<PT_T>(static_cast<PT_T>(V_(i,0)),
                                       static_cast<PT_T>(V_(i,1)),
                                       static_cast<PT_T>(V_(i,2))));
  }
}

#endif
