#include "coeffs_to_pixmap.h"
#include "textured_triangle_mesh.h"

#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>

namespace face3d {

dlib::array2d<vgl_point_3d<float> >
coeffs_to_pixmap(head_mesh const& mesh,
                 vnl_vector<double> const& subject_coeffs, vnl_vector<double> const &expression_coeffs,
                 vnl_matrix<double> const& subject_components, vnl_matrix<double> const& expression_components)
{
  head_mesh mesh_warped(mesh);
  mesh_warped.apply_coefficients(subject_components, expression_components,
                                 subject_coeffs, expression_coeffs);

  const textured_triangle_mesh<head_mesh::TEX_T> face_mesh = mesh_warped.face_mesh();
  const int tex_nx = face_mesh.texture().nc();
  const int tex_ny = face_mesh.texture().nr();

  triangle_mesh::TTYPE const& UV = face_mesh.T();
  triangle_mesh::VTYPE const& V = face_mesh.V();
  triangle_mesh::FTYPE const& F = face_mesh.F();

  igl::AABB<triangle_mesh::TTYPE,2> tex_mesh_tree;
  tex_mesh_tree.init(UV, F);

  dlib::array2d<vgl_point_3d<float> > pixmap(tex_ny, tex_nx);

  // for each pixel in the texture map
  for (int y=0; y<tex_ny; ++y) {
    for (int x=0; x<tex_nx; ++x) {
      // unmapped pixels will be written as 0
      pixmap[y][x] = vgl_point_3d<float>(0,0,0);

      // convert image coordinates to normalized texture coords
      double u = static_cast<double>(x) / tex_nx;
      double v = 1.0 - (static_cast<double>(y) / tex_ny);

      // find the face and barycentric coordinates using texture coordinates
      Eigen::MatrixXd query_points2d(1,2);
      query_points2d(0,0) = u;
      query_points2d(0,1) = v;
      Eigen::MatrixXd query_dists2d;
      Eigen::MatrixXi face_indices2d;
      Eigen::MatrixXd closest_points2d;

      tex_mesh_tree.squared_distance(UV, F, query_points2d, query_dists2d, face_indices2d, closest_points2d);
      // 2d mesh, so we can expect query points to lie exactly on triangles
      if (query_dists2d(0) <= 1e-6) {
        // get the points
        Eigen::MatrixXd barycentric_coords;
        Eigen::MatrixXi vert_indices = F.row(face_indices2d(0));

        // compute the barycentric coordinates of the query point wrt the mesh face
        igl::barycentric_coordinates(query_points2d,
                                     Eigen::MatrixXd(UV.row(vert_indices(0))),
                                     Eigen::MatrixXd(UV.row(vert_indices(1))),
                                     Eigen::MatrixXd(UV.row(vert_indices(2))),
                                     barycentric_coords);

        // get corresponding 3-d point
        Eigen::MatrixXd pt3d =
          barycentric_coords(0)*V.row(vert_indices(0)) +
          barycentric_coords(1)*V.row(vert_indices(1)) +
          barycentric_coords(2)*V.row(vert_indices(2));

        pixmap[y][x] = vgl_point_3d<float>(pt3d(0), pt3d(1), pt3d(2));
      }
    }
  }
  return pixmap;
}

}
