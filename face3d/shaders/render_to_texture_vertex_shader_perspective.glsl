#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

uniform float cam_focal_len;
uniform vec2 cam_principal_pt;
uniform mat3 cam_rot;
uniform vec3 cam_trans;
uniform vec2 img_dims;
uniform vec2 depth_range;

out float z;
out vec2 img_uv;
out vec3 vert_normal;

void main()
{
  vec3 pos3d = cam_rot * position + cam_trans;
  // generate normalized x,y,z coordinates in range 0,1
  vec2 pos2d_norm = (cam_focal_len*pos3d.xy/pos3d.z + cam_principal_pt) / img_dims;
  img_uv = pos2d_norm.xy;

  // texture coordinates will be used as output image location
  // use normalized z value to get proper depth testing
  float depth_norm = (pos3d.z - depth_range[0]) / (depth_range[1] - depth_range[0]);
  vec3 pos2d = vec3(tex_coord.xy, depth_norm);
  // convert texture coordinates to -1,1 since they will be used as output image coords
  pos2d = pos2d * 2 - 1.0;
  // flip y coordinate for right-side-up face texture
  pos2d.y = -pos2d.y;

  gl_Position = vec4(pos2d.xyz, 1.0);
  z = depth_norm;
  vert_normal = normal;
}
