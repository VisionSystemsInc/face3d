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

out VS_OUT {
  vec2 vert_uv;
  vec3 vert_normal;
  vec3 vert_pos;
} vs_out;

void main()
{
  vec3 pos3d = cam_rot * position + cam_trans;
  // generate normalized x,y,z coordinates in range 0,1
  vec2 pos2d = (cam_focal_len*pos3d.xy/pos3d.z + cam_principal_pt) / img_dims;
  float depth_norm = (pos3d.z - depth_range[0]) / (depth_range[1] - depth_range[0]);
  // normalize x,y coords to -1,1
  pos2d = pos2d * 2 - 1.0;
  depth_norm = depth_norm * 2 - 1.0;

  gl_Position = vec4(pos2d.xy, depth_norm, 1.0);
  vs_out.vert_uv = vec2(tex_coord.x, 1 - tex_coord.y);
  vs_out.vert_normal = normal;
  vs_out.vert_pos = position;
}
