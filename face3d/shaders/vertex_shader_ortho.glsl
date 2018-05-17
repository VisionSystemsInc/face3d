#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

uniform float cam_scale;
uniform vec3 cam_trans;
uniform mat3 cam_rot;
uniform vec3 img_dims;

out VS_OUT {
  vec2 vert_uv;
  vec3 vert_normal;
  vec3 vert_pos;
} vs_out;

void main()
{
  vec3 pos3d = cam_rot * position;
  // scale z coordinate to account for cam_scale in depth clipping
  pos3d.z /= cam_scale;
  // generate normalized x,y,z coordinates in range 0,1
  vec3 pos2d = (cam_scale*pos3d + cam_trans) / img_dims;
  // normalize x,y,z coords to -1,1
  pos2d = pos2d * 2 - 1.0;

  gl_Position = vec4(pos2d.xyz, 1.0);
  vs_out.vert_uv = vec2(tex_coord.x, 1 - tex_coord.y);
  vs_out.vert_normal = normal;
  vs_out.vert_pos = position;
}
