#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 tex_coord;

uniform float cam_scale;
uniform vec3 cam_trans;
uniform mat3 cam_rot;
uniform vec3 img_dims;

out float z;
out vec2 img_uv;
out vec3 vert_normal;

void main()
{
  vec3 pos3d = cam_rot * position;
  // scale z coordinate to account for cam_scale in depth clipping
  pos3d.z /= cam_scale;
  // generate normalized x,y,z coordinates in range 0,1
  vec3 pos2d_norm = (cam_scale*pos3d + cam_trans) / img_dims;
  img_uv = pos2d_norm.xy;

  // texture coordinates will be used as output image location
  // use z value from pos2d_norm to get proper depth testing
  vec3 pos2d = vec3(tex_coord.xy, pos2d_norm.z);
  // convert texture coordinates to -1,1 since they will be used as output image coords
  pos2d = pos2d * 2 - 1.0;
  // flip y coordinate for right-side-up face texture
  pos2d.y = -pos2d.y;

  gl_Position = vec4(pos2d.xyz, 1.0);
  z = pos2d_norm.z;
  vert_normal = normal;
}
