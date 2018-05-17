#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 tex_coord;

uniform vec2 img_dims;
uniform vec2 depth_range;

out vec2 vert_uv;

void main()
{
  // generate normalized x,y,z coordinates in range 0,1
  vec2 pos2d = (position.xy) / img_dims;
  // normalize x,y,z coords to -1,1
  pos2d = pos2d * 2 - 1.0;
  float depth_scale = 1.0f/(depth_range.y - depth_range.x);
  float depth_offset = -depth_range.x;
  float depth = depth_scale*(position.z + depth_offset);

  gl_Position = vec4(pos2d.xy, depth, 1.0);
  vert_uv = vec2(tex_coord.x, tex_coord.y);
}
