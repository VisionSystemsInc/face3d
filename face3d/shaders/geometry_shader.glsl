#version 330 core
#extension GL_EXT_geometry_shader : enable

layout (triangles) in;
layout (triangle_strip, max_vertices=3) out;

in VS_OUT {
  vec2 vert_uv;
  vec3 vert_normal;
  vec3 vert_pos;
} gs_in[];

out vec2 vert_uv;
out vec3 vert_normal;
out vec3 vert_pos;
out vec3 vert_bc;

uniform int face_idx_offset;

void main()
{
  gl_PrimitiveID = gl_PrimitiveIDIn + face_idx_offset;
  gl_Position = gl_in[0].gl_Position;
  vert_uv = gs_in[0].vert_uv;
  vert_normal = gs_in[0].vert_normal;
  vert_pos = gs_in[0].vert_pos;
  vert_bc = vec3(1,0,0);
  EmitVertex();

  gl_PrimitiveID = gl_PrimitiveIDIn + face_idx_offset;
  gl_Position = gl_in[1].gl_Position;
  vert_uv = gs_in[1].vert_uv;
  vert_normal = gs_in[1].vert_normal;
  vert_pos = gs_in[1].vert_pos;
  vert_bc = vec3(0,1,0);
  EmitVertex();

  gl_PrimitiveID = gl_PrimitiveIDIn + face_idx_offset;
  gl_Position = gl_in[2].gl_Position;
  vert_uv = gs_in[2].vert_uv;
  vert_normal = gs_in[2].vert_normal;
  vert_pos = gs_in[2].vert_pos;
  vert_bc = vec3(0,0,1);
  EmitVertex();

  EndPrimitive();
}
