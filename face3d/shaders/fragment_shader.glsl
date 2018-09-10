#version 330 core

layout (location=0) out vec4 color;
layout (location=1) out vec3 pos;
layout (location=2) out vec3 norm;
layout (location=3) out int face_idx;
layout (location=4) out vec3 face_bc;
layout (location=5) out vec3 uv;

in vec2 vert_uv;
in vec3 vert_normal;
in vec3 vert_pos;
in vec3 vert_bc;

uniform vec3 light_dir;
uniform float light_ambient_weight;
uniform sampler2D tex;

void main()
{
  color = texture(tex, vert_uv);
  // modulate for lighting
  vec3 vert_normal_unit = normalize(vert_normal);
  vec3 light_dir_unit = normalize(light_dir);
  float dp = dot(light_dir_unit, vert_normal_unit);
  float mod_factor = light_ambient_weight + (1.0-light_ambient_weight)*max(dp,0.0f);

  color.rgb *= mod_factor;
  pos = vert_pos;
  norm = vert_normal_unit;
  face_idx = gl_PrimitiveID;
  face_bc = vert_bc;
  uv = vec3(vert_uv.xy, 1.0f);
}
