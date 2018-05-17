#version 330 core
#extension GL_EXT_shadow_samplers : enable

layout (location=0) out vec4 color;

in float z;
in vec2 img_uv;
in vec3 vert_normal;

uniform vec3 camera_dir;

uniform sampler2D img;
uniform sampler2DShadow img_depth;

void main()
{
  vec3 vert_normal_unit = normalize(vert_normal);
  color = vec4(0,0,0,0);
  float offset = 1e-2;
  vec3 img_uv3 = vec3(img_uv,z-offset);
  float vis = texture(img_depth, img_uv3);
  if (vis > 0) {
    color = texture(img, img_uv);
    float alpha = dot(camera_dir, vert_normal_unit);
    if (alpha < 0) {
      alpha = 0;
    }
    color.w = vis*alpha;
  }
}
