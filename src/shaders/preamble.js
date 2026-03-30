export const glslDirFromUV = [
  "vec3 dirFromUV(vec2 uv) {",
  "  float phi = uv.x * 6.28318530718;",
  "  float theta = (1.0 - uv.y) * 3.14159265359;",
  "  float sinT = sin(theta);",
  "  return vec3(-sinT * cos(phi), cos(theta), sinT * sin(phi));",
  "}"
].join("\n");

export const glslPickColor = [
  "  vec3 color = classColors[0];",
  "  if (bestCI == 1) color = classColors[1];",
  "  else if (bestCI == 2) color = classColors[2];",
  "  else if (bestCI == 3) color = classColors[3];",
  "  else if (bestCI == 4) color = classColors[4];"
].join("\n");
