shader_type canvas_item;

uniform vec3 ForeColor : source_color = vec3(1.0);
uniform vec3 BackColor : source_color = vec3(0.0);
uniform float VerticalBlock = 2;
uniform float HorizontallBlock = 2;

void fragment() {
	vec2 sub_uv = fract((UV * vec2(HorizontallBlock, VerticalBlock)) * 0.5);

	vec3 _color = ((sub_uv.x < 0.5) == (sub_uv.y < 0.5)) ?
			ForeColor :
			BackColor;

	COLOR = vec4(_color, 1.0);
}
