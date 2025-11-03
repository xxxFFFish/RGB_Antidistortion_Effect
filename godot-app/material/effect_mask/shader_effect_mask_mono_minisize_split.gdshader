shader_type canvas_item;

uniform ivec2 ScreenResolution = ivec2(1920, 1080);
uniform int MapShift = 420;
uniform sampler2D ScreenTexture : hint_screen_texture;

uniform sampler2D Map;

void fragment() {
	ivec2 raw_coordinate = ivec2(FRAGCOORD.xy);
	raw_coordinate.x *= 2;
	
	int side_flag = raw_coordinate.x / ScreenResolution.x;
	raw_coordinate.x = raw_coordinate.x - side_flag * ScreenResolution.x;
	
	ivec2 coef = (2 * raw_coordinate) / ScreenResolution;
	ivec2 map_coordinate = (ivec2(1) - 2 * coef) * raw_coordinate +
		coef * (ScreenResolution - ivec2(1));
	
	coef.x = min(1, (map_coordinate.y + MapShift + 1) / (map_coordinate.x + 1));
	coef.y = 1 - coef.x;
	
	map_coordinate = ivec2(
		coef.x * map_coordinate.x + coef.y * (map_coordinate.y + MapShift),
		coef.y * (map_coordinate.x - MapShift) + coef.x * map_coordinate.y
	);
	
	uvec4 data = uvec4(texelFetch(Map, map_coordinate, 0) * 255.0);
	float ratio = uintBitsToFloat(
		data.r |
		(data.g << 8u) |
		(data.b << 16u) |
		(data.a << 24u)
	);
	
	float f_side_flag = float(side_flag) * 0.5;
	vec2 shift = vec2(0.25 + f_side_flag, 0.5);
	vec2 _uv = (UV - shift) * ratio + shift;
	
	COLOR = any(lessThan(_uv, vec2(0.0 + f_side_flag, 0.0))) ||
		any(greaterThan(_uv, vec2(0.5 + f_side_flag, 1.0))) ?
		vec4(0.0, 0.0, 0.0, 1.0) :
		texture(ScreenTexture, _uv);
}
