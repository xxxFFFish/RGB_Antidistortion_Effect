shader_type canvas_item;

#define GREATER_WIDTH true

uniform ivec2 ScreenResolution = ivec2(1920, 1080);
uniform int MapShift = 420;
uniform sampler2D ScreenTexture : hint_screen_texture;

uniform sampler2D Map;

void fragment() {
	ivec2 raw_coordinate = ivec2(FRAGCOORD.xy);
	
	ivec2 coef = (2 * raw_coordinate) / ScreenResolution;
	ivec2 map_coordinate = (ivec2(1) - 2 * coef) * raw_coordinate +
		coef * (ScreenResolution - ivec2(1));
	
	#if GREATER_WIDTH
	coef.x = min(1, (map_coordinate.y + MapShift + 1) / (map_coordinate.x + 1));
	#else
	coef.x = min(1, (map_coordinate.x - MapShift + 1) / (map_coordinate.y + 1));
	#endif
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
	
	vec2 _uv = (UV - vec2(0.5)) * ratio + vec2(0.5);
	
	COLOR = any(lessThan(_uv, vec2(0.0))) || any(greaterThan(_uv, vec2(1.0))) ?
		vec4(0.0, 0.0, 0.0, 1.0) :
		texture(ScreenTexture, _uv);
}
