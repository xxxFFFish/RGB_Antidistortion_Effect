shader_type canvas_item;

uniform vec2 CenterOffset = vec2(0.0);
uniform vec2 UVScale = vec2(1.0);
uniform float RadiusDensity = 128.0;
uniform float AngularDensity = 2.0;
uniform float AngularDensityGrowth = 0.1;

uniform float MinAngularVelocityRatio : hint_range(0, 1.0) = 0.5;
uniform float MaxAngularVelocity = 1.0;

uniform float MinTrailRangeRatio : hint_range(0, 1.0) = 0.5;
uniform float MaxTrailRange = 0.2;
uniform float HeadRange = 0.01;
uniform float EdgeRange : hint_range(0, 0.5) = 0.0;

uniform float AntiAliasRange : hint_range(0, 0.5) = 0.2;

uniform float MinShowHideFrequenceRatio : hint_range(0, 1.0) = 0.5;
uniform float MaxShowHideFrequence = 1.0;
uniform float ShowDuty : hint_range(0, 1.0) = 0.6;
uniform float HideDuty : hint_range(0, 1.0) = 0.2;

uniform float CenterMask = 10.0;

uniform vec4 PointColor : source_color = vec4(1.0);

vec2 hash12(float p)
{
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

void fragment() {
	vec2 _uv = (UV - CenterOffset - vec2(0.5)) * UVScale;
	vec2 polar = vec2(length(_uv) * RadiusDensity, (atan(_uv.y, _uv.x) / PI + 1.0) * 0.5);

	float i_rho = ceil(polar.x);
	vec2 noise = hash12(i_rho);

	float theta = fract(
		polar.y + MaxAngularVelocity * (
			MinAngularVelocityRatio + (1.0 - MinAngularVelocityRatio) * noise.x
		) * TIME
	);

	float theta_time = floor(AngularDensity + i_rho * AngularDensityGrowth);
	theta = fract(theta * theta_time) / theta_time;

	float val = smoothstep(1.0, HeadRange / i_rho,
		MaxTrailRange * theta * (
			MinTrailRangeRatio + (1.0 - MinTrailRangeRatio) * (1.0 - noise.x)
		) * i_rho
	);

	float f_rho = fract(polar.x);
	val *= smoothstep(0.5 - EdgeRange, 0.5 - AntiAliasRange, abs(f_rho - 0.5));

	theta = fract(
		MaxShowHideFrequence * (
			MinShowHideFrequenceRatio + (1.0 - MinShowHideFrequenceRatio) * noise.y
		) * TIME
	);

	val *= smoothstep((1.0 - HideDuty) * 0.5, ShowDuty * 0.5, abs(theta - 0.5));

	val *= step(CenterMask, i_rho);

	COLOR = vec4(mix(vec4(0.0), PointColor, val));
}
