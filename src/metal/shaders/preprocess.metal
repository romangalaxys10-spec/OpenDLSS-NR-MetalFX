// Input feature generation. Sixteen f32 lanes per padded pixel:
// three Box-Muller Gaussian lanes evaluated with the GPU's approximate f32
// transcendentals, constant 1, the centered proxy
// color twice (the second copy is the history when temporal input is valid),
// and the five conditioning lanes.
//
// Port of shaders/vulkan-glsl/preprocess.comp. The Gaussian hash constants and
// the transcendental instruction order are load-bearing: the GPU's approximate
// f32 transcendentals make the sequence order dependent, so sqrt / log2 / cos /
// sin stay in the exact GLSL positions (Metal's native f32 functions are the
// intended semantic). Every roundF16 stays at the same spot.
//
// Vulkan binding i -> [[buffer(i)]]; the push constant block -> constant Push&
// [[buffer(30)]] with identical field order/types (host encodes with setBytes).
//
// local_size (8, 8, 1); the host dispatches a 2D grid of 8x8 threadgroups and
// the kernel reads gid.xy.

#include "nr_common.metal"

struct Push {
  uint fullWidth;
  uint fullHeight;
  uint validWidth;
  uint validHeight;
  uint sourceWidth;
  uint sourceHeight;
  uint seed;
  float autoMask;        // > 0 enables the auto-mask lane pair
  float localTone;
  float localStructure;
  float skinStructure;
  float style;
};

static inline float hashUniform(uint value) {
  uint mixed = value;
  mixed = (mixed >> ((mixed >> 28u) + 4u)) ^ mixed;
  mixed *= 0x108ef2d9u;
  uint integer = ((mixed >> 30u) ^ (mixed >> 8u)) + 1u;
  return (float)integer * as_type<float>(0x33800000u);
}

// Scalar form of the network's Box-Muller sequence (instruction order preserved:
// the approximate transcendentals make it order dependent).
static inline float3 gaussian3(uint x, uint y, uint seed) {
  uint base = (x * 0x8da6b343u) ^ (seed * 0x9e3779b9u) ^ (y * 0xd8163841u) ^ 0x243f6a88u;
  base = (base >> ((base >> 28u) + 4u)) ^ base;
  base *= 0x108ef2d9u;
  base = (base >> 22u) ^ base;
  float u0 = hashUniform(base * 0x2c9277b5u + 0xac564b05u);
  float u1 = hashUniform(base * 0xfa6dc5f9u + 0x4712a88eu);
  float u2 = hashUniform(base * 0xcaa5b80du + 0x21dd796bu);
  float u3 = hashUniform(base * 0x83232c31u + 0x3463e0acu);
  float radius0 = sqrt(log2(u0) * as_type<float>(0x3f317218u) * -2.0f);
  float radius1 = sqrt(log2(u2) * as_type<float>(0x3f317218u) * -2.0f);
  float angle0 = u1 * as_type<float>(0x40c90fdbu);
  float angle1 = u3 * as_type<float>(0x40c90fdbu);
  return float3(roundF16(radius0 * cos(angle0)), roundF16(radius0 * sin(angle0)), roundF16(radius1 * cos(angle1)));
}

kernel void nr_preprocess(
    device const float4* proxy [[buffer(0)]],
    device float* features [[buffer(7)]],
    constant Push& pc [[buffer(30)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint2 id = gid.xy;
  if (id.x >= pc.fullWidth || id.y >= pc.fullHeight) return;
  // Coordinates outside the valid image are mirrored, while the noise channels still hash the original
  // padded coordinate.
  uint sourceX = id.x < pc.validWidth ? id.x : 2u * pc.validWidth - id.x - 2u;
  uint sourceY = id.y < pc.validHeight ? id.y : 2u * pc.validHeight - id.y - 2u;
  uint imageX = ((2u * sourceX + 1u) * pc.sourceWidth) / (2u * pc.validWidth);
  uint imageY = ((2u * sourceY + 1u) * pc.sourceHeight) / (2u * pc.validHeight);
  float4 rgba = proxy[imageY * pc.sourceWidth + imageX];
  // Native: texture sample -> cvt.rn.f16, sub.f16(0.5), mul.f16(0.125).
  float r = roundF16(roundF16(roundF16(rgba.r) - 0.5f) * 0.125f);
  float g = roundF16(roundF16(roundF16(rgba.g) - 0.5f) * 0.125f);
  float b = roundF16(roundF16(roundF16(rgba.b) - 0.5f) * 0.125f);
  float3 noise = gaussian3(id.x, id.y, pc.seed);
  uint base = (id.y * pc.fullWidth + id.x) * 16u;
  features[base + 0u] = noise.x;
  features[base + 1u] = noise.y;
  features[base + 2u] = noise.z;
  features[base + 3u] = 1.0f;
  features[base + 4u] = r;
  features[base + 5u] = g;
  features[base + 6u] = b;
  features[base + 7u] = r;
  features[base + 8u] = g;
  features[base + 9u] = b;
  features[base + 10u] = pc.style / 128.0f;
  features[base + 11u] = roundF16(pc.localTone);
  features[base + 12u] = roundF16(pc.autoMask > 0.0f ? 1.0f : pc.localStructure);
  features[base + 13u] = roundF16(pc.autoMask > 0.0f ? (pc.skinStructure < 0.0f ? pc.localStructure : pc.skinStructure) : -1.0f);
  features[base + 14u] = roundF16(pc.autoMask > 0.0f ? pc.localStructure : -1.0f);
  features[base + 15u] = 0.0f;
}
