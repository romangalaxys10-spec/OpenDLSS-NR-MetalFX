// Global ViT cosine-attention Q/K normalization.
// One thread per (token, head). Differs from the fused-Swin variant: the pair
// squares are summed in f32 then rounded, and Q carries the sqrt(head_dim)
// and learned per-head scales as three separate half multiplies.
//
// Port of shaders/vulkan-glsl/global_normalize.comp. Per-head f32 scales are
// read from binding 4 as raw 32-bit words (uintBitsToFloat -> as_type<float>).
//
// Vulkan binding i -> [[buffer(i)]]; the push constant block -> constant Push&
// [[buffer(30)]] with identical field order/types (host encodes with setBytes).
//
// local_size_x = 256; the host dispatches a 1D grid, so the GLSL
// gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * 256u linear-id
// recovery is simply gid.x (spec section 3).

#include "nr_common.metal"

struct Push {
  uint tokens;
  uint heads;
  uint channels;        // heads * 32
  uint scaleWordOffset; // per-head f32 learned scale, in words within binding 4
};

static inline half treeSum16(thread half* r) {
  // reduction[c] += reduction[c + stride] for stride 8, 4, 2, 1.
  half s8[8];
  for (uint c = 0u; c < 8u; ++c) s8[c] = r[c] + r[c + 8u];
  half s4[4];
  for (uint c = 0u; c < 4u; ++c) s4[c] = s8[c] + s8[c + 4u];
  half s2[2];
  s2[0] = s4[0] + s4[2];
  s2[1] = s4[1] + s4[3];
  return s2[0] + s2[1];
}

kernel void nr_global_normalize(
    device const half* qkv [[buffer(1)]],
    device const uint* aux32 [[buffer(4)]],
    device uchar* normalized [[buffer(5)]],
    constant Push& pc [[buffer(30)]],
    uint3 gid [[thread_position_in_grid]]) {
  uint id = gid.x;
  if (id >= pc.tokens * pc.heads) return;
  uint token = id / pc.heads;
  uint head = id % pc.heads;
  uint base = token * pc.channels * 3u + head * 96u;

  half q[32];
  half k[32];
  for (uint c = 0u; c < 32u; ++c) {
    q[c] = qkv[base + c];
    k[c] = qkv[base + 32u + c];
  }
  half r[16];
  for (uint c = 0u; c < 16u; ++c) {
    half highSquare = q[c + 16u] * q[c + 16u];
    float sum = (float)q[c] * (float)q[c] + (float)highSquare;
    r[c] = (half)sum;
  }
  half qSum = treeSum16(r);
  for (uint c = 0u; c < 16u; ++c) {
    half highSquare = k[c + 16u] * k[c + 16u];
    float sum = (float)k[c] * (float)k[c] + (float)highSquare;
    r[c] = (half)sum;
  }
  half kSum = treeSum16(r);
  half qNorm = (half)rsqrt((float)qSum);
  half kNorm = (half)rsqrt((float)kSum);
  half learnedScale = (half)as_type<float>(aux32[pc.scaleWordOffset + head]);
  half headScale = (half)sqrt(32.0f);
  for (uint c = 0u; c < 32u; ++c) {
    half nq0 = q[c] * qNorm;
    half nq1 = nq0 * headScale;
    half nq2 = nq1 * learnedScale;
    half nk = k[c] * kNorm;
    normalized[base + c] = e4m3Code(nq2);
    normalized[base + 32u + c] = e4m3Code(nk);
    normalized[base + 64u + c] = e4m3Code(qkv[base + 64u + c]);
  }
}
