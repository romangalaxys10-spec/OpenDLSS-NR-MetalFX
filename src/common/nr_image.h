// nr_image.h — image decode/encode for the NR pipeline (stb-based).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nr {
namespace image {

struct Image {
  uint32_t width = 0, height = 0, channels = 0;
  std::vector<float> rgba;   // display code values [0,1], RGBA, row-major [w*h*4]
};

// Decodes PNG/JPEG/BMP/TGA/GIF/PSD/HDR; LDR sources are normalized to [0,1].
// An 8-bit source is treated as sRGB code values (passed through, no transfer
// function change — the network consumes display code values, matching the
// original fixtures).
Image load(const std::string& path);
// Encodes RGBA8 (packed bytes) as PNG (or JPEG when the name ends in .jpg/.jpeg).
void save(const std::string& path, const uint8_t* rgba, uint32_t width, uint32_t height);

}  // namespace image
}  // namespace nr
