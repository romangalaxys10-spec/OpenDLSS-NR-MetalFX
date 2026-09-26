// nr_image.cpp — stb-backed image I/O.
#include "nr_image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_GIF
#define STBI_ASSERT(x)
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_ASSERT(x)
#include "stb_image_write.h"

#include <cstring>
#include <stdexcept>

namespace nr {
namespace image {

Image load(const std::string& path) {
  int w = 0, h = 0, channels = 0;
  stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
  if (!pixels) throw std::runtime_error("cannot decode image " + path + ": " + stbi_failure_reason());
  Image image;
  image.width = (uint32_t)w;
  image.height = (uint32_t)h;
  image.channels = 4;
  image.rgba.resize((size_t)w * h * 4);
  for (size_t i = 0; i < image.rgba.size(); ++i) image.rgba[i] = pixels[i] * (1.0f / 255.0f);
  stbi_image_free(pixels);
  return image;
}

void save(const std::string& path, const uint8_t* rgba, uint32_t width, uint32_t height) {
  const bool jpeg = path.size() > 4 && (path.compare(path.size() - 4, 4, ".jpg") == 0 ||
                                        path.compare(path.size() - 5, 5, ".jpeg") == 0);
  int ok = jpeg ? stbi_write_jpg(path.c_str(), (int)width, (int)height, 4, rgba, 95)
                : stbi_write_png(path.c_str(), (int)width, (int)height, 4, rgba, (int)width * 4);
  if (!ok) throw std::runtime_error("cannot write image " + path);
}

}  // namespace image
}  // namespace nr
