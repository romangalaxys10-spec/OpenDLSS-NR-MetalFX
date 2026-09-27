// Single translation unit that compiles the vendored stb libraries.
// stb_image / stb_image_write are public domain (see NOTICE).
#ifdef STB_IMAGE_IMPLEMENTATION
#error "stb already configured"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
