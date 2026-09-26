#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <filament-iblprefilter/IBLPrefilterContext.h>
#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <filament/TransformManager.h>
#include <cstring>
#include <gltfio/Animator.h>
#include <gltfio/AssetLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/FilamentInstance.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/TextureProvider.h>
#include <gltfio/materials/uberarchive.h>
#include <math/mat3.h>
#include <stb_image.h>
#include <utils/EntityManager.h>

#include "json.h"

using namespace filament;
using namespace filament::math;
namespace fs = std::filesystem;

namespace {

std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

float3 toFloat3(const json::Value& v, float3 fallback) {
  if (v.kind != json::Value::Array || v.size() < 3) return fallback;
  return {(float)v[0].number, (float)v[1].number, (float)v[2].number};
}

float number(const json::Value& object, const char* key, float fallback) {
  return object.has(key) && object[key].kind == json::Value::Number ? (float)object[key].number : fallback;
}

// the first file of the directory with one of the extensions (a bare model / environment name in view.json is
// optional when there is only one candidate)
std::string findFile(const std::string& directory, std::initializer_list<const char*> extensions) {
  std::vector<std::string> candidates;
  for (const auto& entry : fs::directory_iterator(directory)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    for (const char* wanted : extensions)
      if (ext == wanted) candidates.push_back(entry.path().string());
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.empty() ? "" : candidates[0];
}

}  // namespace

SceneView parseSceneView(const std::string& viewJsonPath, const std::string& directory) {
  SceneView view;
  view.directory = directory;
  json::Value root = json::Parser(readFile(viewJsonPath)).parse();
  view.name = root.has("name") ? root["name"].str() : fs::path(directory).filename().string();
  view.model = root.has("model") ? (fs::path(directory) / root["model"].str()).string() : findFile(directory, {".gltf", ".glb"});
  view.environment = root.has("environment") ? (fs::path(directory) / root["environment"].str()).string() : findFile(directory, {".hdr"});
  view.environmentRotation = number(root, "environmentRotation", 0.0f);
  if (root.has("camera")) {
    const json::Value& camera = root["camera"];
    view.hasCamera = true;
    view.cameraPosition = toFloat3(camera.has("position") ? camera["position"] : json::Value{}, view.cameraPosition);
    view.cameraTarget = toFloat3(camera.has("target") ? camera["target"] : json::Value{}, view.cameraTarget);
    view.fov = number(camera, "fov", view.fov);
    view.nearPlane = number(camera, "near", view.nearPlane);
    view.farPlane = number(camera, "far", view.farPlane);
  }
  view.moveSpeed = number(root, "moveSpeed", 0.0f);
  view.exposure = number(root, "exposure", 1.0f);
  view.msaa = std::max(1, std::min(8, (int)number(root, "msaa", 1.0f)));
  view.environmentIntensity = number(root, "environmentIntensity", 1.0f);
  if (root.has("sun")) {
    view.hasSun = true;
    view.sunDirection = toFloat3(root["sun"].has("direction") ? root["sun"]["direction"] : json::Value{}, view.sunDirection);
    view.sunIntensity = number(root["sun"], "intensity", 1.0f);
  }
  if (root.has("background") && root["background"].kind == json::Value::String) {
    const std::string& hex = root["background"].str();
    if (hex.size() == 7 && hex[0] == '#') {
      unsigned rgb = (unsigned)strtoul(hex.c_str() + 1, nullptr, 16);
      auto srgb = [](float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); };
      view.background = {srgb(((rgb >> 16) & 255) / 255.0f), srgb(((rgb >> 8) & 255) / 255.0f), srgb((rgb & 255) / 255.0f)};
    }
  }
  view.backgroundEnvironment = root.has("backgroundEnvironment") && root["backgroundEnvironment"].boolean;
  if (view.model.empty()) throw std::runtime_error("no model in " + directory);
  return view;
}

std::vector<SceneView> scanScenes(const std::string& root) {
  std::vector<SceneView> scenes;
  if (!fs::is_directory(root)) return scenes;
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) continue;
    fs::path viewJson = entry.path() / "view.json";
    if (!fs::exists(viewJson)) continue;
    try {
      scenes.push_back(parseSceneView(viewJson.string(), entry.path().string()));
    } catch (const std::exception& e) {
      fprintf(stderr, "[scene] %s skipped: %s\n", entry.path().string().c_str(), e.what());
    }
  }
  std::sort(scenes.begin(), scenes.end(), [](const SceneView& a, const SceneView& b) { return a.name < b.name; });
  return scenes;
}

SceneView sceneFromArguments(const std::string& model, const std::string& environment) {
  // a model inside a scene directory takes that directory's view.json (camera, lighting); the arguments override
  fs::path directory = fs::path(model).parent_path();
  if (fs::exists(directory / "view.json")) {
    try {
      SceneView view = parseSceneView((directory / "view.json").string(), directory.string());
      view.model = model;
      if (!environment.empty()) view.environment = environment;
      return view;
    } catch (const std::exception& e) {
      fprintf(stderr, "[scene] %s ignored: %s\n", (directory / "view.json").string().c_str(), e.what());
    }
  }
  SceneView view;
  view.name = fs::path(model).stem().string();
  view.directory = directory.string();
  view.model = model;
  view.environment = environment;
  view.backgroundEnvironment = !environment.empty();
  return view;
}

// ------------------------------------------------------------------------------------------------------------

LoadedScene::LoadedScene(Engine& engine, const SceneView& view) : engine_(engine), view_(view) {
  scene_ = engine.createScene();
  // ---- the glTF through gltfio (ubershaders: every material variant is precompiled in the archive)
  materials_ = gltfio::createUbershaderProvider(&engine, UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
  assetLoader_ = gltfio::AssetLoader::create({&engine, materials_});
  std::string bytes = readFile(view.model);
  asset_ = assetLoader_->createAsset((const uint8_t*)bytes.data(), (uint32_t)bytes.size());
  if (!asset_) throw std::runtime_error("cannot parse " + view.model);
  // (the glTF's path resolves the .bin buffers, which cgltf reads from disk; the images go through the URI cache,
  // read here, so that the deprecated filesystem fallback is never used for them)
  gltfio::ResourceConfiguration configuration{};
  configuration.engine = &engine;
  configuration.gltfPath = view.model.c_str();
  configuration.normalizeSkinningWeights = true;
  resources_ = new gltfio::ResourceLoader(configuration);
  stb_ = gltfio::createStbProvider(&engine);
  ktx_ = gltfio::createKtx2Provider(&engine);
  resources_->addTextureProvider("image/png", stb_);
  resources_->addTextureProvider("image/jpeg", stb_);
  resources_->addTextureProvider("image/ktx2", ktx_);
  fs::path base = fs::path(view.model).parent_path();
  for (size_t i = 0, c = asset_->getResourceUriCount(); i < c; i++) {
    const char* uri = asset_->getResourceUris()[i];
    std::string ext = fs::path(uri).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    if (ext == ".bin") continue;
    std::string path = (base / uri).string();
    std::ifstream file(path, std::ios::binary);
    if (!file) { fprintf(stderr, "[scene] missing resource %s\n", path.c_str()); continue; }
    auto* buffer = new std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    gltfio::ResourceLoader::BufferDescriptor descriptor(buffer->data(), buffer->size(),
        [](void*, size_t, void* user) { delete static_cast<std::vector<uint8_t>*>(user); }, buffer);
    resources_->addResourceData(uri, std::move(descriptor));
  }
  if (!resources_->loadResources(asset_)) throw std::runtime_error("cannot load the resources of " + view.model);
  asset_->releaseSourceData();
  animator_ = asset_->getInstance()->getAnimator();
  scene_->addEntities(asset_->getEntities(), asset_->getEntityCount());
  renderables_ = asset_->getRenderableEntityCount();
  {
    // hair grooms (nodes flagged by the converter): alpha-blended strand tubes with depth writes, so that a
    // pile of strands does not accumulate into an opaque mass (the front strands win, as in a raster hair pass)
    auto& rcm = engine.getRenderableManager();
    size_t grooms = 0;
    for (size_t i = 0; i < asset_->getRenderableEntityCount(); i++) {
      utils::Entity entity = asset_->getRenderableEntities()[i];
      const char* extras = asset_->getExtras(entity);
      if (!extras || !strstr(extras, "\"groom\"")) continue;
      auto ri = rcm.getInstance(entity);
      for (size_t p = 0, c = rcm.getPrimitiveCount(ri); p < c; p++) {
        MaterialInstance* mi = rcm.getMaterialInstanceAt(ri, p);
        mi->setDepthWrite(true);
      }
      grooms++;
    }
    if (grooms) fprintf(stderr, "[scene] %zu groom renderables draw with depth writes\n", grooms);
  }
  lights_ = asset_->getLightEntityCount();
  Aabb bounds = asset_->getBoundingBox();
  center_ = bounds.center();
  radius_ = std::max(length(bounds.extent()), 1e-3f);
  if (!std::isfinite(radius_) || !std::isfinite(center_.x) || !std::isfinite(center_.y) || !std::isfinite(center_.z)) {
    // (a degenerate node makes gltfio's bounds infinite: fall back to a unit scene)
    center_ = {0, 0, 0};
    radius_ = 1.0f;
  }

  // ---- the lighting
  if (!view.environment.empty()) loadEnvironment(view.environment);
  colorSkybox_ = Skybox::Builder().color({view.background, 1.0f}).build(engine);
  setBackgroundEnvironment(view.backgroundEnvironment && skybox_);
  if (view.hasSun) {
    utils::Entity sun = utils::EntityManager::get().create();
    LightManager::Builder(LightManager::Type::DIRECTIONAL)
        .color({1.0f, 1.0f, 1.0f})
        .intensity(view.sunIntensity)
        .direction(normalize(-view.sunDirection))
        .castShadows(true)
        .build(engine, sun);
    scene_->addEntity(sun);
    sun_ = sun.getId();
  }
  fprintf(stderr, "[scene] %s: %zu renderables, %zu lights, %zu animations, bounds radius %.2f\n", view.name.c_str(), renderables_, lights_,
          animator_ ? animator_->getAnimationCount() : 0, radius_);
}

void LoadedScene::loadEnvironment(const std::string& path) {
  int w = 0, h = 0, n = 0;
  float* data = stbi_loadf(path.c_str(), &w, &h, &n, 3);
  if (!data) { fprintf(stderr, "[scene] cannot read the environment %s\n", path.c_str()); return; }
  Texture::PixelBufferDescriptor buffer(data, (size_t)w * h * 3 * sizeof(float), Texture::Format::RGB, Texture::Type::FLOAT,
                                        [](void* p, size_t, void*) { stbi_image_free(p); }, nullptr);
  Texture* equirect = Texture::Builder()
      .width((uint32_t)w).height((uint32_t)h).levels(0xff)
      .format(Texture::InternalFormat::R11F_G11F_B10F)
      .sampler(Texture::Sampler::SAMPLER_2D)
      .usage(Texture::Usage::DEFAULT | Texture::Usage::GEN_MIPMAPPABLE)
      .build(engine_);
  equirect->setImage(engine_, 0, std::move(buffer));
  IBLPrefilterContext context(engine_);
  IBLPrefilterContext::EquirectangularToCubemap equirectangularToCubemap(context);
  IBLPrefilterContext::SpecularFilter specularFilter(context);
  environmentCubemap_ = equirectangularToCubemap(equirect);
  engine_.destroy(equirect);
  reflections_ = specularFilter(environmentCubemap_);
  indirectLight_ = IndirectLight::Builder().reflections(reflections_).intensity(view_.environmentIntensity).build(engine_);
  setEnvironmentRotation(view_.environmentRotation);
  scene_->setIndirectLight(indirectLight_);
  skybox_ = Skybox::Builder().environment(environmentCubemap_).intensity(view_.environmentIntensity).build(engine_);
}

void LoadedScene::setEnvironmentIntensity(float intensity) {
  if (indirectLight_) indirectLight_->setIntensity(intensity);
  // the skybox has no intensity setter: rebuild it when the value changes
  if (skybox_ && std::abs(skybox_->getIntensity() - intensity) > 1e-6f) {
    bool shown = scene_->getSkybox() == skybox_;
    engine_.destroy(skybox_);
    skybox_ = Skybox::Builder().environment(environmentCubemap_).intensity(intensity).build(engine_);
    if (shown) scene_->setSkybox(skybox_);
  }
}

void LoadedScene::setSunIntensity(float intensity) {
  if (!sun_) return;
  auto& lcm = engine_.getLightManager();
  lcm.setIntensity(lcm.getInstance(utils::Entity::import((int32_t)sun_)), intensity);
}

void LoadedScene::setBackgroundEnvironment(bool environment) {
  scene_->setSkybox(environment && skybox_ ? skybox_ : colorSkybox_);
}

void LoadedScene::setEnvironmentRotation(float radians) {
  if (indirectLight_) indirectLight_->setRotation(mat3f::rotation(radians, float3{0, 1, 0}));
}

LoadedScene::~LoadedScene() {
  scene_->removeEntities(asset_->getEntities(), asset_->getEntityCount());
  if (sun_) {
    utils::Entity sun = utils::Entity::import((int32_t)sun_);
    scene_->remove(sun);
    engine_.destroy(sun);
    utils::EntityManager::get().destroy(sun);
  }
  engine_.destroy(skybox_);
  engine_.destroy(colorSkybox_);
  engine_.destroy(indirectLight_);
  engine_.destroy(reflections_);
  engine_.destroy(environmentCubemap_);
  assetLoader_->destroyAsset(asset_);
  delete resources_;
  delete stb_;
  delete ktx_;
  materials_->destroyMaterials();
  delete materials_;
  gltfio::AssetLoader::destroy(&assetLoader_);
  engine_.destroy(scene_);
}
