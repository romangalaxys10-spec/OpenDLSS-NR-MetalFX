// Demo scenes for the Filament renderer: the scene directories (a glTF, an equirectangular .hdr and a view.json:
// camera, lighting, look), their loading through gltfio, the image-based lighting through iblprefilter, the sun,
// the background.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include <math/vec3.h>

namespace filament {
class Engine;
class Scene;
class Texture;
class IndirectLight;
class Skybox;
namespace gltfio {
class AssetLoader;
class FilamentAsset;
class MaterialProvider;
class ResourceLoader;
class TextureProvider;
class Animator;
}
}

// view.json (see demo/README.md); everything but the model and the environment has a default
struct SceneView {
  std::string name, directory, model, environment;
  float environmentRotation = 0.0f;
  filament::math::float3 cameraPosition{0, 1, 5}, cameraTarget{0, 0, 0};
  float fov = 50.0f, nearPlane = 0.05f, farPlane = 1000.0f;
  float moveSpeed = 0.0f;   // 0: the distance to the target per second
  float exposure = 1.0f, environmentIntensity = 1.0f;
  int msaa = 1;   // samples of the scene pass (1, 2, 4, 8): sub-pixel strands (hair grooms) need 4+
  bool hasSun = false;
  filament::math::float3 sunDirection{0.5f, 0.8f, 0.3f};   // towards the sun
  float sunIntensity = 1.0f;
  filament::math::float3 background{0, 0, 0};
  bool backgroundEnvironment = false;
  bool hasCamera = false;
};

// every directory under `root` with a view.json, sorted by name
std::vector<SceneView> scanScenes(const std::string& root);
// a view for a bare model / environment pair on the command line
SceneView sceneFromArguments(const std::string& model, const std::string& environment);
SceneView parseSceneView(const std::string& viewJsonPath, const std::string& directory);

// The loaded state of one scene: the glTF asset in a filament::Scene, its lighting, its animator
class LoadedScene {
 public:
  LoadedScene(filament::Engine& engine, const SceneView& view);
  ~LoadedScene();
  filament::Scene* scene() const { return scene_; }
  filament::gltfio::Animator* animator() const { return animator_; }
  const SceneView& view() const { return view_; }
  filament::math::float3 boundsCenter() const { return center_; }
  float boundsRadius() const { return radius_; }
  size_t renderableCount() const { return renderables_; }
  size_t lightCount() const { return lights_; }
  // the look controls (the UI's): applied to the lights / camera exposure by the app each frame
  void setEnvironmentIntensity(float intensity);
  void setSunIntensity(float intensity);
  void setBackgroundEnvironment(bool environment);
  void setEnvironmentRotation(float radians);

 private:
  void loadEnvironment(const std::string& path);
  filament::Engine& engine_;
  SceneView view_;
  filament::Scene* scene_ = nullptr;
  filament::gltfio::AssetLoader* assetLoader_ = nullptr;
  filament::gltfio::MaterialProvider* materials_ = nullptr;
  filament::gltfio::ResourceLoader* resources_ = nullptr;
  filament::gltfio::TextureProvider* stb_ = nullptr;
  filament::gltfio::TextureProvider* ktx_ = nullptr;
  filament::gltfio::FilamentAsset* asset_ = nullptr;
  filament::gltfio::Animator* animator_ = nullptr;
  filament::Texture* environmentCubemap_ = nullptr;
  filament::Texture* reflections_ = nullptr;
  filament::IndirectLight* indirectLight_ = nullptr;
  filament::Skybox* skybox_ = nullptr;      // the environment
  filament::Skybox* colorSkybox_ = nullptr; // the solid background
  uint32_t sun_ = 0;   // utils::Entity id
  filament::math::float3 center_{0, 0, 0};
  float radius_ = 1.0f;
  size_t renderables_ = 0, lights_ = 0;
};
