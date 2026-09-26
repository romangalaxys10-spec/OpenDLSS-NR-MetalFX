// The pinned WebGI IIFE publishes its runtime API directly on window. This
// facade gives the local TypeScript bundle normal imports without bundling a
// second WebGI or Three.js instance.
const api = globalThis as any;

if (!api.ViewerApp || !api.GenericFilterPlugin || !api.Mesh) {
  throw new Error('WebGI 0.22.1 CDN bundle must load before the DLSS add-on');
}

export const {
  ACESFilmicToneMapping,
  AgXToneMapping,
  BackSide,
  BaseGroundPlugin,
  BasicShadowMap,
  Box3,
  BufferAttribute,
  BufferGeometry,
  Camera,
  CanvasTexture,
  Color,
  ContactShadowGroundPlugin,
  DataTexture,
  DebugPlugin,
  DirectionalLight,
  DirectionalLight2,
  DiamondPlugin,
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  FloatType,
  FogExp2,
  FrontSide,
  Frustum,
  GBufferPlugin,
  GenericFilterPlugin,
  GemRefractionPlugin,
  GLSL3,
  GLTFMeshOptPlugin,
  Group,
  HalfFloatType,
  InstancedMesh,
  KTX2LoadPlugin,
  Light,
  LinearFilter,
  LinearMipmapLinearFilter,
  Material,
  MathUtils,
  Matrix3,
  Matrix4,
  Mesh,
  MeshDepthMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  MeshStandardMaterial2,
  NearestFilter,
  NeutralToneMapping,
  NoBlending,
  NoColorSpace,
  Object3D,
  onChange,
  OrthographicCamera,
  PCFShadowMap,
  PCFSoftShadowMap,
  PerspectiveCamera,
  Plane,
  PropertyBinding,
  ProgressivePlugin,
  RedFormat,
  Reflector2,
  RGBAFormat,
  RGFormat,
  Scene,
  serialize,
  setThreeRendererMode,
  shaderReplaceString,
  ShadowMapBaker,
  ShaderMaterial,
  Side,
  Sphere,
  SpotLight,
  SRGBColorSpace,
  SSAOPlugin,
  SSBevelPlugin,
  SSContactShadows,
  TemporalAAPlugin,
  Texture,
  TonemapPlugin,
  uiFolder,
  uiToggle,
  UnsignedByteType,
  UnsignedIntType,
  Vector2,
  Vector3,
  Vector4,
  ViewerApp,
  VSMShadowMap,
  WebGLRenderer,
  WebGLRenderTarget,
  addBasePlugins,
  mergeGeometries,
} = api;

// OrbitControls is used only as a TypeScript annotation by the scene modules.
export type OrbitControls = any;
export type DlssBridgeFrame = any;
export type DlssBridgeFrameConsumer = any;
export type GroundOptions = any;
export type IEvent<T = any> = any;
export type IMaterial = any;
export type IRenderTarget = any;
export type IShaderPropertiesUpdater = any;
export type IUniform = any;
export type IViewerPluginAsync = any;
export type MaterialExtension = any;
export type UiObjectConfig = any;
