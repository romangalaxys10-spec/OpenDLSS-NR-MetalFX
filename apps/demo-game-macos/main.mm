// main.mm - OpenDLSS NR Realtime: the macOS game-path demo.
//
// The complete game pipeline of OpenDLSS-NR-MetalFX, end to end:
//   1. A procedural scene (drifting SDF shapes, animated glow) renders at low
//      res into color + motion-vector + depth textures (a stand-in for a real
//      engine's G-buffer pass; docs/GAMES.md has the engine contract).
//   2. MetalFX Spatial Scaler (Adaptive) upscales + anti-aliases the frame
//      (macOS 13+, M1/M2/M3; engines with real motion vectors switch to
//      MTLFXTemporalScaler - docs/GAMES.md shows both).
//   3. The NR network (the OpenDLSS-NR 71-block generative renderer) runs on
//      the upscaled frame with the demo's style conditioning.
//   4. The composed frame presents through MTKView.
//
// Build: scripts/macos/build.sh demo   (CMake target demo-game-macos)
// Run:   build/demo-game-macos  (set OPENDLSS_MODEL, default models/nr)
#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#import "nr_metal.h"
#import "nr_frame.h"

#import <chrono>
#import <cmath>
#import <cstdlib>
#import <memory>
#import <vector>

namespace {

constexpr uint32_t kRenderScale = 2;   // low-res render scale of the "game" pass

const char* kSceneSource = R"(#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
  float time;
  float aspect;
  float glow;
  float pad;
};

vertex float4 vsFullscreen(uint vid [[vertex_id]]) {
  float2 pos[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  return float4(pos[vid], 0, 1);
}

// The "game": animated SDF scene with per-pixel color, motion and depth.
fragment void fsScene(float4 pos [[stage_in]],
                      constant SceneUniforms& u [[buffer(0)]],
                      texture2d<float, access::write> color [[texture(0)]],
                      texture2d<float, access::write> motion [[texture(1)]],
                      texture2d<float, access::write> depth [[texture(2)]]) {
  uint2 px = (uint2)pos.xy;
  uint2 size = uint2(color.get_width(), color.get_height());
  if (px.x >= size.x || px.y >= size.y) return;
  float2 uv = ((float2)px + 0.5) / float2(size);
  float2 p = float2((uv.x - 0.5) * u.aspect, uv.y - 0.5);

  float t = u.time;
  float2 c1 = 0.30 * float2(sin(t * 1.1), cos(t * 0.7));
  float2 c2 = -0.28 * float2(cos(t * 0.9), sin(t * 1.3));
  float d1 = length(p - c1) - 0.16;
  float d2 = length(p - c2) - 0.11;
  float ring = abs(length(p) - 0.42) - 0.02;
  float d = min(min(d1, d2), ring);
  float shape = smoothstep(0.004, -0.004, d);
  float3 base = mix(float3(0.02, 0.03, 0.05),
                    select(float3(0.85, 0.45, 0.20), float3(0.20, 0.65, 0.95), d2 < d1), shape);
  base += u.glow * float3(0.30, 0.15, 0.05) * exp(-abs(d) * 12.0);
  base = pow(base, float3(1.0 / 2.2));   // encode; the NR stage consumes display codes
  color.write(float4(base, 1.0), px);
  motion.write(float4(0, 0, 0, 0), px);
  depth.write(float4(select(1.0, 0.5 - 0.2 * d1, shape > 0.5), 0, 0, 0), px);
}

fragment float4 fsPresent(float4 pos [[stage_in]], texture2d<float> source [[texture(0)]]) {
  constexpr sampler s(address::clamp_to_edge, filter::linear);
  float2 uv = float2(pos.xy) / float2(source.get_width(), source.get_height());
  return source.sample(s, uv);
}
)";

struct DemoUniforms {
  float time;
  float aspect;
  float glow;
  float pad;
};

}  // namespace

@interface AppDelegate : NSObject <NSApplicationDelegate, MTKViewDelegate>
@property(strong) NSWindow* window;
@property(strong) MTKView* view;
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) id<MTLLibrary> sceneLibrary;
@property(strong) id<MTLRenderPipelineState> scenePipeline;
@property(strong) id<MTLRenderPipelineState> presentPipeline;
@property(strong) id<MTLFXSpatialScaler> spatialScaler;
@property(assign) uint32_t lowWidth, lowHeight, outWidth, outHeight;
@property(assign) double startTime;
@property(strong) id<MTLTexture> lowColor, lowMotion, lowDepth, upscaled, nrTexture;
@property std::unique_ptr<nr::frame::Session> session;
@property std::vector<uint8_t> readback;
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification*)notification {
  (void)notification;
  std::string model = "models/nr";
  float style = 64.0f;
  if (const char* m = getenv("OPENDLSS_MODEL")) model = m;
  if (const char* s = getenv("OPENDLSS_STYLE")) style = (float)atof(s);

  self.device = MTLCreateSystemDefaultDevice();
  if (!self.device) { NSLog(@"OpenDLSS NR Realtime: no Metal device"); exit(1); }
  self.queue = [self.device newCommandQueue];

  NSRect frame = NSMakeRect(0, 0, 1600, 900);
  NSUInteger styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable;
  self.window = [[NSWindow alloc] initWithContentRect:frame styleMask:styleMask
                                              backing:NSBackingStoreBuffered defer:NO];
  self.window.title = @"OpenDLSS NR Realtime - MetalFX + Neural Rendering";
  self.window.contentMinSize = NSMakeSize(640, 360);

  self.view = [[MTKView alloc] initWithFrame:frame device:self.device];
  self.view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
  self.view.depthStencilPixelFormat = MTLPixelFormatInvalid;
  self.view.delegate = self;
  self.window.contentView = self.view;

  NSError* error = nil;
  self.sceneLibrary = [self.device newLibraryWithSource:[NSString stringWithUTF8String:kSceneSource]
                                                options:nil error:&error];
  if (!self.sceneLibrary) { NSLog(@"scene shader compile failed: %@", error.localizedDescription); exit(1); }
  id<MTLFunction> vs = [self.sceneLibrary newFunctionWithName:@"vsFullscreen"];
  id<MTLFunction> fsScene = [self.sceneLibrary newFunctionWithName:@"fsScene"];
  id<MTLFunction> fsPresent = [self.sceneLibrary newFunctionWithName:@"fsPresent"];

  MTLRenderPipelineDescriptor* sceneDesc = [MTLRenderPipelineDescriptor new];
  sceneDesc.vertexFunction = vs;
  sceneDesc.fragmentFunction = fsScene;
  sceneDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  self.scenePipeline = [self.device newRenderPipelineStateWithDescriptor:sceneDesc error:&error];
  if (!self.scenePipeline) { NSLog(@"scene pipeline failed: %@", error.localizedDescription); exit(1); }

  MTLRenderPipelineDescriptor* presentDesc = [MTLRenderPipelineDescriptor new];
  presentDesc.vertexFunction = vs;
  presentDesc.fragmentFunction = fsPresent;
  presentDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  self.presentPipeline = [self.device newRenderPipelineStateWithDescriptor:presentDesc error:&error];
  if (!self.presentPipeline) { NSLog(@"present pipeline failed: %@", error.localizedDescription); exit(1); }

  self.startTime = [NSDate timeIntervalSinceReferenceDate];

  nr::frame::SessionOptions sessionOptions;
  sessionOptions.modelDirectory = model;
  sessionOptions.params.validWidth = 1600;
  sessionOptions.params.validHeight = 900;
  sessionOptions.params.sourceWidth = 1600;
  sessionOptions.params.sourceHeight = 900;
  sessionOptions.params.style = style;
  sessionOptions.params.skinStructure = -1.0f;
  try {
    self.session = nr::frame::createSession(sessionOptions);
  } catch (const std::exception& e) {
    NSAlert* alert = [NSAlert new];
    alert.messageText = @"OpenDLSS NR Realtime needs a model directory";
    alert.informativeText = [NSString stringWithFormat:@"%s\n\nSet OPENDLSS_MODEL (default models/nr "
                                                     @"next to the app); see docs/INSTALL.md.", e.what()];
    [alert runModal];
    exit(2);
  }

  [self.window center];
  [self.window makeKeyAndOrderFront:nil];
  [NSApp activateIgnoringOtherApps:YES];
}

- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  [self rebuildTargets:(uint32_t)size.width height:(uint32_t)size.height];
}

- (void)rebuildTargets:(uint32_t)width height:(uint32_t)height {
  self.outWidth = width;
  self.outHeight = height;
  self.lowWidth = width / kRenderScale;
  self.lowHeight = height / kRenderScale;

  auto makeTex = ^(MTLPixelFormat format, uint32_t w, uint32_t h, MTLTextureUsage usage) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h
                                                                            mipmapped:NO];
    d.usage = usage;
    d.storageMode = MTLStorageModeShared;
    return [self.device newTextureWithDescriptor:d];
  };
  const MTLTextureUsage gpu = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  self.lowColor = makeTex(MTLPixelFormatBGRA8Unorm, self.lowWidth, self.lowHeight, gpu);
  self.lowMotion = makeTex(MTLPixelFormatRG16Float, self.lowWidth, self.lowHeight, gpu);
  self.lowDepth = makeTex(MTLPixelFormatR32Float, self.lowWidth, self.lowHeight, gpu);
  self.upscaled = makeTex(MTLPixelFormatBGRA8Unorm, self.outWidth, self.outHeight, gpu);
  self.nrTexture = makeTex(MTLPixelFormatRGBA8Unorm, self.outWidth, self.outHeight, gpu);
  self.readback.assign((size_t)self.outWidth * self.outHeight * 4, 0);

  if (@available(macOS 13.0, *)) {
    MTLFXSpatialScalerDescriptor* desc = [MTLFXSpatialScalerDescriptor new];
    desc.inputContentWidth = self.lowWidth;
    desc.inputContentHeight = self.lowHeight;
    desc.outputContentWidth = self.outWidth;
    desc.outputContentHeight = self.outHeight;
    desc.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
    desc.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
    desc.scalerType = MTLFXSpatialScalerTypeAdaptive;
    self.spatialScaler = [desc newSpatialScalerWithDevice:self.device];
  }
}

- (void)drawInMTKView:(MTKView*)view {
  (void)view;
  if (self.lowWidth == 0)
    [self rebuildTargets:(uint32_t)view.drawableSize.width height:(uint32_t)view.drawableSize.height];
  double now = [NSDate timeIntervalSinceReferenceDate] - self.startTime;

  @autoreleasepool {
    id<MTLCommandBuffer> commands = [self.queue commandBuffer];

    DemoUniforms uniforms{(float)now, (float)self.outWidth / (float)self.outHeight, 1.25f, 0.0f};
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = self.lowColor;
    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> scene = [commands renderCommandEncoderWithDescriptor:pass];
    [scene setRenderPipelineState:self.scenePipeline];
    [scene setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    [scene setFragmentTexture:self.lowColor atIndex:0];
    [scene setFragmentTexture:self.lowMotion atIndex:1];
    [scene setFragmentTexture:self.lowDepth atIndex:2];
    [scene drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [scene endEncoding];

    if (@available(macOS 13.0, *)) {
      [self.spatialScaler encodeColorConversionToCommandBuffer:commands
                                                 sourceTexture:self.lowColor
                                            destinationTexture:self.upscaled];
      [self.spatialScaler encodeScaleToCommandBuffer:commands
                                       sourceTexture:self.lowColor
                                  destinationTexture:self.upscaled];
    }

    [commands commit];
    [commands waitUntilCompleted];
    [self.upscaled getBytes:self.readback.data()
                bytesPerRow:self.outWidth * 4
                 fromRegion:MTLRegionMake2D(0, 0, self.outWidth, self.outHeight)
                mipmapLevel:0];
    std::vector<float> proxy((size_t)self.outWidth * self.outHeight * 4);
    for (size_t i = 0; i < proxy.size(); ++i) proxy[i] = self.readback[i] * (1.0f / 255.0f);
    std::vector<uint8_t> nrOut;
    self.session->processFrame(proxy.data(), self.outWidth, nrOut);

    [self.nrTexture replaceRegion:MTLRegionMake2D(0, 0, self.outWidth, self.outHeight)
                      mipmapLevel:0
                        withBytes:nrOut.data()
                      bytesPerRow:self.outWidth * 4];
    id<MTLCommandBuffer> presentCommands = [self.queue commandBuffer];
    id<CAMetalDrawable> drawable = [view currentDrawable];
    if (!drawable) return;
    MTLRenderPassDescriptor* presentPass = [MTLRenderPassDescriptor renderPassDescriptor];
    presentPass.colorAttachments[0].texture = drawable.texture;
    presentPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    presentPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [presentCommands renderCommandEncoderWithDescriptor:presentPass];
    [encoder setRenderPipelineState:self.presentPipeline];
    [encoder setFragmentTexture:self.nrTexture atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [presentCommands presentDrawable:drawable];
    [presentCommands commit];
  }
}

@end

int main(int argc, char** argv) {
  @autoreleasepool {
    NSApplication* app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    AppDelegate* delegate = [AppDelegate new];
    app.delegate = delegate;
    [app run];
    (void)argc;
    (void)argv;
  }
  return 0;
}
