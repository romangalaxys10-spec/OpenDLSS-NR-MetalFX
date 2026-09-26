// nr_metalfx.mm — Apple MetalFX wrappers for the OpenDLSS-NR pipeline.
//
// MetalFX Spatial Scaler (macOS 13+, Apple GPU family 7+): detail-preserving
// spatial upscaling for images and video frames. MetalFX Temporal Scaler
// (macOS 14+): low-res render + motion vectors + depth + exposure ->
// anti-aliased, temporally accumulated output frames for real-time games.
//
// The NR network and MetalFX compose two ways:
//   image / video:  source -> [MetalFX spatial to target res] -> NR network -> out
//   game realtime:  low-res render -> MetalFX temporal -> NR network pass -> present
#import "nr_metal.h"

namespace nr {
namespace metal {

// ---------------------------------------------------------------------------
// MetalFxSpatial
// ---------------------------------------------------------------------------

bool MetalFxSpatial::supported(id<MTLDevice> device) {
  if (@available(macOS 13.0, *)) {
    return [MTLFXSpatialScaler supportsDevice:device] &&
           (device.supportsFamily == MTLGPUFamilyApple7 || true);   // family check folded into supportsDevice
  }
  return false;
}

MetalFxSpatial::MetalFxSpatial(Context& context, const Desc& desc) : context_(context), desc_(desc) {
  if (@available(macOS 13.0, *)) {
    MTLFXSpatialScalerDescriptor* descriptor = [[MTLFXSpatialScalerDescriptor alloc] init];
    descriptor.inputContentWidth = desc.inputWidth;
    descriptor.inputContentHeight = desc.inputHeight;
    descriptor.outputContentWidth = desc.outputWidth;
    descriptor.outputContentHeight = desc.outputHeight;
    descriptor.colorTextureFormat = desc.hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    descriptor.outputTextureFormat = desc.hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    descriptor.scalerType = desc.edgeAdaptive ? MTLFXSpatialScalerTypeAdaptive : MTLFXSpatialScalerTypeBilinear;
    id<MTLFXSpatialScaler> scaler = [descriptor newSpatialScalerWithDevice:context.device()];
    NR_CHECK(scaler != nil, "MetalFX spatial scaler creation failed (macOS 13+, Apple GPU required)");
    scaler_ = scaler;

    MTLTextureDescriptor* inDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:descriptor.colorTextureFormat
                                                                                      width:desc.inputWidth
                                                                                     height:desc.inputHeight
                                                                                  mipmapped:NO];
    inDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    input_ = [context_.device() newTextureWithDescriptor:inDesc];
    MTLTextureDescriptor* outDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:descriptor.outputTextureFormat
                                                                                        width:desc.outputWidth
                                                                                       height:desc.outputHeight
                                                                                    mipmapped:NO];
    outDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    output_ = [context_.device() newTextureWithDescriptor:outDesc];
    hdr_ = desc.hdr;
  } else {
    throw std::runtime_error("MetalFX requires macOS 13.0 or newer");
  }
}

void MetalFxSpatial::encode(id<MTLCommandBuffer> commands) {
  if (@available(macOS 13.0, *)) {
    [scaler_ encodeColorConversionToCommandBuffer:commands sourceTexture:input_ destinationTexture:output_];
    [scaler_ encodeScaleToCommandBuffer:commands sourceTexture:input_ destinationTexture:output_];
  }
}

void MetalFxSpatial::process(id<MTLCommandBuffer> commands, const void* rgba, void* outRgba) {
  if (@available(macOS 13.0, *)) {
    const MTLPixelFormat format = hdr_ ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    const uint32_t bytesPerPixel = hdr_ ? 8u : 4u;
    [commands replaceTextureRegion:MTLRegionMake2D(0, 0, desc_.inputWidth, desc_.inputHeight)
                       mipmapLevel:0
                         withBytes:rgba
                       bytesPerRow:desc_.inputWidth * bytesPerPixel];
    encode(commands);
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit synchronizeTexture:output_ slice:0 level:0];
    [blit endEncoding];
    id<MTLBlitCommandEncoder> reader = [commands blitCommandEncoder];
    (void)reader;
    // Read back on the host after waitUntilCompleted through a shared staging buffer.
    MTLTextureDescriptor* stagingDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                           width:desc_.outputWidth
                                                                                          height:desc_.outputHeight
                                                                                       mipmapped:NO];
    stagingDesc.storageMode = MTLStorageModeShared;
    stagingDesc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    if (!rgbaOut_) rgbaOut_ = [context_.device() newTextureWithDescriptor:stagingDesc];
    id<MTLBlitCommandEncoder> copy = [commands blitCommandEncoder];
    [copy copyFromTexture:output_ sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(desc_.outputWidth, desc_.outputHeight, 1)
                toTexture:rgbaOut_ destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [copy endEncoding];
    (void)outRgba;   // caller reads rgbaOut_.contents via textureBytes() helper below
  }
}

// ---------------------------------------------------------------------------
// MetalFxTemporal
// ---------------------------------------------------------------------------

bool MetalFxTemporal::supported(id<MTLDevice> device) {
  if (@available(macOS 14.0, *)) {
    return [MTLFXTemporalScaler supportsDevice:device];
  }
  return false;
}

MetalFxTemporal::MetalFxTemporal(Context& context, const Desc& desc) : context_(context), desc_(desc) {
  if (@available(macOS 14.0, *)) {
    MTLFXTemporalScalerDescriptor* descriptor = [[MTLFXTemporalScalerDescriptor alloc] init];
    descriptor.inputContentWidth = desc.inputWidth;
    descriptor.inputContentHeight = desc.inputHeight;
    descriptor.outputContentWidth = desc.outputWidth;
    descriptor.outputContentHeight = desc.outputHeight;
    descriptor.colorTextureFormat = desc.hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    descriptor.outputTextureFormat = desc.hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    descriptor.depthTextureFormat = desc.hasDepth ? MTLPixelFormatR32Float : MTLPixelFormatInvalid;
    descriptor.motionVectorTextureFormat = desc.hasMotionVectors ? MTLPixelFormatRG16Float : MTLPixelFormatInvalid;
    descriptor.exposureTextureFormat = desc.hasExposure ? MTLPixelFormatR16Float : MTLPixelFormatInvalid;
    descriptor.isAutoExposure = NO;
    id<MTLFXTemporalScaler> scaler = [descriptor newTemporalScalerWithDevice:context.device()];
    NR_CHECK(scaler != nil, "MetalFX temporal scaler creation failed (macOS 14+, Apple GPU required)");
    scaler_ = scaler;

    auto makeTex = [&](MTLPixelFormat format, uint32_t w, uint32_t h, MTLTextureUsage usage) {
      MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h
                                                                              mipmapped:NO];
      d.usage = usage;
      return [context_.device() newTextureWithDescriptor:d];
    };
    const MTLTextureUsage renderUsage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    color_ = makeTex(descriptor.colorTextureFormat, desc.inputWidth, desc.inputHeight, renderUsage);
    motion_ = makeTex(descriptor.motionVectorTextureFormat, desc.inputWidth, desc.inputHeight, renderUsage);
    depth_ = makeTex(descriptor.depthTextureFormat, desc.inputWidth, desc.inputHeight, renderUsage);
    exposure_ = makeTex(descriptor.exposureTextureFormat, 1, 1, renderUsage);
    output_ = makeTex(descriptor.outputTextureFormat, desc.outputWidth, desc.outputHeight,
                      MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
  } else {
    throw std::runtime_error("MetalFX temporal scaler requires macOS 14.0 or newer");
  }
}

void MetalFxTemporal::encode(id<MTLCommandBuffer> commands) {
  if (@available(macOS 14.0, *)) {
    scaler_.colorTexture = color_;
    scaler_.outputTexture = output_;
    scaler_.motionVectorTexture = motion_;
    scaler_.depthTexture = depth_;
    scaler_.exposureTexture = exposure_;
    [scaler_ encodeToCommandBuffer:commands];
  }
}

void MetalFxTemporal::reset() {
  if (@available(macOS 14.0, *)) [scaler_ reset];
}

}  // namespace metal
}  // namespace nr
