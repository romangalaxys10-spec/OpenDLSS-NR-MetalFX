// MetalBackend.mm — the MetalFX-accelerated macOS backend (Apple Silicon).
//
// MetalFX integration:
//   * image / video scaling  -> MTLFXSpatialScaler (macOS 13+, Apple Silicon)
//   * game temporal scaling  -> MTLFXTemporalScaler (motion vectors, depth,
//                               exposure, jitter feedback — the DLSS-SR role)
//   * the neural rendering graph runs in our Metal compute kernels
//     (Graph.metal), fp16 storage + E4M3 publications, identical semantics
//     to the CPU reference in core/backends/cpu.
//
// Every op falls back to our own kernels when the MetalFX object is
// unavailable, so the backend always works — MetalFX is the accelerator,
// not a requirement.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include "opendlss/backend.h"
#include "opendlss/fp16.h"
#include "opendlss/pipeline.h"
#include "opendlss/logging.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace opendlss {

namespace {

#define ODL_OBJC(x) (reinterpret_cast<id>(x))

id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib,
                                     std::map<std::string, id<MTLComputePipelineState>>& cache,
                                     id<MTLDevice> dev, const char* fn) {
    auto it = cache.find(fn);
    if (it != cache.end()) return it->second;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
    if (!f) { log_error("metal: missing kernel %s", fn); return nil; }
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { log_error("metal: PSO failed for %s: %s", fn, err.localizedDescription.UTF8String); return nil; }
    cache[fn] = p;
    return p;
}

// Host-side buffer wrapper
struct GpuBuffer {
    id<MTLBuffer> buf = nil;
    size_t bytes = 0;
};

} // namespace

class MetalBackend final : public IBackend {
public:
    MetalBackend() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) { log_error("metal: no Metal device"); return; }
        queue_ = [device_ newCommandQueue];
        NSString* srcPath = @OPENDLSS_METAL_SOURCE_PATH;
        NSError* err = nil;
        lib_ = [device_ newDefaultLibrary];
        if (!lib_) {
            // runtime compile fallback from source tree
            NSString* src = [NSString stringWithContentsOfFile:srcPath
                                                      encoding:NSUTF8StringEncoding error:nil];
            if (src) lib_ = [device_ newLibraryWithSource:src options:nil error:&err];
        }
        if (!lib_) {
            log_error("metal: shader library unavailable (%s)", err.localizedDescription.UTF8String);
            return;
        }
        info_.kind = BackendKind::Metal;
        info_.name = "metal+metalfx";
        info_.deviceName = device_.name.UTF8String;
        info_.neuralGraph = true;
        info_.fp16Storage = true;
        info_.e4m3Quant = true;
        info_.maxTextureDim = 16384;
        if (@available(macOS 13.0, *)) {
            // MetalFX availability: spatial on all Apple Silicon, temporal needs Apple7+
            info_.metalFxSpatial = true;
            info_.metalFxTemporal = [device_ supportsFamily:MTLGPUFamilyApple7] ? true : false;
        }
        info_.details = "MetalFX spatial/temporal scalers + Metal compute graph";
        ok_ = true;
    }

    const BackendInfo& info() const override { return info_; }

    // ---------------- image ops ----------------
    bool denoiseSpatial(const Image& input, Image& output, float strength) override {
        if (!ok_ || strength <= 0) { output = input; return true; }
        id<MTLTexture> src = makeTexture(input);
        id<MTLTexture> dst = newTexture(input.width, input.height);
        GpuBuffer sbuf = floatBuf({strength});
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_denoise");
        [enc setComputePipelineState:p];
        [enc setTexture:src atIndex:0];
        [enc setBuffer:sbuf.buf offset:0 atIndex:0];
        [enc setTexture:dst atIndex:1];
        dispatch(enc, p, MTLSizeMake(input.width, input.height, 1), MTLSizeMake(16, 16, 1));
        [enc endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        readTexture(dst, output);
        return true;
    }

    bool upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) override {
        if (!ok_) { output = input; return false; }
        const uint32_t OW = input.width * factor, OH = input.height * factor;
        output = Image(OW, OH);

        id<MTLTexture> src = makeTexture(input);
        id<MTLTexture> tmp = newTexture(OW, input.height);
        id<MTLTexture> up  = newTexture(OW, OH);
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // 1. MetalFX spatial scaler when available, else our Lanczos3 passes
        id<MTLFXSpatialScaler> fx = spatialScaler(input.width, input.height, OW, OH);
        if (fx) {
            fx.colorTexture = src;
            fx.outputTexture = up;
            [fx encodeToCommandBuffer:cb];
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            cb = [queue_ commandBuffer];
            enc = [cb computeCommandEncoder];
        } else {
            GpuBuffer w = floatBuf({float(OW)});
            id<MTLComputePipelineState> ph = pipeline(lib_, pipes_, device_, "odl_upscale_h");
            [enc setComputePipelineState:ph];
            [enc setTexture:src atIndex:0]; [enc setBuffer:w.buf offset:0 atIndex:0];
            [enc setTexture:tmp atIndex:1];
            dispatch(enc, ph, MTLSizeMake(OW, input.height, 1), MTLSizeMake(16, 16, 1));
            GpuBuffer h = floatBuf({float(OH)});
            id<MTLComputePipelineState> pv = pipeline(lib_, pipes_, device_, "odl_upscale_v");
            [enc setComputePipelineState:pv];
            [enc setTexture:tmp atIndex:0]; [enc setBuffer:h.buf offset:0 atIndex:0];
            [enc setTexture:up atIndex:1];
            dispatch(enc, pv, MTLSizeMake(OW, OH, 1), MTLSizeMake(16, 16, 1));
        }

        // 2. sharpen
        if (sharpen > 0) {
            id<MTLTexture> fin = newTexture(OW, OH);
            GpuBuffer sbuf = floatBuf({sharpen});
            id<MTLComputePipelineState> ps = pipeline(lib_, pipes_, device_, "odl_sharpen");
            [enc setComputePipelineState:ps];
            [enc setTexture:up atIndex:0]; [enc setBuffer:sbuf.buf offset:0 atIndex:0];
            [enc setTexture:fin atIndex:1];
            dispatch(enc, ps, MTLSizeMake(OW, OH, 1), MTLSizeMake(16, 16, 1));
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            readTexture(fin, output);
            return true;
        }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        readTexture(up, output);
        return true;
    }

    // ---------------- neural graph ----------------
    bool loadModel(const std::string& modelDir) override {
        model_ = Model::load(modelDir);
        if (!model_) return false;
        tensorBufs_.clear();
        // upload every tensor as a half buffer (host decodes E4M3 -> f16 once)
        for (const auto& t : model_->tensors()) {
            GpuBuffer gb;
            gb.bytes = t.f16.size() * 2;
            gb.buf = [device_ newBufferWithBytes:t.f16.data()
                                          length:gb.bytes
                                         options:MTLResourceStorageModeShared];
            tensorBufs_[t.name] = gb;
        }
        return true;
    }
    bool hasNeuralGraph() const override { return model_ != nullptr && ok_; }
    const Model* model() const override { return model_.get(); }

    bool runNeuralGraph(const Geometry& geom, const uint16_t* features,
                        float* head, uint64_t frameSeed) override {
        if (!model_) return false;
        const auto& sched = model_->schedule();
        if (sched.empty()) return false;
        const uint32_t FW = geom.fieldWidth, FH = geom.fieldHeight;
        const uint32_t nLevels = model_->config().levels;

        // features buffer (field*16 half)
        GpuBuffer feat;
        feat.bytes = size_t(FW) * FH * 16 * 2;
        feat.buf = [device_ newBufferWithLength:feat.bytes options:MTLResourceStorageModeShared];
        std::memcpy(feat.buf.contents, features, feat.bytes);

        size_t maxTC = size_t(FW) * FH * 32;
        for (const auto& se : sched) {
            const uint32_t lw = se.onField ? FW : geom.levels[std::min(se.level, nLevels-1)].width;
            const uint32_t lh = se.onField ? FH : geom.levels[std::min(se.level, nLevels-1)].height;
            maxTC = std::max(maxTC, size_t(lw) * lh * se.channels);
            maxTC = std::max(maxTC, size_t(lw) * lh * se.channels * 3);  // qkv
        }
        GpuBuffer state = makeBuffer(maxTC * 2);
        GpuBuffer ffnOut = makeBuffer(maxTC * 2);
        GpuBuffer projOut = makeBuffer(maxTC * 2);
        projIn_ = makeBuffer(maxTC * 2);
        qkvBuf_ = makeBuffer(maxTC * 2);
        GpuBuffer headBuf = makeBuffer(size_t(FW) * FH * 4 * 4);

        // resolve block geometry, tracking the live token count
        uint32_t liveW = FW, liveH = FH, liveC = 0;
        GpuBuffer skipBufs[9]; bool hasSkip[9] = {};
        GpuBuffer block0Buf; bool hasBlock0 = false;

        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (size_t bi = 0; bi < sched.size(); ++bi) {
            const BlockScheduleEntry e = sched[bi];
            const uint32_t bw = e.onField ? FW : geom.levels[std::min(e.level, nLevels-1)].width;
            const uint32_t bh = e.onField ? FH : geom.levels[std::min(e.level, nLevels-1)].height;
            const uint32_t tokens = bw * bh;
            const uint32_t C = e.channels;

            if (bi == 0) {
                liveC = C;
                runInputEmbed(enc, feat, state, FW, FH, C);
            } else {
                const BlockScheduleEntry p = sched[bi-1];
                const int32_t effE = e.onField ? -1 : int32_t(e.level);
                const int32_t effP = p.onField ? -1 : int32_t(p.level);
                if (effE > effP) {
                    // pool + up-GEMM
                    GpuBuffer pooled = makeBuffer(size_t(bw) * bh * p.channels * 2);
                    runPool(enc, state, liveW, liveH, bw, bh, p.channels, pooled);
                    runGemm(enc, pooled, tensorNamed("trans" + std::to_string(p.onField ? -1 : p.level) + ".up"),
                            GpuBuffer{}, bw*bh, C, p.channels, 0, state);
                    liveC = C;
                } else if (effE < effP) {
                    // down-GEMM + upsample + skip
                    GpuBuffer proj = makeBuffer(size_t(liveW) * liveH * C * 2);
                    runGemm(enc, state, tensorNamed("trans" + std::to_string(p.level) + ".down"),
                            GpuBuffer{}, liveW*liveH, C, p.channels, 1, proj);
                    GpuBuffer up = makeBuffer(size_t(bw) * bh * C * 2);
                    runUpsample(enc, proj, liveW, liveH, bw, bh, C, up);
                    GpuBuffer out = makeBuffer(size_t(bw) * bh * C * 2);
                    bool hasSk = e.onField ? hasBlock0 : hasSkip[e.level];
                    GpuBuffer sk = e.onField ? block0Buf : skipBufs[e.level];
                    runDecoderSkip(enc, up, hasSk ? sk : GpuBuffer(),
                                   tensorNamed("trans" + std::to_string(p.level) + ".scale"),
                                   bw*bh, C, out, hasSk);
                    state = out; liveC = C;
                } else if (p.channels * 2 == C) {
                    runGemm(enc, state, tensorNamed("vitin.up"), GpuBuffer{}, tokens, C, p.channels, 0, state);
                    liveC = C;
                } else if (p.channels == C * 2) {
                    runGemm(enc, state, tensorNamed("vitout.down"), GpuBuffer{}, tokens, C, p.channels, 1, state);
                    liveC = C;
                }
            }
            // keep live dims in sync
            liveW = bw; liveH = bh;

            if (bi == 0) { block0Pending_ = true; }
            runBlock(enc, e, state, ffnOut, projOut);
            if (bi == 0) { block0Buf = state; hasBlock0 = true; block0Pending_ = false; }
            // save encoder skips
            if (bi + 1 < sched.size()) {
                const BlockScheduleEntry nx = sched[bi+1];
                const int32_t effE2 = e.onField ? -1 : int32_t(e.level);
                const int32_t effN = nx.onField ? -1 : int32_t(nx.level);
                if (e.onField) { skipBufs[8] = state; hasSkip[8] = true; }
                else if (effN > effE2) { skipBufs[e.level] = state; hasSkip[e.level] = true; }
            }
        }

        // head
        {
            const uint32_t tokens = liveW * liveH;
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_head");
            [enc setComputePipelineState:p];
            [enc setBuffer:state.buf offset:0 atIndex:0];
            setTensor(enc, 1, "head.w"); setTensor(enc, 2, "head.b");
            setUint(enc, 3, tokens); setUint(enc, 4, liveC);
            [enc setBuffer:headBuf.buf offset:0 atIndex:5];
            dispatch1d(enc, p, tokens, 64);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(head, headBuf.buf.contents, size_t(FW) * FH * 4 * 4);
        return true;
    }

    // ---------------- video/game temporal ops ----------------
    bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) override {
        if (!ok_) return false;
        mv.width = currLuma.width; mv.height = currLuma.height;
        mv.xy.assign(size_t(mv.width) * mv.height * 2, 0.f);
        id<MTLTexture> prev = makeTexture(prevLuma);
        id<MTLTexture> curr = makeTexture(currLuma);
        id<MTLTexture> mvTex = newTextureRG32(mv.width, mv.height);
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_motion");
        [enc setComputePipelineState:p];
        [enc setTexture:prev atIndex:0]; [enc setTexture:curr atIndex:1];
        [enc setTexture:mvTex atIndex:2];
        dispatch(enc, p, MTLSizeMake(mv.width, mv.height, 1), MTLSizeMake(8, 8, 1));
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        // read back RG32
        {
            const size_t rowBytes = size_t(mv.width) * 8;
            std::vector<float> tmp(size_t(mv.width) * mv.height * 4);
            [mvTex getBytes:tmp.data() bytesPerRow:rowBytes
                 fromRegion:MTLRegionMake2D(0, 0, mv.width, mv.height) mipmapLevel:0];
            for (size_t i = 0; i < size_t(mv.width) * mv.height; ++i) {
                mv.xy[i*2+0] = tmp[i*4+0]; mv.xy[i*2+1] = tmp[i*4+1];
            }
        }
        return true;
    }

    bool reprojectHistory(const Image& historyColor, const MotionField& mv,
                          Image& reproj, Image& confidence) override {
        if (!ok_) return false;
        reproj = Image(mv.width, mv.height);
        confidence = Image(mv.width, mv.height);
        id<MTLTexture> mvTex = newTextureRG32(mv.width, mv.height);
        [mvTex replaceRegion:MTLRegionMake2D(0,0,mv.width,mv.height) mipmapLevel:0
                   withBytes:mv.xy.data() bytesPerRow:size_t(mv.width)*8];
        id<MTLTexture> hist = historyColor.width ? makeTexture(historyColor) : newTexture(0,0);
        id<MTLTexture> rp = newTexture(mv.width, mv.height);
        id<MTLTexture> cf = newTexture(mv.width, mv.height);
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_reproject");
        [enc setComputePipelineState:p];
        [enc setTexture:hist atIndex:0]; [enc setTexture:mvTex atIndex:1];
        [enc setTexture:rp atIndex:2];  [enc setTexture:cf atIndex:3];
        dispatch(enc, p, MTLSizeMake(mv.width, mv.height, 1), MTLSizeMake(16, 16, 1));
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        readTexture(rp, reproj); readTexture(cf, confidence);
        return true;
    }

    bool temporalBlend(const Image& current, const Image& reproj,
                       const Image& confidence, float maxBlend, Image& out) override {
        if (!ok_) return false;
        out = Image(current.width, current.height);
        id<MTLTexture> cur = makeTexture(current);
        id<MTLTexture> rp = makeTexture(reproj);
        id<MTLTexture> cf = makeTexture(confidence);
        id<MTLTexture> dst = newTexture(current.width, current.height);
        GpuBuffer mb = floatBuf({maxBlend});
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_temporal_blend");
        [enc setComputePipelineState:p];
        [enc setTexture:cur atIndex:0]; [enc setTexture:rp atIndex:1]; [enc setTexture:cf atIndex:2];
        [enc setBuffer:mb.buf offset:0 atIndex:0];
        [enc setTexture:dst atIndex:3];
        dispatch(enc, p, MTLSizeMake(current.width, current.height, 1), MTLSizeMake(16,16,1));
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        readTexture(dst, out);
        return true;
    }

    // ---------------- game mode: MTLFXTemporalScaler ----------------
    bool beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) override {
        gRenderW_ = renderW; gRenderH_ = renderH; gOutW_ = outputW; gOutH_ = outputH;
        tScaler_ = nil;
        if (@available(macOS 13.0, *)) {
            if (info_.metalFxTemporal) {
                MTLFXTemporalScalerDescriptor* d = [MTLFXTemporalScalerDescriptor new];
                d.inputWidth = renderW; d.inputHeight = renderH;
                d.outputWidth = outputW; d.outputHeight = outputH;
                d.colorTextureFormat = MTLPixelFormatRGBA16Float;
                d.depthTextureFormat = MTLPixelFormatR32Float;
                d.motionTextureFormat = MTLPixelFormatRG16Float;
                d.autoExposureEnabled = NO;   // we hand the scaler pre-exposure
                tScaler_ = [d newTemporalScalerWithDevice:device_];
            }
        }
        gColorTex_ = newTextureHalf(renderW, renderH);
        gDepthTex_ = newTextureR32(renderW, renderH);
        gMotionTex_ = newTextureRG16(renderW, renderH);
        gOutTex_ = newTextureHalf(outputW, outputH);
        gHasOut_ = false;
        return true;
    }

    bool submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) override {
        if (!ok_ || !in.color) return false;
        auto t0 = [NSDate date];
        // upload inputs
        uploadImageHalf(*in.color, gColorTex_);
        if (in.depth) {
            [gDepthTex_ replaceRegion:MTLRegionMake2D(0,0,gRenderW_,gRenderH_) mipmapLevel:0
                           withBytes:in.depth bytesPerRow:size_t(gRenderW_)*4];
        }
        if (in.motion) {
            std::vector<uint16_t> rg16(size_t(gRenderW_) * gRenderH_ * 2);
            for (size_t i = 0; i < size_t(gRenderW_) * gRenderH_; ++i) {
                rg16[i*2+0] = f32_to_f16(in.motion->xy[i*2+0]);
                rg16[i*2+1] = f32_to_f16(in.motion->xy[i*2+1]);
            }
            [gMotionTex_ replaceRegion:MTLRegionMake2D(0,0,gRenderW_,gRenderH_) mipmapLevel:0
                             withBytes:rg16.data() bytesPerRow:size_t(gRenderW_)*4];
        }
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (tScaler_) {
            tScaler_.colorTexture = gColorTex_;
            tScaler_.depthTexture = gDepthTex_;
            tScaler_.motionTexture = gMotionTex_;
            if (!tScaler_.outputTexture) tScaler_.outputTexture = gOutTex_;
            tScaler_.preExposure = in.exposure > 0 ? in.exposure : 1.0f;
            tScaler_.reset = in.resetHistory ? YES : NO;
            [tScaler_ encodeToCommandBuffer:cb];
        } else {
            // fallback: Lanczos upscale + our temporal blend at output res
            // (the CPU-shaped path; acceptable before MetalFX-capable hardware)
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLTexture> tmp = newTextureHalf(gOutW_, gRenderH_);
            GpuBuffer w = floatBuf({float(gOutW_)});
            id<MTLComputePipelineState> ph = pipeline(lib_, pipes_, device_, "odl_upscale_h");
            [enc setComputePipelineState:ph];
            [enc setTexture:gColorTex_ atIndex:0]; [enc setBuffer:w.buf offset:0 atIndex:0];
            [enc setTexture:tmp atIndex:1];
            dispatch(enc, ph, MTLSizeMake(gOutW_, gRenderH_, 1), MTLSizeMake(16,16,1));
            GpuBuffer h = floatBuf({float(gOutH_)});
            id<MTLComputePipelineState> pv = pipeline(lib_, pipes_, device_, "odl_upscale_v");
            [enc setComputePipelineState:pv];
            [enc setTexture:tmp atIndex:0]; [enc setBuffer:h.buf offset:0 atIndex:0];
            [enc setTexture:gOutTex_ atIndex:1];
            dispatch(enc, pv, MTLSizeMake(gOutW_, gOutH_, 1), MTLSizeMake(16,16,1));
            [enc endEncoding];
        }
        [cb commit]; [cb waitUntilCompleted];
        readTextureHalf(gOutTex_, out.upscaled);
        if (in.depth == nullptr) { /* depth optional for the fallback path */ }
        out.gpuMs = -[t0 timeIntervalSinceNow] * 1000.0;
        gHasOut_ = true;
        return true;
    }

    void endGame() override {
        tScaler_ = nil;
        gColorTex_ = nil; gDepthTex_ = nil; gMotionTex_ = nil; gOutTex_ = nil;
        gHasOut_ = false;
    }

private:
    // ---- helpers ----
    GpuBuffer makeBuffer(size_t bytes) {
        GpuBuffer gb; gb.bytes = bytes;
        gb.buf = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        return gb;
    }
    GpuBuffer floatBuf2(const float* v, size_t n) {
        GpuBuffer gb = makeBuffer(n * 4);
        std::memcpy(gb.buf.contents, v, n * 4);
        return gb;
    }
    GpuBuffer floatBuf(std::initializer_list<float> v) { return floatBuf2(v.begin(), v.size()); }
    GpuBuffer uintBuf(uint32_t v) {
        GpuBuffer gb = makeBuffer(4);
        *reinterpret_cast<uint32_t*>(gb.buf.contents) = v;
        return gb;
    }
    GpuBuffer uint2Buf(uint32_t x, uint32_t y) {
        GpuBuffer gb = makeBuffer(8);
        uint32_t* p = reinterpret_cast<uint32_t*>(gb.buf.contents);
        p[0] = x; p[1] = y;
        return gb;
    }

    id<MTLTexture> newTexture(uint32_t w, uint32_t h) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return [device_ newTextureWithDescriptor:d];
    }
    id<MTLTexture> newTextureHalf(uint32_t w, uint32_t h) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
        return [device_ newTextureWithDescriptor:d];
    }
    id<MTLTexture> newTextureR32(uint32_t w, uint32_t h) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        return [device_ newTextureWithDescriptor:d];
    }
    id<MTLTexture> newTextureRG16(uint32_t w, uint32_t h) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float
                                            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        return [device_ newTextureWithDescriptor:d];
    }
    id<MTLTexture> newTextureRG32(uint32_t w, uint32_t h) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float
                                            width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        return [device_ newTextureWithDescriptor:d];
    }

    id<MTLTexture> makeTexture(const Image& img) {
        id<MTLTexture> t = newTexture(img.width, img.height);
        [t replaceRegion:MTLRegionMake2D(0,0,img.width,img.height) mipmapLevel:0
               withBytes:img.pixels.data() bytesPerRow:size_t(img.width)*16];
        return t;
    }
    void readTexture(id<MTLTexture> t, Image& img) {
        [t getBytes:img.pixels.data() bytesPerRow:size_t(img.width)*16
          fromRegion:MTLRegionMake2D(0,0,img.width,img.height) mipmapLevel:0];
    }
    void uploadImageHalf(const Image& img, id<MTLTexture> t) {
        std::vector<uint16_t> h(img.pixels.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = f32_to_f16(img.pixels[i]);
        [t replaceRegion:MTLRegionMake2D(0,0,img.width,img.height) mipmapLevel:0
               withBytes:h.data() bytesPerRow:size_t(img.width)*8];
    }
    void readTextureHalf(id<MTLTexture> t, Image& img) {
        std::vector<uint16_t> h(size_t(img.width) * img.height * 4);
        [t getBytes:h.data() bytesPerRow:size_t(img.width)*8
          fromRegion:MTLRegionMake2D(0,0,img.width,img.height) mipmapLevel:0];
        for (size_t i = 0; i < h.size(); ++i) img.pixels[i] = f16_to_f32(h[i]);
    }

    void dispatch1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                    uint32_t n, uint32_t tg) {
        NSUInteger groups = (n + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    }
    void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                  MTLSize threads, MTLSize tg) {
        MTLSize groups = MTLSizeMake((threads.width + tg.width - 1)/tg.width,
                                     (threads.height + tg.height - 1)/tg.height,
                                     1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    }

    GpuBuffer tensorNamed(const std::string& name) {
        auto it = tensorBufs_.find(name);
        return it != tensorBufs_.end() ? it->second : GpuBuffer{};
    }
    void setTensor(id<MTLComputeCommandEncoder> enc, uint32_t idx, const std::string& name) {
        GpuBuffer gb = tensorNamed(name);
        if (gb.buf) [enc setBuffer:gb.buf offset:0 atIndex:idx];
    }
    void setUint(id<MTLComputeCommandEncoder> enc, uint32_t idx, uint32_t v) {
        [enc setBytes:&v length:4 atIndex:idx];
    }

    id<MTLFXSpatialScaler> spatialScaler(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh) {
        if (!info_.metalFxSpatial) return nil;
        const std::string key = std::to_string(iw)+"x"+std::to_string(ih)+"->"+std::to_string(ow)+"x"+std::to_string(oh);
        auto it = spatialScalers_.find(key);
        if (it != spatialScalers_.end()) return it->second;
        if (@available(macOS 13.0, *)) {
            MTLFXSpatialScalerDescriptor* d = [MTLFXSpatialScalerDescriptor new];
            d.inputWidth = iw; d.inputHeight = ih; d.outputWidth = ow; d.outputHeight = oh;
            d.colorTextureFormat = MTLPixelFormatRGBA16Float;
            d.outputTextureFormat = MTLPixelFormatRGBA16Float;
            id<MTLFXSpatialScaler> s = [d newSpatialScalerWithDevice:device_];
            spatialScalers_[key] = s;
            return s;
        }
        return nil;
    }

    // ---- graph sub-steps (single-use encoders) ----
    void runInputEmbed(id<MTLComputeCommandEncoder> enc, GpuBuffer& feat, GpuBuffer& state,
                       uint32_t fw, uint32_t fh, uint32_t C) {
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_input_embed");
        [enc setComputePipelineState:p];
        [enc setBuffer:feat.buf offset:0 atIndex:0];
        setTensor(enc, 1, "input_proj");
        [enc setBuffer:state.buf offset:0 atIndex:2];
        GpuBuffer fs = uint2Buf(fw, fh); [enc setBuffer:fs.buf offset:0 atIndex:3];
        GpuBuffer cb = uintBuf(C); [enc setBuffer:cb.buf offset:0 atIndex:4];
        dispatch1d(enc, p, fw * fh, 64);
    }
    void runPool(id<MTLComputeCommandEncoder> enc, GpuBuffer& src, uint32_t sw, uint32_t sh,
                 uint32_t dw, uint32_t dh, uint32_t C, GpuBuffer& dst) {
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_pool2x2");
        [enc setComputePipelineState:p];
        [enc setBuffer:src.buf offset:0 atIndex:0];
        GpuBuffer ss = uint2Buf(sw, sh); [enc setBuffer:ss.buf offset:0 atIndex:1];
        GpuBuffer ds = uint2Buf(dw, dh); [enc setBuffer:ds.buf offset:0 atIndex:2];
        GpuBuffer cb = uintBuf(C); [enc setBuffer:cb.buf offset:0 atIndex:3];
        [enc setBuffer:dst.buf offset:0 atIndex:4];
        dispatch1d(enc, p, dw * dh, 64);
    }
    void runGemm(id<MTLComputeCommandEncoder> enc, GpuBuffer& x, GpuBuffer w, GpuBuffer b,
                 uint32_t tokens, uint32_t outCh, uint32_t inCh, uint32_t pubMode, GpuBuffer& out) {
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_channel_gemm");
        [enc setComputePipelineState:p];
        [enc setBuffer:x.buf offset:0 atIndex:0];
        [enc setBuffer:w.buf offset:0 atIndex:1];
        if (b.buf) [enc setBuffer:b.buf offset:0 atIndex:2];
        GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:3];
        GpuBuffer ob = uintBuf(outCh);  [enc setBuffer:ob.buf offset:0 atIndex:4];
        GpuBuffer ib = uintBuf(inCh);   [enc setBuffer:ib.buf offset:0 atIndex:5];
        GpuBuffer pm = uintBuf(pubMode);[enc setBuffer:pm.buf offset:0 atIndex:6];
        [enc setBuffer:out.buf offset:0 atIndex:7];
        dispatch1d(enc, p, tokens * outCh, 64);
    }
    void runUpsample(id<MTLComputeCommandEncoder> enc, GpuBuffer& src, uint32_t sw, uint32_t sh,
                     uint32_t dw, uint32_t dh, uint32_t C, GpuBuffer& dst) {
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_nearest_upsample2x");
        [enc setComputePipelineState:p];
        [enc setBuffer:src.buf offset:0 atIndex:0];
        GpuBuffer ss = uint2Buf(sw, sh); [enc setBuffer:ss.buf offset:0 atIndex:1];
        GpuBuffer ds = uint2Buf(dw, dh); [enc setBuffer:ds.buf offset:0 atIndex:2];
        GpuBuffer cb = uintBuf(C); [enc setBuffer:cb.buf offset:0 atIndex:3];
        [enc setBuffer:dst.buf offset:0 atIndex:4];
        dispatch1d(enc, p, dw * dh * C, 64);
    }
    void runDecoderSkip(id<MTLComputeCommandEncoder> enc, GpuBuffer& up, GpuBuffer skip,
                        GpuBuffer scale, uint32_t tokens, uint32_t C, GpuBuffer& out, bool hasSkip) {
        id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_decoder_skip");
        [enc setComputePipelineState:p];
        [enc setBuffer:up.buf offset:0 atIndex:0];
        if (hasSkip && skip.buf) [enc setBuffer:skip.buf offset:0 atIndex:1];
        GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:3];
        GpuBuffer cb = uintBuf(C); [enc setBuffer:cb.buf offset:0 atIndex:4];
        [enc setBuffer:out.buf offset:0 atIndex:5];
        dispatch1d(enc, p, tokens * C, 64);
    }
    void runBlock(id<MTLComputeCommandEncoder> enc, const BlockScheduleEntry& e,
                  GpuBuffer& state, GpuBuffer& ffnOut, GpuBuffer& projOut) {
        const uint32_t tokens = e.levelWidth * e.levelHeight;
        const uint32_t C = e.channels;
        const std::string b = std::to_string(e.index);

        // 1. FFN -> ffnOut (raw ffn result)
        {
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_ffn");
            [enc setComputePipelineState:p];
            [enc setBuffer:state.buf offset:0 atIndex:0];
            setTensor(enc, 1, "block" + b + ".layer0.w1");
            setTensor(enc, 2, "block" + b + ".layer0.b1");
            setTensor(enc, 3, "block" + b + ".layer0.w2");
            setTensor(enc, 4, "block" + b + ".layer0.b2");
            setTensor(enc, 5, "block" + b + ".layer0.w3");
            setTensor(enc, 6, "block" + b + ".layer0.b3");
            GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:7];
            GpuBuffer cb = uintBuf(C); [enc setBuffer:cb.buf offset:0 atIndex:8];
            GpuBuffer hb = uintBuf(e.isViT ? 4096u : 128u); [enc setBuffer:hb.buf offset:0 atIndex:9];
            GpuBuffer pb = uintBuf(e.isViT ? 1u : (C == 32 ? 1u : C / 32u)); [enc setBuffer:pb.buf offset:0 atIndex:10];
            [enc setBuffer:ffnOut.buf offset:0 atIndex:11];
            // threadgroup scratch: hidden*paths + C*paths halfs
            uint32_t hidden = e.isViT ? 4096u : 128u;
            uint32_t paths = e.isViT ? 1u : (C == 32 ? 1u : C / 32u);
            [enc setThreadgroupMemoryLength:(size_t(hidden) * paths + C * paths) * 2 atIndex:0];
            dispatch1d(enc, p, tokens, 128);
        }
        // 2. y = x*ffnScale + ffnOut (published per block class) -> projIn
        {
            // done inline via odl_block_skip kernel? We reuse odl_decoder_skip with no skip:
            // y = pub(x*ffnScale + ffn). For clarity we compute it in a small dedicated kernel
            // "odl_block_skip" declared in Graph.metal.
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_block_skip");
            [enc setComputePipelineState:p];
            [enc setBuffer:state.buf offset:0 atIndex:0];
            [enc setBuffer:ffnOut.buf offset:0 atIndex:1];
            setTensor(enc, 2, "block" + b + ".layer0.scale_ffn");
            GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:3];
            GpuBuffer cb2 = uintBuf(C); [enc setBuffer:cb2.buf offset:0 atIndex:4];
            GpuBuffer raw = uintBuf(C == 32 && !e.isViT ? 1u : 0u); [enc setBuffer:raw.buf offset:0 atIndex:5];
            [enc setBuffer:projIn_.buf offset:0 atIndex:6];
            dispatch1d(enc, p, tokens * C, 64);
        }
        // 3. attention
        if (e.isViT) {
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_global_attention");
            [enc setComputePipelineState:p];
            [enc setBuffer:projIn_.buf offset:0 atIndex:0];
            setTensor(enc, 1, "block" + b + ".layer1.qscale");
            GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:2];
            GpuBuffer cb2 = uintBuf(C); [enc setBuffer:cb2.buf offset:0 atIndex:3];
            GpuBuffer hb = uintBuf(e.heads); [enc setBuffer:hb.buf offset:0 atIndex:4];
            [enc setBuffer:projOut.buf offset:0 atIndex:5];
            dispatch1d(enc, p, tokens, 32);
        } else {
            // QKV GEMM first (published E4M3)
            runGemm(enc, projIn_, tensorNamed("block" + b + ".layer1.qkv"), GpuBuffer{},
                    tokens, 3 * C, C, 0, qkvBuf_);
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_window_attention");
            [enc setComputePipelineState:p];
            [enc setBuffer:qkvBuf_.buf offset:0 atIndex:0];
            setTensor(enc, 1, "block" + b + ".layer1.prior");
            setTensor(enc, 2, "block" + b + ".layer1.qscale");
            GpuBuffer ls = uint2Buf(e.levelWidth, e.levelHeight); [enc setBuffer:ls.buf offset:0 atIndex:3];
            GpuBuffer cb2 = uintBuf(C); [enc setBuffer:cb2.buf offset:0 atIndex:4];
            GpuBuffer hb = uintBuf(e.heads); [enc setBuffer:hb.buf offset:0 atIndex:5];
            GpuBuffer ph = uintBuf(e.windowPhase); [enc setBuffer:ph.buf offset:0 atIndex:6];
            [enc setBuffer:projOut.buf offset:0 atIndex:7];
            // threadgroup memories, in the kernel's declaration order: qE, kE, vE, Oacc
            [enc setThreadgroupMemoryLength:size_t(e.heads) * 64 * 32 * 2 atIndex:0];
            [enc setThreadgroupMemoryLength:size_t(e.heads) * 64 * 32 * 2 atIndex:1];
            [enc setThreadgroupMemoryLength:size_t(e.heads) * 64 * 32 * 2 atIndex:2];
            [enc setThreadgroupMemoryLength:size_t(e.heads) * 32 * 4 atIndex:3];
            // window grid (matches the kernel's own computation)
            const int shift = (e.windowPhase == 0) ? 0 : 4;
            const uint32_t gridX = (e.levelWidth + shift + 7) / 8;
            const uint32_t gridY = (e.levelHeight + shift + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(gridX * gridY, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        // 4. epilogue: out = pub(y*attnScale + projOut)
        {
            id<MTLComputePipelineState> p = pipeline(lib_, pipes_, device_, "odl_block_epilogue");
            [enc setComputePipelineState:p];
            [enc setBuffer:projIn_.buf offset:0 atIndex:0];
            [enc setBuffer:projOut.buf offset:0 atIndex:1];
            setTensor(enc, 2, "block" + b + ".layer0.scale_attn");
            GpuBuffer tb = uintBuf(tokens); [enc setBuffer:tb.buf offset:0 atIndex:3];
            GpuBuffer cb2 = uintBuf(C); [enc setBuffer:cb2.buf offset:0 atIndex:4];
            [enc setBuffer:state.buf offset:0 atIndex:5];
            dispatch1d(enc, p, tokens * C, 64);
        }
    }

    BackendInfo info_;
    bool ok_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> lib_ = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipes_;
    std::map<std::string, id<MTLFXSpatialScaler>> spatialScalers_;
    id<MTLFXTemporalScaler> tScaler_ = nil;
    std::unique_ptr<Model> model_;
    std::map<std::string, GpuBuffer> tensorBufs_;
    GpuBuffer projIn_, qkvBuf_;
    bool block0Pending_ = false;

    // game state
    uint32_t gRenderW_ = 0, gRenderH_ = 0, gOutW_ = 0, gOutH_ = 0;
    id<MTLTexture> gColorTex_ = nil, gDepthTex_ = nil, gMotionTex_ = nil, gOutTex_ = nil;
    bool gHasOut_ = false;
};

} // namespace opendlss

namespace opendlss {
std::unique_ptr<IBackend> create_metal_backend() {
    auto b = std::make_unique<MetalBackend>();
    // If the backend failed to initialize (no Metal device), fall back to CPU.
    return b;
}
} // namespace opendlss
