// Backend contract. One interface, four implementations:
//   Cpu    — reference numerics, runs everywhere (CI, fallback, parity testing)
//   Metal  — macOS Apple Silicon, MetalFX-accelerated (the flagship path)
//   D3D12  — Windows, DirectML-accelerated GEMMs + HLSL compute
//   Vulkan — Linux, compute shaders (port of the reference's execution model)
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include "opendlss/geometry.h"
#include "opendlss/image.h"
#include "opendlss/model.h"

#include <memory>
#include <string>
#include <vector>

namespace opendlss {

enum class BackendKind { Cpu, Metal, Vulkan, D3D12 };

struct BackendInfo {
    BackendKind kind = BackendKind::Cpu;
    std::string name;          // "cpu", "metal+metalfx", "vulkan", "d3d12+directml"
    std::string deviceName;    // adapter / GPU name
    bool metalFxSpatial  = false;   // MTLFXSpatialScaler available
    bool metalFxTemporal = false;   // MTLFXTemporalScaler available
    bool neuralGraph     = false;   // can execute the Swin/ViT graph
    bool fp16Storage     = false;   // native half arithmetic
    bool e4m3Quant       = false;   // publishes through the E4M3 grid
    uint32_t maxTextureDim = 16384;
    std::string details;
};

// Motion vectors from block matching / engine input: two floats per pixel (dx, dy in pixels).
struct MotionField {
    uint32_t width = 0, height = 0;
    std::vector<float> xy;   // [y*width + x] * 2
};

// Frame submission for game (real-time) mode — mirrors what engines feed
// DLSS-style temporal scalers.
struct GameFrameInput {
    const Image*  color = nullptr;       // rendered frame (render resolution)
    const float*  depth  = nullptr;      // linear depth, [0..1], width*height
    const MotionField* motion = nullptr; // per-pixel motion, prev->curr, render resolution
    float exposure = 1.0f;               // pre-exposure multiplier applied by the engine
    float jitterX = 0.0f, jitterY = 0.0f;// sub-pixel jitter used when rendering
    bool  resetHistory = false;          // camera cut / first frame
};

struct GameFrameOutput {
    Image upscaled;       // output-resolution RGBA
    double gpuMs = 0.0;   // execution time (wall clock on CPU backend)
};

// Tunables shared by every mode (CLI exposes these 1:1).
struct PipelineOptions {
    float denoiseStrength = 0.6f;   // analytical bilateral (0 = off)
    float sharpen = 0.25f;          // adaptive sharpen amount (0 = off)
    float style = 0.0f;             // conditioning lane 10: style id / 128
    float tone = 0.5f;              // conditioning lane 11: local tone
    float structure = 0.5f;         // lane 12
    float skin   = 0.5f;            // lane 13
    float autoMask = 0.5f;          // lane 14
    float temporalMaxBlend = 0.85f; // cap on history weight in video/game blend
    uint32_t upscaleFactor = 2;     // 1, 2, 3, 4
    bool useNeural = true;          // run the graph when a model is loaded
    bool useMetalFx = true;         // prefer MetalFX scalers when available
    uint64_t seed = 0x5EED1234ull;  // per-run noise seed base
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual const BackendInfo& info() const = 0;

    // ---------- image mode ----------
    virtual bool denoiseSpatial(const Image& input, Image& output, float strength) = 0;
    virtual bool upscaleSpatial(const Image& input, Image& output, uint32_t factor, float sharpen) = 0;

    // ---------- neural graph ----------
    virtual bool loadModel(const std::string& modelDir) = 0;
    virtual bool hasNeuralGraph() const = 0;
    virtual const Model* model() const = 0;
    // features: [fieldH * fieldW][16] f16 bits; head: [fieldH * fieldW][4] f32.
    virtual bool runNeuralGraph(const Geometry& geom, const uint16_t* features,
                                float* head, uint64_t frameSeed) = 0;

    // ---------- video / game temporal ops ----------
    virtual bool estimateMotion(const Image& prevLuma, const Image& currLuma, MotionField& mv) = 0;
    virtual bool reprojectHistory(const Image& historyColor, const MotionField& mv,
                                  Image& reproj, Image& confidence) = 0;
    // blend = lerp(current, reproj, min(confidence, maxBlend))
    virtual bool temporalBlend(const Image& current, const Image& reproj,
                               const Image& confidence, float maxBlend, Image& out) = 0;

    // ---------- game (real-time) mode ----------
    // On Metal this drives MTLFXTemporalScaler; elsewhere our reprojection
    // scaler with the same contract.
    virtual bool beginGame(uint32_t renderW, uint32_t renderH, uint32_t outputW, uint32_t outputH) = 0;
    virtual bool submitGameFrame(const GameFrameInput& in, GameFrameOutput& out) = 0;
    virtual void endGame() = 0;
};

// Registry: enumerate and create the backends compiled into this build.
std::vector<BackendKind> available_backends();
const char* backend_kind_name(BackendKind kind);
std::unique_ptr<IBackend> create_backend(BackendKind kind);
// Preferred = Metal on macOS, D3D12 on Windows, Vulkan on Linux, Cpu always last.
std::unique_ptr<IBackend> create_preferred_backend();

// Platform factories (each platform backend defines the ones it builds;
// guarded by OPENDLSS_HAS_METAL / OPENDLSS_HAS_VULKAN / OPENDLSS_HAS_D3D12).
std::unique_ptr<IBackend> create_metal_backend();
std::unique_ptr<IBackend> create_vulkan_backend();
std::unique_ptr<IBackend> create_d3d12_backend();

} // namespace opendlss
