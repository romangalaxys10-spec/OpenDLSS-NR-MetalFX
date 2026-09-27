// Model manifest and weight loading.
//
// A model directory holds:
//   manifest.json   — { "config": {...}, "stages": [...], "tensors": [...] }
//   <stage files>   — packed E4M3 weights, exactly like the reference layout:
//                     each stage entry has id, file, packedByteLength, sha256;
//                     each tensor entry has name, block, layer, parameter,
//                     stage, stageOffset, byteLength.
//
// The host decodes E4M3 -> f16 (or f32) once at load and re-lays the flat
// bytes out into the tensor shapes the kernels consume. Layouts are
// documented in docs/WEIGHTS.md. This layout is byte-compatible with the
// reference repository's model directories, so a full-size original manifest
// loads unmodified (the block schedule comes from config).
//
// tools/make_demo_weights.py writes a small config the demo runs out of the box.
//
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace opendlss {

// The network configuration a manifest declares (config section).
struct ModelConfig {
    uint32_t levels = 6;              // pooling levels (reference: 6)
    uint32_t minField = 320;          // geometry floor (reference: 320)
    uint32_t windowSize = 8;          // attention window
    std::string blendScaleTensor;     // block70.layer0.blend_scale equivalent
};

enum class ParamKind { Weight, Bias, Scale, Prior, BlendScale };

struct TensorRecord {
    std::string name;      // e.g. "block12.layer1.qkv.weight"
    uint32_t    block = 0;
    uint32_t    layer = 0;
    std::string parameter; // "w1","b1","w2","b2","w3","b3","qkv","proj","prior","scale","blend_scale"
    std::string stage;     // stage id
    uint64_t    stageOffset = 0;
    uint64_t    byteLength = 0;
    // filled at load:
    std::vector<uint16_t> f16;             // decoded values (host layout, row-major)
    std::vector<uint32_t> shape;           // logical shape [out, in] or [heads,64,64] etc.
};

struct StageRecord {
    std::string id;
    std::string file;
    uint64_t    packedByteLength = 0;
    std::string sha256;
};

struct BlockScheduleEntry {
    uint32_t index = 0;        // block id 0..N-1
    uint32_t level = 0;        // resolution level the block runs at
    uint32_t channels = 32;    // token width
    uint32_t heads = 1;        // channels / 32
    bool     isViT = false;    // global attention block
    uint32_t windowPhase = 0;  // 0..3, the shifted-window origin cycle
    bool     onField = false;  // runs on the padded field itself (blocks 0/70 class)
    // filled by the executor at run time:
    uint32_t levelWidth = 0, levelHeight = 0;
};

class Model {
public:
    static std::unique_ptr<Model> load(const std::string& modelDir);

    const ModelConfig& config() const { return config_; }
    const std::vector<BlockScheduleEntry>& schedule() const { return schedule_; }
    const std::vector<StageRecord>& stages() const { return stages_; }
    const std::vector<TensorRecord>& tensors() const { return tensors_; }

    // Tensor access by name; nullptr when absent.
    const TensorRecord* tensor(const std::string& name) const;
    // Convenience: named "block<b>.layer<l>.<param>".
    const TensorRecord* tensor(uint32_t block, uint32_t layer, const std::string& param) const;

    float blendScale() const;   // head temporal blend scale (learned f16)

    // Build the default block schedule matching the reference topology when
    // config doesn't embed one: 71 blocks over 6 levels with transitions at
    // 30->31 (into ViT) and 38->39 (out), widening 32->64->128->256->512->1024.
    static std::vector<BlockScheduleEntry> make_reference_schedule();

    std::string dir() const { return dir_; }

private:
    Model() = default;
    bool loadManifest(const std::string& path);
    bool loadStages();

    std::string dir_;
    ModelConfig config_;
    std::vector<BlockScheduleEntry> schedule_;
    std::vector<StageRecord> stages_;
    std::vector<TensorRecord> tensors_;
    std::map<std::string, size_t> byName_;
    std::vector<std::vector<uint8_t>> stageBytes_;
    float blendScale_ = 1.0f;
};

// Minimal JSON reader (no dependency; only what manifests need).
namespace json {
struct Value;
std::unique_ptr<Value> parse(const std::string& text);
struct Value {
    virtual ~Value() = default;
    virtual bool isObject() const { return false; }
    virtual bool isArray()  const { return false; }
    virtual bool isString() const { return false; }
    virtual bool isNumber() const { return false; }
    virtual bool isBool()   const { return false; }
    virtual const Value* get(const std::string& /*key*/) const { return nullptr; }
    virtual const Value* at(size_t /*i*/) const { return nullptr; }
    virtual size_t size() const { return 0; }
    virtual std::string asString(const std::string& def = {}) const { return def; }
    virtual double asNumber(double def = 0) const { return def; }
    virtual bool asBool(bool def = false) const { return def; }
};
} // namespace json

} // namespace opendlss
