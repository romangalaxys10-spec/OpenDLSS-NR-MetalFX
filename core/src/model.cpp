// Model loading: manifest parse, E4M3 stage decode, tensor relayout.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/model.h"
#include "opendlss/e4m3.h"
#include "opendlss/fp16.h"
#include "opendlss/logging.h"

#include <fstream>
#include <sstream>
#include <functional>
#include <algorithm>

namespace opendlss {
namespace json {

namespace {

void skipWs(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

struct Obj;
struct Arr;
struct Str;
struct Num;
struct Bool;
struct Nul;

struct Obj : Value {
    std::map<std::string, std::unique_ptr<Value>> members;
    bool isObject() const override { return true; }
    const Value* get(const std::string& k) const override {
        auto it = members.find(k);
        return it == members.end() ? nullptr : it->second.get();
    }
    size_t size() const override { return members.size(); }
};
struct Arr : Value {
    std::vector<std::unique_ptr<Value>> items;
    bool isArray() const override { return true; }
    const Value* at(size_t i) const override { return i < items.size() ? items[i].get() : nullptr; }
    size_t size() const override { return items.size(); }
};
struct Str : Value {
    std::string v;
    bool isString() const override { return true; }
    std::string asString(const std::string& = {}) const override { return v; }
};
struct Num : Value {
    double v = 0;
    bool isNumber() const override { return true; }
    double asNumber(double = 0) const override { return v; }
    bool asBool(bool = false) const override { return v != 0; }
};
struct BoolV : Value {
    bool v = false;
    bool isBool() const override { return true; }
    bool asBool(bool = false) const override { return v; }
};
struct Nul : Value {};

std::unique_ptr<Value> parseValue(const std::string& s, size_t& i) {
    skipWs(s, i);
    if (i >= s.size()) return nullptr;
    char c = s[i];
    if (c == '{') {
        auto o = std::make_unique<Obj>();
        ++i; skipWs(s, i);
        if (i < s.size() && s[i] == '}') { ++i; return o; }
        while (true) {
            skipWs(s, i);
            auto key = parseValue(s, i);
            if (!key) return nullptr;
            skipWs(s, i);
            if (i >= s.size() || s[i] != ':') return nullptr;
            ++i;
            auto val = parseValue(s, i);
            if (!val) return nullptr;
            o->members[key->asString()] = std::move(val);
            skipWs(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; return o; }
            return nullptr;
        }
    }
    if (c == '[') {
        auto a = std::make_unique<Arr>();
        ++i; skipWs(s, i);
        if (i < s.size() && s[i] == ']') { ++i; return a; }
        while (true) {
            auto val = parseValue(s, i);
            if (!val) return nullptr;
            a->items.push_back(std::move(val));
            skipWs(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == ']') { ++i; return a; }
            return nullptr;
        }
    }
    if (c == '"') {
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break;
                    case 'r': out += '\r'; break; case '"': out += '"'; break;
                    case '\\': out += '\\'; break; case '/': out += '/'; break;
                    default: out += s[i];
                }
            } else out += s[i];
            ++i;
        }
        if (i >= s.size()) return nullptr;
        ++i;
        auto v = std::make_unique<Str>();
        v->v = std::move(out);
        return v;
    }
    if (c == 't' && s.compare(i, 4, "true") == 0) { i += 4; auto v = std::make_unique<BoolV>(); v->v = true; return v; }
    if (c == 'f' && s.compare(i, 5, "false") == 0) { i += 5; auto v = std::make_unique<BoolV>(); v->v = false; return v; }
    if (c == 'n' && s.compare(i, 4, "null") == 0) { i += 4; return std::make_unique<Nul>(); }
    // number
    {
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) ++i;
        if (start == i) return nullptr;
        auto v = std::make_unique<Num>();
        v->v = std::strtod(s.c_str() + start, nullptr);
        return v;
    }
}

} // namespace

std::unique_ptr<Value> parse(const std::string& text) {
    size_t i = 0;
    auto v = parseValue(text, i);
    if (!v) return nullptr;
    skipWs(text, i);
    return v;
}

} // namespace json

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

std::unique_ptr<Model> Model::load(const std::string& modelDir) {
    auto m = std::unique_ptr<Model>(new Model());
    m->dir_ = modelDir;
    if (!m->loadManifest(modelDir + "/manifest.json")) return nullptr;
    if (!m->loadStages()) return nullptr;
    log_info("model: loaded '%s' (%zu tensors, %zu stages, %u blocks)",
             modelDir.c_str(), m->tensors_.size(), m->stages_.size(), unsigned(m->schedule_.size()));
    return m;
}

bool Model::loadManifest(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { log_error("model: cannot open %s", path.c_str()); return false; }
    std::stringstream ss; ss << in.rdbuf();
    auto root = json::parse(ss.str());
    if (!root || !root->isObject()) { log_error("model: manifest is not a JSON object"); return false; }

    if (const auto* cfg = root->get("config")) {
        if (const auto* v = cfg->get("levels"))    config_.levels    = uint32_t(v->asNumber(6));
        if (const auto* v = cfg->get("minField"))  config_.minField  = uint32_t(v->asNumber(320));
        if (const auto* v = cfg->get("windowSize")) config_.windowSize = uint32_t(v->asNumber(8));
    }

    // Block schedule: embedded, else derived from config.
    if (const auto* sched = root->get("schedule")) {
        for (size_t i = 0; i < sched->size(); ++i) {
            const auto* e = sched->at(i);
            if (!e) return false;
            BlockScheduleEntry b;
            b.index = uint32_t(e->get("index")->asNumber(i));
            b.level = uint32_t(e->get("level")->asNumber(0));
            b.channels = uint32_t(e->get("channels")->asNumber(32));
            b.heads = uint32_t(e->get("heads")->asNumber(b.channels / 32));
            b.isViT = e->get("vit")->asBool(false);
            b.windowPhase = uint32_t(e->get("phase")->asNumber(0));
            if (const auto* of = e->get("onField")) b.onField = of->asBool(false);
            schedule_.push_back(b);
        }
    } else {
        schedule_ = make_reference_schedule();
        if (config_.levels != 6) {
            schedule_.clear();
            // Data-driven small schedule: 2 blocks per level + 1 ViT block at the last level.
            uint32_t idx = 0;
            for (uint32_t lvl = 0; lvl < config_.levels; ++lvl) {
                for (int k = 0; k < 2; ++k) {
                    schedule_.push_back({idx++, lvl, 32, 1, false, uint32_t((idx) % 4)});
                }
            }
            schedule_.push_back({idx++, config_.levels - 1, 32, 1, true, 0});
        }
    }

    if (const auto* st = root->get("stages")) {
        for (size_t i = 0; i < st->size(); ++i) {
            const auto* e = st->at(i);
            StageRecord r;
            r.id = e->get("id")->asString();
            r.file = e->get("file")->asString();
            r.packedByteLength = uint64_t(e->get("packedByteLength")->asNumber(0));
            r.sha256 = e->get("sha256")->asString();
            stages_.push_back(std::move(r));
        }
    }

    if (const auto* ts = root->get("tensors")) {
        for (size_t i = 0; i < ts->size(); ++i) {
            const auto* e = ts->at(i);
            TensorRecord t;
            t.name = e->get("name")->asString();
            t.block = uint32_t(e->get("block")->asNumber(0));
            t.layer = uint32_t(e->get("layer")->asNumber(0));
            t.parameter = e->get("parameter")->asString();
            t.stage = e->get("stage")->asString();
            t.stageOffset = uint64_t(e->get("stageOffset")->asNumber(0));
            t.byteLength = uint64_t(e->get("byteLength")->asNumber(0));
            if (const auto* sh = e->get("shape")) {
                for (size_t k = 0; k < sh->size(); ++k) t.shape.push_back(uint32_t(sh->at(k)->asNumber(0)));
            }
            tensors_.push_back(std::move(t));
        }
    }
    return true;
}

bool Model::loadStages() {
    stageBytes_.resize(stages_.size());
    for (size_t si = 0; si < stages_.size(); ++si) {
        const auto& st = stages_[si];
        std::ifstream in(dir_ + "/" + st.file, std::ios::binary);
        if (!in) { log_error("model: cannot open stage file %s", st.file.c_str()); return false; }
        std::vector<uint8_t>& buf = stageBytes_[si];
        in.seekg(0, std::ios::end);
        size_t sz = size_t(in.tellg());
        in.seekg(0);
        buf.resize(sz);
        in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(sz));
        if (st.packedByteLength && sz < st.packedByteLength) {
            log_error("model: stage '%s' short read (%zu < %llu)", st.id.c_str(), sz,
                      (unsigned long long)st.packedByteLength);
            return false;
        }
        // TODO(quality-gate): verify sha256 when a hashing backend is wired into CI.
    }
    // Decode every tensor E4M3 -> f16 (host layout).
    for (auto& t : tensors_) {
        const std::vector<uint8_t>* bytes = nullptr;
        for (size_t si = 0; si < stages_.size(); ++si)
            if (stages_[si].id == t.stage) bytes = &stageBytes_[si];
        if (!bytes) { log_error("model: tensor '%s' references unknown stage '%s'", t.name.c_str(), t.stage.c_str()); return false; }
        if (t.stageOffset + t.byteLength > bytes->size()) {
            log_error("model: tensor '%s' out of stage bounds", t.name.c_str());
            return false;
        }
        size_t count = t.byteLength;   // 1 byte per E4M3 element
        t.f16.resize(count);
        for (size_t i = 0; i < count; ++i)
            t.f16[i] = opendlss::f32_to_f16(E4M3::decode((*bytes)[t.stageOffset + i]));
        byName_[t.name] = size_t(&t - tensors_.data());
    }
    // blend scale
    if (const TensorRecord* bs = tensor("block70.layer0.blend_scale")) {
        if (!bs->f16.empty()) blendScale_ = f16_to_f32(bs->f16[0]);
    } else {
        // Search any tensor with parameter == "blend_scale".
        for (auto& t : tensors_)
            if (t.parameter == "blend_scale" && !t.f16.empty()) { blendScale_ = f16_to_f32(t.f16[0]); break; }
    }
    return true;
}

const TensorRecord* Model::tensor(const std::string& name) const {
    auto it = byName_.find(name);
    return it == byName_.end() ? nullptr : &tensors_[it->second];
}

const TensorRecord* Model::tensor(uint32_t block, uint32_t layer, const std::string& param) const {
    return tensor("block" + std::to_string(block) + ".layer" + std::to_string(layer) + "." + param);
}

float Model::blendScale() const { return blendScale_; }

std::vector<BlockScheduleEntry> Model::make_reference_schedule() {
    // Our reconstruction of the reference topology: 71 blocks, 6 levels.
    // Encoder 0..30, ViT 31..38, decoder 39..70; transitions documented at
    // 30->31 (pool + 512->1024) and 38->39 (1024->512 + upsample).
    // Per-level encoder counts chosen so level 2 has six blocks (documented).
    struct LevSpec { uint32_t count; uint32_t channels; };
    const LevSpec enc[6] = {{5, 32}, {5, 64}, {6, 128}, {6, 256}, {4, 512}, {4, 512}};
    const LevSpec dec[6] = {{4, 512}, {4, 512}, {6, 256}, {6, 128}, {5, 64}, {5, 32}};
    std::vector<BlockScheduleEntry> out;
    uint32_t idx = 0, phaseCounter = 0;
    for (uint32_t lvl = 0; lvl < 6; ++lvl) {
        for (uint32_t k = 0; k < enc[lvl].count; ++k) {
            BlockScheduleEntry b;
            b.index = idx++; b.level = lvl; b.channels = enc[lvl].channels;
            b.heads = b.channels / 32; b.windowPhase = phaseCounter++ % 4;
            out.push_back(b);
        }
    }
    for (uint32_t k = 0; k < 8; ++k) {   // ViT
        BlockScheduleEntry b;
        b.index = idx++; b.level = 5; b.channels = 1024; b.heads = 32; b.isViT = true;
        out.push_back(b);
    }
    phaseCounter = 2;   // decoder continues its encoder's count (documented behavior)
    for (uint32_t lvl = 5; ; --lvl) {
        for (uint32_t k = 0; k < dec[lvl].count; ++k) {
            BlockScheduleEntry b;
            b.index = idx++; b.level = lvl; b.channels = dec[lvl].channels;
            b.heads = b.channels / 32; b.windowPhase = phaseCounter++ % 4;
            out.push_back(b);
        }
        if (lvl == 0) break;
    }
    return out;
}

} // namespace opendlss
