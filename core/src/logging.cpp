// Logging implementation.
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#include "opendlss/logging.h"

#include <cstdio>
#include <cstdarg>
#include <atomic>
#include <ctime>

namespace opendlss {

namespace {
std::atomic<int> g_level{int(LogLevel::Info)};
std::mutex g_mutex;

void vlog(LogLevel level, const char* fmt, va_list args) {
    if (int(level) < g_level.load()) return;
    char stack_buf[1024];
    va_list copy;
    va_copy(copy, args);
    int n = std::vsnprintf(stack_buf, sizeof(stack_buf), fmt, args);
    std::string line;
    if (n >= int(sizeof(stack_buf))) {
        line.resize(size_t(n) + 1);
        std::vsnprintf(line.data(), line.size(), fmt, copy);
        line.resize(size_t(n));
    } else if (n < 0) {
        line = "<log format error>";
    } else {
        line.assign(stack_buf, size_t(n));
    }
    va_end(copy);

    const char* tag = "I";
    if (level == LogLevel::Debug) tag = "D";
    else if (level == LogLevel::Warn) tag = "W";
    else if (level == LogLevel::Error) tag = "E";

    std::lock_guard<std::mutex> lock(g_mutex);
    std::FILE* out = level >= LogLevel::Warn ? stderr : stdout;
    std::fprintf(out, "[%s] %s\n", tag, line.c_str());
}

} // namespace

void log_set_level(LogLevel level) { g_level.store(int(level)); }
LogLevel log_get_level() { return LogLevel(g_level.load()); }

LogLevel log_level_from_string(const std::string& s) {
    if (s == "debug") return LogLevel::Debug;
    if (s == "warn" || s == "warning") return LogLevel::Warn;
    if (s == "error") return LogLevel::Error;
    return LogLevel::Info;
}

void log_info(const char* fmt, ...)  { va_list a; va_start(a, fmt); vlog(LogLevel::Info,  fmt, a); va_end(a); }
void log_warn(const char* fmt, ...)  { va_list a; va_start(a, fmt); vlog(LogLevel::Warn,  fmt, a); va_end(a); }
void log_error(const char* fmt, ...) { va_list a; va_start(a, fmt); vlog(LogLevel::Error, fmt, a); va_end(a); }
void log_debug(const char* fmt, ...) { va_list a; va_start(a, fmt); vlog(LogLevel::Debug, fmt, a); va_end(a); }

} // namespace opendlss
