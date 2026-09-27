// Minimal logging (no dependencies; thread-safe via a mutex).
// Copyright (c) 2026 OpenDLSS-NR MetalFX contributors. MIT License.
#pragma once

#include <string>
#include <mutex>

namespace opendlss {

enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

void log_set_level(LogLevel level);
LogLevel log_get_level();
LogLevel log_level_from_string(const std::string& s);

// printf-style; implemented in logging.cpp.
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);
void log_debug(const char* fmt, ...);

} // namespace opendlss
