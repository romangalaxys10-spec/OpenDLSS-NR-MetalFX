// A crash report on Windows: an unhandled exception prints the exception code, the faulting address and the
// call stack (module + offset, symbolized when the PDB is next to the executable) to stderr and to
// dlss5-demo-crash.txt next to the executable, so that a crash in the field can be reported with something.
#include "crash_report.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>

#include <cstdio>
#include <string>

#pragma comment(lib, "dbghelp.lib")

namespace {

std::string g_reportPath;

LONG WINAPI onCrash(EXCEPTION_POINTERS* info) {
  const DWORD code = info->ExceptionRecord->ExceptionCode;
  FILE* files[2] = {stderr, fopen(g_reportPath.c_str(), "w")};
  HANDLE process = GetCurrentProcess();
  SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
  SymInitialize(process, nullptr, TRUE);
  void* frames[64];
  const USHORT count = CaptureStackBackTrace(0, 64, frames, nullptr);
  for (FILE* f : files) {
    if (!f) continue;
    fprintf(f, "[crash] exception 0x%08lX at %p\n", code, info->ExceptionRecord->ExceptionAddress);
    if (code == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2)
      fprintf(f, "[crash] access violation: %s address %p\n", info->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
              (void*)info->ExceptionRecord->ExceptionInformation[1]);
    // the faulting frame first, then the stack of this (the crashing) thread
    void* list[65] = {info->ExceptionRecord->ExceptionAddress};
    for (USHORT i = 0; i < count; ++i) list[i + 1] = frames[i];
    for (USHORT i = 0; i < count + 1; ++i) {
      char buffer[sizeof(SYMBOL_INFO) + 256] = {};
      SYMBOL_INFO* symbol = (SYMBOL_INFO*)buffer;
      symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol->MaxNameLen = 255;
      DWORD64 displacement = 0;
      HMODULE module = nullptr;
      char moduleName[MAX_PATH] = "?";
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)list[i], &module)) {
        GetModuleFileNameA(module, moduleName, MAX_PATH);
        const char* slash = strrchr(moduleName, '\\');
        if (slash) memmove(moduleName, slash + 1, strlen(slash + 1) + 1);
      }
      const uintptr_t offset = module ? (uintptr_t)list[i] - (uintptr_t)module : (uintptr_t)list[i];
      IMAGEHLP_LINE64 line = {};
      line.SizeOfStruct = sizeof(line);
      DWORD lineDisplacement = 0;
      const bool hasLine = SymGetLineFromAddr64(process, (DWORD64)list[i], &lineDisplacement, &line) != FALSE;
      if (SymFromAddr(process, (DWORD64)list[i], &displacement, symbol))
        fprintf(f, "[crash] %2u %s+0x%zx %s+0x%llx%s%s%s%lu\n", i, moduleName, (size_t)offset, symbol->Name, (unsigned long long)displacement,
                hasLine ? " (" : "", hasLine ? line.FileName : "", hasLine ? ":" : "", hasLine ? line.LineNumber : 0ul);
      else
        fprintf(f, "[crash] %2u %s+0x%zx\n", i, moduleName, (size_t)offset);
    }
    if (f != stderr) fclose(f);
  }
  fprintf(stderr, "[crash] report written to %s\n", g_reportPath.c_str());
  return EXCEPTION_EXECUTE_HANDLER;
}

}  // namespace

void installCrashReport(const std::string& directory) {
  g_reportPath = directory + "dlss5-demo-crash.txt";
  SetUnhandledExceptionFilter(onCrash);
}

#else
void installCrashReport(const std::string&) {}
#endif
