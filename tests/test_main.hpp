// test_main.cpp — minimal test harness (no external dependency).
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace testing {
struct Case {
  std::string name;
  std::function<void()> fn;
};
inline std::vector<Case>& cases() {
  static std::vector<Case> value;
  return value;
}
inline int& failures() {
  static int value = 0;
  return value;
}
inline void run(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
inline int main() {
  for (const Case& c : cases()) {
    int before = failures();
    c.fn();
    printf("[%s] %s\n", before == failures() ? " ok " : "FAIL", c.name.c_str());
  }
  printf("%s: %d failure(s) over %zu checks\n", failures() == 0 ? "PASS" : "FAILED", failures(), cases().size());
  return failures() == 0 ? 0 : 1;
}
}  // namespace testing

#define NR_TEST(name)                                                                        \
  static void name##_impl();                                                                 \
  static const bool name##_registered = (testing::run(#name, name##_impl), true);            \
  static void name##_impl()

#define NR_CHECK(cond)                                                                       \
  do {                                                                                       \
    if (!(cond)) {                                                                           \
      fprintf(stderr, "  check failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__);            \
      ++testing::failures();                                                                 \
    }                                                                                        \
  } while (0)


