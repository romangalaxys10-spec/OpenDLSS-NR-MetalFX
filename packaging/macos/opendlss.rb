# Homebrew formula: brew install ./opendlss.rb  (builds from source)
class Opendlss < Formula
  desc "Neural rendering (DLSS 5 NR architecture) with MetalFX - macOS CLI + SDK"
  homepage "https://github.com/maanHimself/OpenDLSS-NR-MetalFX"
  url "https://github.com/maanHimself/OpenDLSS-NR-MetalFX/archive/refs/tags/v1.0.0.tar.gz"
  version "1.0.0"
  license "MIT"
  depends_on xcode: :build
  depends_on arch: :arm64

  def install
    system "cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_OSX_ARCHITECTURES=arm64", *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
    (share/"opendlss").install Dir["build/opendlss.metallib"]
  end

  test do
    system "#{bin}/opendlss", "selftest"
  end
end
