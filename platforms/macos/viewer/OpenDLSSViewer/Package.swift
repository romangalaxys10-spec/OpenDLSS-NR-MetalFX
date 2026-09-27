// swift-tools-version:5.9
// OpenDLSSViewer — MetalFX temporal scaler demo viewer (Apple Silicon, macOS 13+)
import PackageDescription

let package = Package(
    name: "OpenDLSSViewer",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "OpenDLSSViewer",
            path: "."
        )
    ]
)
