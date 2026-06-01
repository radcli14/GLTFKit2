// swift-tools-version: 5.9
// Converted from binary xcframework distribution to a source-based package to
// allow adding GLB/GLTF export support.
//
// Two-target design (required by SwiftPM — mixed Swift+ObjC in one target is
// Xcode-only and not supported by swift build):
//   GLTFKit2ObjC  — all Objective-C/C/C++ sources (the core loader)
//   GLTFKit2      — Swift wrapper at Sources/GLTFKit2; re-exports ObjC module

import PackageDescription

let package = Package(
    name: "GLTFKit2",
    platforms: [
        .macOS(.v12), .macCatalyst(.v14), .iOS(.v15), .tvOS(.v15), .visionOS(.v1)
    ],
    products: [
        .library(name: "GLTFKit2", targets: ["GLTFKit2"])
    ],
    targets: [
        // MARK: ObjC core — the Xcode project's source directory minus Swift files
        .target(
            name: "GLTFKit2ObjC",
            path: "GLTFKit2/GLTFKit2",          // resolves to …/GLTFKit2/GLTFKit2/GLTFKit2/
            exclude: [
                "Info.plist",
                "PrivacyInfo.xcprivacy",
                "GLTFRealityKit.swift",          // lives in Sources/GLTFKit2/ for SPM
                "impl/WorkflowShaders.txt",
                "impl/GLTFAnimationHelpers.swift",
            ],
            publicHeadersPath: ".",
            cSettings: [
                // "." → source root itself, for "GLTFAsset.h" etc.
                .headerSearchPath("."),
                // "impl" → impl sub-headers
                .headerSearchPath("impl"),
                // cgltf single-file library (CGLTF_IMPLEMENTATION in GLTFAssetReader.m,
                // CGLTF_WRITE_IMPLEMENTATION in GLTFAssetWriter.m)
                .headerSearchPath("../deps/cgltf"),
                // ".." → parent dir, so <GLTFKit2/GLTFAsset.h> style imports resolve
                .headerSearchPath(".."),
            ],
        ),

        // MARK: Swift wrapper — re-exports ObjC module; adds the RealityKit bridge
        .target(
            name: "GLTFKit2",
            dependencies: ["GLTFKit2ObjC"],
            path: "Sources/GLTFKit2"
        ),

        // MARK: Tests
        .testTarget(
            name: "GLTFKit2Tests",
            dependencies: ["GLTFKit2"],
            path: "Tests/GLTFKit2Tests"
        )
    ]
)
