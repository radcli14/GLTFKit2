#if !os(tvOS)

import Testing
import CoreGraphics
import Foundation
import ImageIO
import Metal
@preconcurrency import RealityKit
import GLTFKit2

// Khronos Damaged Helmet — single mesh, full PBR textures (base color, normal, ORM, emissive, occlusion).
private let damagedHelmetURL = URL(string: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/refs/heads/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb")!

// MARK: - Helpers

/// Downloads the Damaged Helmet GLB once per process and caches it.
private actor HelmetCache {
    static let shared = HelmetCache()
    private var task: Task<Data, any Error>?
    func data() async throws -> Data {
        if task == nil {
            task = Task {
                let (data, response) = try await URLSession.shared.data(from: damagedHelmetURL)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw URLError(.badServerResponse)
                }
                return data
            }
        }
        return try await task!.value
    }
}

/// Loads a GLB from raw data via GLTFKit2's RealityKit bridge.
@MainActor
private func loadEntity(from data: Data) async throws -> Entity {
    let tempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("glb")
    try data.write(to: tempURL)
    defer { try? FileManager.default.removeItem(at: tempURL) }
    return try await GLTFRealityKitLoader.load(from: tempURL)
}

/// Copies a TextureResource into a CPU-readable `.rgba8Unorm` Metal texture and returns the raw bytes.
/// The buffer is pre-filled with 255 so that alpha defaults to fully opaque when the source has no
/// alpha channel and the copy leaves those bytes unwritten.
@MainActor
private func readTextureBytes(_ resource: TextureResource) -> [UInt8]? {
    guard let device = MTLCreateSystemDefaultDevice() else { return nil }
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: resource.width, height: resource.height, mipmapped: false)
    desc.usage = .shaderWrite
    desc.storageMode = .shared
    guard let tex = device.makeTexture(descriptor: desc) else { return nil }
    try? resource.copy(to: tex)
    let bpr = 4 * resource.width
    var bytes = [UInt8](repeating: 255, count: resource.height * bpr)
    bytes.withUnsafeMutableBytes { ptr in
        tex.getBytes(ptr.baseAddress!, bytesPerRow: bpr,
                     from: MTLRegion(origin: .init(),
                                     size: MTLSize(width: resource.width, height: resource.height, depth: 1)),
                     mipmapLevel: 0)
    }
    return bytes
}

/// Mean value of one RGBA channel (0=R, 1=G, 2=B, 3=A) across all pixels.
private func meanChannel(_ bytes: [UInt8], channel: Int) -> Float {
    var sum = 0
    var i = channel
    while i < bytes.count { sum += Int(bytes[i]); i += 4 }
    let count = bytes.count / 4
    return count > 0 ? Float(sum) / Float(count) : 0
}

/// Finds the first ModelEntity with a PhysicallyBasedMaterial anywhere in the entity tree.
@MainActor
private func firstPBR(in entity: Entity) -> PhysicallyBasedMaterial? {
    if let me = entity as? ModelEntity,
       let pbr = me.model?.materials.first as? PhysicallyBasedMaterial { return pbr }
    for child in entity.children { if let found = firstPBR(in: child) { return found } }
    return nil
}

/// Counts all PhysicallyBasedMaterial instances across the full entity tree.
@MainActor
private func countPBRs(in entity: Entity) -> Int {
    var count = 0
    if let me = entity as? ModelEntity {
        count += me.model?.materials.filter { $0 is PhysicallyBasedMaterial }.count ?? 0
    }
    for child in entity.children { count += countPBRs(in: child) }
    return count
}

/// Walks the GLTFAsset scene graph depth-first and returns the first base color texture sampler.
private func firstBaseColorSampler(in asset: GLTFAsset) -> GLTFTextureSampler? {
    func walkNode(_ node: GLTFNode) -> GLTFTextureSampler? {
        if let mesh = node.mesh {
            for primitive in mesh.primitives {
                if let sampler = primitive.material?.metallicRoughness?.baseColorTexture?.texture.sampler {
                    return sampler
                }
            }
        }
        for child in node.childNodes {
            if let found = walkNode(child) { return found }
        }
        return nil
    }
    guard let scene = asset.defaultScene else { return nil }
    for node in scene.nodes {
        if let found = walkNode(node) { return found }
    }
    return nil
}

// MARK: - PNG Encoding Diagnostic

/// Verifies that CGImage/CGImageDestination preserves pixel values through a PNG encode/decode cycle.
/// If this fails, the darkening is inside our makePNG encoding.
/// If this passes, the darkening is inside RealityKit's TextureResource loading pipeline.
@Test func testPNGEncodingPreservesValues() throws {
    let input: [UInt8] = [64, 66, 66, 255]
    let cs = CGColorSpace(name: CGColorSpace.linearSRGB)!
    guard let provider = CGDataProvider(data: Data(input) as CFData),
          let cgImage = CGImage(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: 4, space: cs,
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false,
                                intent: .defaultIntent) else {
        Issue.record("CGImage creation failed"); return
    }
    let pngData = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(pngData as CFMutableData, "public.png" as CFString, 1, nil) else {
        Issue.record("CGImageDestination creation failed"); return
    }
    CGImageDestinationAddImage(dest, cgImage, nil)
    guard CGImageDestinationFinalize(dest) else { Issue.record("PNG finalize failed"); return }

    guard let decodeProvider = CGDataProvider(data: pngData as CFData),
          let decoded = CGImage(pngDataProviderSource: decodeProvider, decode: nil,
                                shouldInterpolate: false, intent: .defaultIntent) else {
        Issue.record("PNG decode failed"); return
    }
    var output = [UInt8](repeating: 0, count: 4)
    output.withUnsafeMutableBytes { ptr in
        guard let ctx = CGContext(data: ptr.baseAddress, width: 1, height: 1,
                                  bitsPerComponent: 8, bytesPerRow: 4, space: cs,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
        else { return }
        ctx.draw(decoded, in: CGRect(x: 0, y: 0, width: 1, height: 1))
    }
    #expect(output[0] == 64, "R: \(output[0]) (expected 64)")
    #expect(output[1] == 66, "G: \(output[1]) (expected 66)")
    #expect(output[2] == 66, "B: \(output[2]) (expected 66)")
}

// MARK: - UV Coordinate Helpers

/// Returns the total number of vertices that have tangent vectors set across all mesh parts.
@MainActor
private func tangentVertexCount(in entity: Entity) -> Int {
    var count = 0
    func walk(_ e: Entity) {
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts {
                    count += part.tangents?.elements.count ?? 0
                }
            }
        }
        for child in e.children { walk(child) }
    }
    walk(entity)
    return count
}

/// Extracts (position, UV) pairs from all ModelEntity parts in the entity tree (depth-first).
@MainActor
private func extractPositionUVPairs(from entity: Entity) -> [(pos: SIMD3<Float>, uv: SIMD2<Float>)] {
    var pairs: [(SIMD3<Float>, SIMD2<Float>)] = []
    func walk(_ e: Entity) {
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts {
                    let positions = part.positions.elements
                    guard let uvs = part.textureCoordinates?.elements,
                          uvs.count == positions.count else { continue }
                    for i in 0..<positions.count {
                        pairs.append((positions[i], uvs[i]))
                    }
                }
            }
        }
        for child in e.children { walk(child) }
    }
    walk(entity)
    return pairs
}

/// Euclidean distance between two SIMD3<Float> points.
private func dist3(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Float {
    let d = a - b
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
}

// MARK: - UV Round-Trip Test

/// Verifies that UV coordinates survive a GLB round-trip using a synthetic quad.
///
/// A unit square has 4 vertices with UV corners: (0,0) (1,0) (1,1) (0,1).
/// After export→reload each source vertex is spatially matched to the closest
/// loaded vertex and U and V must agree within 0.01.
///
/// If V is being incorrectly flipped the test fails with messages like:
///   "V not preserved at pos (0,0,0): src=0.0, loaded=1.0"
@Test @MainActor func testUVCoordinatesRoundTripSynthetic() async throws {
    var descriptor = MeshDescriptor(name: "UVTestQuad")
    let positions: [SIMD3<Float>] = [
        SIMD3(0, 0, 0),   // v0 → UV (0.0, 0.0)
        SIMD3(1, 0, 0),   // v1 → UV (1.0, 0.0)
        SIMD3(1, 1, 0),   // v2 → UV (1.0, 1.0)
        SIMD3(0, 1, 0),   // v3 → UV (0.0, 1.0)
    ]
    let uvs: [SIMD2<Float>] = [
        SIMD2(0.0, 0.0), SIMD2(1.0, 0.0), SIMD2(1.0, 1.0), SIMD2(0.0, 1.0)
    ]
    descriptor.positions = MeshBuffers.Positions(positions)
    descriptor.textureCoordinates = MeshBuffers.TextureCoordinates(uvs)
    descriptor.primitives = .triangles([0, 1, 2, 0, 2, 3] as [UInt32])

    let mesh = try MeshResource.generate(from: [descriptor])
    let modelEntity = ModelEntity(mesh: mesh, materials: [PhysicallyBasedMaterial()])
    let wrapper = Entity()
    wrapper.addChild(modelEntity)

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    try GLTFRealityKitExporter().writeEntity(wrapper, to: outURL)
    let loaded = try await GLTFRealityKitLoader.load(from: outURL)

    let srcPairs = extractPositionUVPairs(from: wrapper)
    let ldPairs  = extractPositionUVPairs(from: loaded)
    #expect(!ldPairs.isEmpty, "No UV data in reloaded entity — mesh was not exported")

    for (srcPos, srcUV) in srcPairs {
        guard let best = ldPairs.min(by: { dist3($0.pos, srcPos) < dist3($1.pos, srcPos) }) else { continue }
        #expect(dist3(best.pos, srcPos) < 0.001,
                "Position not preserved: src=\(srcPos), loaded=\(best.pos)")
        #expect(abs(best.uv.x - srcUV.x) < 0.01,
                "U not preserved at pos \(srcPos): src=\(srcUV.x), loaded=\(best.uv.x)")
        #expect(abs(best.uv.y - srcUV.y) < 0.01,
                "V not preserved at pos \(srcPos): src=\(srcUV.y), loaded=\(best.uv.y)")
    }
}

/// Verifies that UV coordinates survive a GLB round-trip on a complex real-world model.
///
/// Downloads DamagedHelmet, exports via GLTFRealityKitExporter, reloads, then spatially
/// matches a sample of source vertices to the closest loaded vertices and checks that
/// U and V agree within 0.02. A tight position tolerance (0.01) guards against false
/// matches across distant vertices.
@Test @MainActor func testUVCoordinatesRoundTripHelmet() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    try GLTFRealityKitExporter().writeEntity(source, to: outURL)
    let loaded = try await GLTFRealityKitLoader.load(from: outURL)

    let srcPairs = extractPositionUVPairs(from: source)
    let ldPairs  = extractPositionUVPairs(from: loaded)

    #expect(!srcPairs.isEmpty, "No UV data in source entity")
    #expect(!ldPairs.isEmpty,  "No UV data in reloaded entity")

    // Sample every Nth pair so the test runs in reasonable time.
    let step = max(1, srcPairs.count / 100)
    var mismatches = 0
    for i in Swift.stride(from: 0, to: srcPairs.count, by: step) {
        let (srcPos, srcUV) = srcPairs[i]
        // Gather ALL loaded vertices within position tolerance.
        // Using "any nearby match" rather than "nearest single vertex" correctly handles
        // UV seams, where the same 3D position hosts multiple vertices with distinct UVs.
        let nearby = ldPairs.filter { dist3($0.pos, srcPos) < 0.01 }
        guard !nearby.isEmpty else { continue }
        let hasMatch = nearby.contains { abs($0.uv.x - srcUV.x) < 0.02 && abs($0.uv.y - srcUV.y) < 0.02 }
        if !hasMatch {
            mismatches += 1
            if mismatches <= 3 {
                let best = nearby.min(by: { dist3($0.pos, srcPos) < dist3($1.pos, srcPos) })!
                Issue.record("UV mismatch at pos \(srcPos): src=\(srcUV), best loaded=\(best.uv) (\(nearby.count) nearby vertices)")
            }
        }
    }
    #expect(mismatches == 0, "\(mismatches) UV mismatches found in helmet round-trip")
}

/// Documents that RealityKit does not expose tangent vectors through MeshResource.Part.tangents.
///
/// Even though the DamagedHelmet GLTF contains TANGENT attributes, and GLTFRealityKitLoader
/// calls `part[MeshBuffers.tangents] = MeshBuffers.Tangents(...)` during load, RealityKit
/// manages tangents internally and returns nil from `part.tangents` on a generated MeshResource.
/// This means:
///   - GLTFWriterMeshPart's lack of tangentData cannot be detected through this API
///   - Both source and round-trip entities have auto-generated tangents from the same
///     positions+normals+UVs, so tangent loss does not explain visual differences between them
///
/// This test confirms that behavior and serves as a canary: if it ever starts passing
/// (srcTangents > 0), the API has changed and tangent round-trip should be tested properly.
@Test @MainActor func testTangentsRoundTripHelmet() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    let srcTangents = tangentVertexCount(in: source)
    print("tangentVertexCount (via part.tangents): src=\(srcTangents)")

    // RealityKit does not surface tangents through part.tangents on generated meshes.
    // If this changes and srcTangents > 0, extend this test to compare source vs. loaded.
    #expect(srcTangents == 0,
            "part.tangents now returns data — update this test to compare tangents across the round-trip")
}

// MARK: - Material Slot Round-Trip Tests

/// Checks that every PBR texture slot present in the source also appears in the round-trip,
/// and that the emissive and base color mean pixel values are preserved.
///
/// A failure here identifies which specific slot is being dropped or corrupted by the exporter.
@Test @MainActor func testAllMaterialSlotsRoundTrip() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    guard let srcPBR = firstPBR(in: source) else {
        Issue.record("No PBR material in source entity"); return
    }

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    try GLTFRealityKitExporter().writeEntity(source, to: outURL)
    let loaded = try await GLTFRealityKitLoader.load(from: outURL)

    guard let ldPBR = firstPBR(in: loaded) else {
        Issue.record("No PBR material in loaded entity"); return
    }

    // --- Texture presence ---
    let slots: [(String, Bool, Bool)] = [
        ("baseColor",  srcPBR.baseColor.texture != nil,      ldPBR.baseColor.texture != nil),
        ("normal",     srcPBR.normal.texture != nil,         ldPBR.normal.texture != nil),
        ("emissive",   srcPBR.emissiveColor.texture != nil,  ldPBR.emissiveColor.texture != nil),
        ("metallic",   srcPBR.metallic.texture != nil,       ldPBR.metallic.texture != nil),
        ("roughness",  srcPBR.roughness.texture != nil,      ldPBR.roughness.texture != nil),
        ("occlusion",  srcPBR.ambientOcclusion.texture != nil, ldPBR.ambientOcclusion.texture != nil),
    ]
    for (name, srcHas, ldHas) in slots {
        #expect(srcHas == ldHas,
                "\(name) texture presence mismatch: src=\(srcHas), loaded=\(ldHas)")
    }

    // --- Scalar factors ---
    #expect(abs(ldPBR.metallic.scale  - srcPBR.metallic.scale)  < 0.05,
            "metallicFactor: src=\(srcPBR.metallic.scale), loaded=\(ldPBR.metallic.scale)")
    #expect(abs(ldPBR.roughness.scale - srcPBR.roughness.scale) < 0.05,
            "roughnessFactor: src=\(srcPBR.roughness.scale), loaded=\(ldPBR.roughness.scale)")

    // --- Emissive pixel values (the visor glow) ---
    if let srcRes = srcPBR.emissiveColor.texture?.resource,
       let ldRes  = ldPBR.emissiveColor.texture?.resource,
       let srcB   = readTextureBytes(srcRes),
       let ldB    = readTextureBytes(ldRes) {
        let srcR = meanChannel(srcB, channel: 0)
        let srcG = meanChannel(srcB, channel: 1)
        let srcBB = meanChannel(srcB, channel: 2)
        let ldR  = meanChannel(ldB,  channel: 0)
        let ldG  = meanChannel(ldB,  channel: 1)
        let ldBB = meanChannel(ldB,  channel: 2)
        #expect(abs(ldR  - srcR)  < 10, "emissive mean R:  src=\(srcR),  loaded=\(ldR)")
        #expect(abs(ldG  - srcG)  < 10, "emissive mean G:  src=\(srcG),  loaded=\(ldG)")
        #expect(abs(ldBB - srcBB) < 10, "emissive mean B:  src=\(srcBB), loaded=\(ldBB)")
    }

    // --- Normal map presence sanity ---
    #expect(srcPBR.normal.texture != nil,
            "DamagedHelmet should have a normal texture — source assumption is wrong")
}

/// Verifies that metallic and roughness texture content survives a GLB round-trip.
///
/// The DamagedHelmet has a combined ORM texture; GLTFKit2 splits it into separate
/// grayscale TextureResources for metallic and roughness. The exporter must read back
/// those grayscale resources correctly — if `TextureResource.copy(to:)` silently
/// fails to write into a 4-channel `.rgba8Unorm` Metal texture when the source is
/// a single-channel (grayscale) resource, the buffer stays filled with 255, causing
/// metallic=1.0 and roughness=1.0 on every pixel, which renders as a near-black
/// matte-metallic surface.
///
/// Failure pattern:
///   - srcMetallicMean ≈ 255  → grayscale readback broken even for the source
///   - srcMetallicMean correct but ldMetallicMean ≈ 255  → round-trip exports 1.0 everywhere
@Test @MainActor func testMetallicRoughnessTextureRoundTrip() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    guard let srcPBR = firstPBR(in: source) else {
        Issue.record("No PBR material in source entity"); return
    }

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    try GLTFRealityKitExporter().writeEntity(source, to: outURL)
    let loaded = try await GLTFRealityKitLoader.load(from: outURL)

    guard let ldPBR = firstPBR(in: loaded) else {
        Issue.record("No PBR material in loaded entity"); return
    }

    // --- Metallic ---
    guard let srcMetRes = srcPBR.metallic.texture?.resource,
          let srcMetBytes = readTextureBytes(srcMetRes) else {
        Issue.record("Source has no metallic texture — DamagedHelmet assumption wrong"); return
    }
    let srcMetMean = meanChannel(srcMetBytes, channel: 0)
    print("srcMetallicMean=\(srcMetMean)")

    // If srcMetMean ≈ 255, readTextureBytes cannot copy grayscale TextureResources:
    // the 4-channel .rgba8Unorm destination receives nothing and stays all-255.
    #expect(srcMetMean < 250,
            "Source metallic texture reads as all-1.0 (\(srcMetMean)/255) — TextureResource.copy(to:) may be silently failing for grayscale resources")

    if let ldMetRes = ldPBR.metallic.texture?.resource,
       let ldMetBytes = readTextureBytes(ldMetRes) {
        let ldMetMean = meanChannel(ldMetBytes, channel: 0)
        print("ldMetallicMean=\(ldMetMean)")
        #expect(abs(ldMetMean - srcMetMean) < 15,
                "Metallic texture mean mismatch: src=\(srcMetMean), loaded=\(ldMetMean)")
    } else {
        Issue.record("Loaded entity has no metallic texture")
    }

    // --- Roughness ---
    guard let srcRghRes = srcPBR.roughness.texture?.resource,
          let srcRghBytes = readTextureBytes(srcRghRes) else {
        Issue.record("Source has no roughness texture — DamagedHelmet assumption wrong"); return
    }
    let srcRghMean = meanChannel(srcRghBytes, channel: 0)
    print("srcRoughnessMean=\(srcRghMean)")

    #expect(srcRghMean < 250,
            "Source roughness texture reads as all-1.0 (\(srcRghMean)/255) — same grayscale readback issue as metallic")

    if let ldRghRes = ldPBR.roughness.texture?.resource,
       let ldRghBytes = readTextureBytes(ldRghRes) {
        let ldRghMean = meanChannel(ldRghBytes, channel: 0)
        print("ldRoughnessMean=\(ldRghMean)")
        #expect(abs(ldRghMean - srcRghMean) < 15,
                "Roughness texture mean mismatch: src=\(srcRghMean), loaded=\(ldRghMean)")
    } else {
        Issue.record("Loaded entity has no roughness texture")
    }
}

// MARK: - Round-Trip Texture Tests

/// Verifies that the base color texture's mean pixel values are preserved through a GLB round-trip.
/// Source (loaded from original GLB) and loaded (reloaded from our exported GLB) should produce
/// similar raw byte values when read back through a Metal `.rgba8Unorm` texture.
@Test @MainActor func testBaseColorTextureRoundTrip() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    guard let srcPBR = firstPBR(in: source),
          let srcResource = srcPBR.baseColor.texture?.resource,
          let srcBytes = readTextureBytes(srcResource) else {
        Issue.record("Could not read source base color texture"); return
    }

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    let exporter = GLTFRealityKitExporter()
    try exporter.writeEntity(source, to: outURL)

    let loaded = try await GLTFRealityKitLoader.load(from: outURL)

    guard let ldPBR = firstPBR(in: loaded),
          let ldResource = ldPBR.baseColor.texture?.resource,
          let ldBytes = readTextureBytes(ldResource) else {
        Issue.record("Could not read loaded base color texture"); return
    }

    let srcR = meanChannel(srcBytes, channel: 0)
    let srcG = meanChannel(srcBytes, channel: 1)
    let srcB = meanChannel(srcBytes, channel: 2)
    let ldR  = meanChannel(ldBytes,  channel: 0)
    let ldG  = meanChannel(ldBytes,  channel: 1)
    let ldB  = meanChannel(ldBytes,  channel: 2)

    // Verify the entity trees contain the same number of PBR materials so we know
    // the firstPBR() calls above are comparing equivalent slots.
    let srcPBRCount = countPBRs(in: source)
    let ldPBRCount  = countPBRs(in: loaded)
    #expect(srcPBRCount == ldPBRCount,
            "PBR material count mismatch: src=\(srcPBRCount), loaded=\(ldPBRCount)")

    // Verify texture dimensions survived the round-trip.
    #expect(srcResource.width  == ldResource.width,
            "Texture width mismatch: src=\(srcResource.width), loaded=\(ldResource.width)")
    #expect(srcResource.height == ldResource.height,
            "Texture height mismatch: src=\(srcResource.height), loaded=\(ldResource.height)")

    // Per-pixel comparison across all three RGB channels.
    // Only runs when dimensions match to avoid index-out-of-bounds; dimension failures above
    // already flag the mismatch when they differ.
    //
    // Threshold is 5 (not 20) so we detect systematic shifts. A mean difference of ~10
    // means every pixel could be off by ~10 — a threshold of 20 would miss that entirely.
    if srcResource.width == ldResource.width && srcResource.height == ldResource.height {
        let pixelCount = srcBytes.count / 4
        var maxDiff = 0
        var gt1 = 0, gt5 = 0, gt10 = 0, gt20 = 0
        for px in 0..<pixelCount {
            for c in 0..<3 {
                let diff = abs(Int(srcBytes[px * 4 + c]) - Int(ldBytes[px * 4 + c]))
                if diff > maxDiff { maxDiff = diff }
                if diff > 1  { gt1  += 1 }
                if diff > 5  { gt5  += 1 }
                if diff > 10 { gt10 += 1 }
                if diff > 20 { gt20 += 1 }
            }
        }
        let total = pixelCount * 3
        let allowed: Int = total / 100 // Tune for strictness 
        print("Base color pixel diff distribution (of \(total) channel values): >1=\(gt1), >5=\(gt5), >10=\(gt10), >20=\(gt20), maxDiff=\(maxDiff)")
        #expect(gt5 < allowed,
                "Base color pixel mismatch: \(gt5)/\(total) channel values differ by >5; maxDiff=\(maxDiff). Distribution: >1=\(gt1), >10=\(gt10), >20=\(gt20)")
    }

    #expect(abs(ldR - srcR) < 10, "Base color mean R: src=\(srcR), loaded=\(ldR)")
    #expect(abs(ldG - srcG) < 10, "Base color mean G: src=\(srcG), loaded=\(ldG)")
    #expect(abs(ldB - srcB) < 10, "Base color mean B: src=\(srcB), loaded=\(ldB)")
}

// MARK: - Skinned Mesh Baking Helpers

/// Collects all vertex positions from the entire entity tree (bind-pose, no skinning applied).
@MainActor
private func allPositions(from entity: Entity) -> [SIMD3<Float>] {
    var result: [SIMD3<Float>] = []
    func walk(_ e: Entity) {
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts { result.append(contentsOf: part.positions.elements) }
            }
        }
        for child in e.children { walk(child) }
    }
    walk(entity)
    return result
}

/// Applies LBS to one mesh part's positions using model-space joint transforms.
/// Mirrors the exact math in GLTFRealityKitExporter.bakeSkinnedGeometry.
/// modelSpaceTransforms must be in entity-local (model) space — composed from parent-relative sources.
@available(macOS 15.0, iOS 18.0, *)
private func applyLBS(
    positions: [SIMD3<Float>],
    influences: MeshResource.JointInfluences,
    skeleton: MeshResource.Skeleton,
    modelSpaceTransforms: [simd_float4x4]
) -> [SIMD3<Float>] {
    let skinMtx = skeleton.joints.indices.map { i -> simd_float4x4 in
        let M = i < modelSpaceTransforms.count ? modelSpaceTransforms[i] : matrix_identity_float4x4
        return M * skeleton.joints[i].inverseBindPoseMatrix
    }

    let allInf = influences.influences.elements
    let ipv = positions.count > 0 ? allInf.count / positions.count : 0
    guard ipv > 0 else { return positions }

    return positions.indices.map { v in
        var dp = SIMD4<Float>.zero
        for k in 0..<ipv {
            let idx = v * ipv + k
            guard idx < allInf.count else { break }
            let inf = allInf[idx]
            guard inf.weight > 0, inf.jointIndex < skinMtx.count else { continue }
            let M = skinMtx[inf.jointIndex]
            dp += inf.weight * (M * SIMD4<Float>(positions[v].x, positions[v].y, positions[v].z, 1))
        }
        return SIMD3<Float>(dp.x, dp.y, dp.z)
    }
}

/// Returns the expected baked positions for the full entity tree, applying LBS where joint influences exist.
@available(macOS 15.0, iOS 18.0, *)
@MainActor
private func expectedBakedPositions(from entity: Entity) -> [SIMD3<Float>] {
    var result: [SIMD3<Float>] = []
    func walk(_ e: Entity) {
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts {
                    let positions = part.positions.elements
                    guard !positions.isEmpty else { continue }
                    if let ji = part.jointInfluences,
                       let skelID = part.skeletonID,
                       let skel = model.mesh.contents.skeletons[skelID] {
                        // jointTransforms are parent-relative — compose up the hierarchy to model-space.
                        let jt = me.jointTransforms
                        let localXforms: [simd_float4x4] = jt.count == skel.joints.count
                            ? jt.map { $0.matrix }
                            : skel.joints.map { $0.restPoseTransform.matrix }
                        var modelSpaceXforms = [simd_float4x4](repeating: matrix_identity_float4x4, count: skel.joints.count)
                        for (i, joint) in skel.joints.enumerated() {
                            modelSpaceXforms[i] = joint.parentIndex.map { modelSpaceXforms[$0] * localXforms[i] } ?? localXforms[i]
                        }
                        result.append(contentsOf: applyLBS(positions: positions, influences: ji,
                                                           skeleton: skel, modelSpaceTransforms: modelSpaceXforms))
                    } else {
                        result.append(contentsOf: positions)
                    }
                }
            }
        }
        for child in e.children { walk(child) }
    }
    walk(entity)
    return result
}

/// Component-wise AABB over a position array.
private func aabb(of positions: [SIMD3<Float>]) -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
    guard !positions.isEmpty else { return nil }
    var mn = SIMD3<Float>(repeating: Float.infinity)
    var mx = SIMD3<Float>(repeating: -.infinity)
    for p in positions { mn = min(mn, p); mx = max(mx, p) }
    return (mn, mx)
}

// MARK: - Skinned Mesh Baking Test

/// Verifies that the GLTFRealityKitExporter correctly bakes linear-blend skinning into
/// static vertex positions when exporting a skinned USDZ mesh.
///
/// The test replicates the exporter's LBS math independently to compute "expected" baked
/// positions, then exports the entity to GLB, reloads it, and compares:
///   - Bounding box dimensions (detects scale/units errors and gross deformation bugs)
///   - Total vertex count (detects dropped mesh parts)
///
/// Diagnostic output includes the entity hierarchy (names, transforms, skeleton info)
/// and side-by-side AABBs so you can immediately see which axis is stretched.
@Test @MainActor
@available(macOS 15.0, iOS 18.0, *)
func testSkinnedBakingLeftHand() async throws {
    let testDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    let sourceURL = testDir.appendingPathComponent("left_hand.usdz")
    guard FileManager.default.fileExists(atPath: sourceURL.path) else {
        Issue.record("left_hand.usdz not found at \(sourceURL.path)"); return
    }

    let source = try await Entity(contentsOf: sourceURL)

    // Print entity tree for diagnosing coordinate-space and skeleton structure issues.
    func printTree(_ e: Entity, depth: Int = 0) {
        let indent = String(repeating: "  ", count: depth)
        let t = e.transform
        print("\(indent)'\(e.name)' scale=\(t.scale) trans=\(t.translation)")
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts {
                    let skelID = part.skeletonID ?? "none"
                    let jointCount = part.skeletonID.flatMap { model.mesh.contents.skeletons[$0] }?.joints.count ?? 0
                    let infCount = part.jointInfluences?.influences.elements.count ?? 0
                    let ipv = part.positions.elements.count > 0 ? infCount / part.positions.elements.count : 0
                    print("\(indent)  part: verts=\(part.positions.elements.count) skelID='\(skelID)' joints=\(jointCount) ipv=\(ipv)")
                }
            }
            print("\(indent)  jointTransforms.count=\(me.jointTransforms.count)")
        }
        for child in e.children { printTree(child, depth: depth + 1) }
    }
    printTree(source)

    // DIAGNOSTIC: Rest-pose identity check and joint coordinate comparison.
    // Composing restPoseTransform (parent-relative) up the hierarchy gives model-space joint positions.
    // Applying LBS with those transforms to bind-pose vertices MUST reproduce the bind-pose vertices
    // (within floating-point error) — if it doesn't, restPoseTransform and inverseBindPoseMatrix
    // are not in the same coordinate space.
    func findSkinnedModelEntity(_ e: Entity) -> ModelEntity? {
        if let me = e as? ModelEntity, let model = me.model {
            for rkModel in model.mesh.contents.models {
                for part in rkModel.parts { if part.jointInfluences != nil { return me } }
            }
        }
        for child in e.children { if let found = findSkinnedModelEntity(child) { return found } }
        return nil
    }
    if let me = findSkinnedModelEntity(source), let model = me.model,
       let rkModel = model.mesh.contents.models.first,
       let part = rkModel.parts.first(where: { $0.jointInfluences != nil }),
       let skelID = part.skeletonID,
       let skel = model.mesh.contents.skeletons[skelID],
       let ji = part.jointInfluences {

        let bindPositions = part.positions.elements

        // Compose restPoseTransform up the skeleton hierarchy → model-space per joint.
        var restXforms = [simd_float4x4](repeating: matrix_identity_float4x4, count: skel.joints.count)
        for (i, joint) in skel.joints.enumerated() {
            let local = joint.restPoseTransform.matrix
            restXforms[i] = joint.parentIndex.map { restXforms[$0] * local } ?? local
        }

        // Print first 4 joints: compare restPose, invBind, and live jointTransforms translations + rotations.
        let jt = me.jointTransforms
        let invBindXforms = skel.joints.map { simd_inverse($0.inverseBindPoseMatrix) }
        print("--- Joint Coordinate Comparison ---")
        for i in 0..<min(4, skel.joints.count) {
            let joint = skel.joints[i]
            let rt = restXforms[i].columns.3
            let bt = invBindXforms[i].columns.3
            let lt: SIMD3<Float> = i < jt.count ? jt[i].translation : .zero
            let restRot = Transform(matrix: restXforms[i]).rotation
            let bindRot = Transform(matrix: invBindXforms[i]).rotation
            let liveRot: simd_quatf = i < jt.count ? jt[i].rotation : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
            let fmt = { (v: Float) in String(format: "%.4f", v) }
            print("  [\(i)] parent=\(joint.parentIndex.map(String.init) ?? "nil")")
            print("    restPose_model:  t=(\(fmt(rt.x)), \(fmt(rt.y)), \(fmt(rt.z)))  q=(\(fmt(restRot.vector.x)), \(fmt(restRot.vector.y)), \(fmt(restRot.vector.z)), \(fmt(restRot.vector.w)))")
            print("    bindPose_model:  t=(\(fmt(bt.x)), \(fmt(bt.y)), \(fmt(bt.z)))  q=(\(fmt(bindRot.vector.x)), \(fmt(bindRot.vector.y)), \(fmt(bindRot.vector.z)), \(fmt(bindRot.vector.w)))  [inv(invBind)]")
            print("    jointTransforms: t=(\(fmt(lt.x)), \(fmt(lt.y)), \(fmt(lt.z)))  q=(\(fmt(liveRot.vector.x)), \(fmt(liveRot.vector.y)), \(fmt(liveRot.vector.z)), \(fmt(liveRot.vector.w)))")
        }

        // InvBind identity check: using inv(invBind) as joint transforms must give back bind-pose positions.
        // This verifies the LBS math itself is correct. If this fails, the applyLBS function has a bug.
        let invBindBaked = applyLBS(positions: bindPositions, influences: ji, skeleton: skel, modelSpaceTransforms: invBindXforms)
        var maxInvBindErr: Float = 0
        for (a, b) in zip(bindPositions, invBindBaked) { maxInvBindErr = max(maxInvBindErr, length(a - b)) }
        print("InvBind identity check: maxError=\(maxInvBindErr) m  (must be ≈ 0 — if not, applyLBS has a bug)")
        #expect(maxInvBindErr < 0.001, "LBS identity failed: inv(invBind) as joint transforms did not reproduce bind positions (maxError=\(maxInvBindErr) m)")

        // Rest-pose identity check: LBS with composed restXforms should reproduce bind-pose positions
        // when rest pose == bind pose. A large error means restPoseTransform and inverseBindPoseMatrix
        // are in different coordinate spaces, OR rest pose ≠ bind pose for this model.
        let restBaked = applyLBS(positions: bindPositions, influences: ji, skeleton: skel, modelSpaceTransforms: restXforms)
        var maxRestErr: Float = 0
        for (a, b) in zip(bindPositions, restBaked) { maxRestErr = max(maxRestErr, length(a - b)) }
        print("Rest-pose LBS identity check (restPoseTransform vs bind-pose): maxError=\(maxRestErr) m")

        // Compose jointTransforms (parent-relative) up the hierarchy → model-space, then apply LBS.
        // This is the same computation the exporter now uses. The resulting AABB should match
        // the "Expected AABB" printed below (since both use the same composed model-space transforms).
        if jt.count == skel.joints.count {
            var composedJt = [simd_float4x4](repeating: matrix_identity_float4x4, count: skel.joints.count)
            for (i, joint) in skel.joints.enumerated() {
                composedJt[i] = joint.parentIndex.map { composedJt[$0] * jt[i].matrix } ?? jt[i].matrix
            }
            let composedBaked = applyLBS(positions: bindPositions, influences: ji, skeleton: skel, modelSpaceTransforms: composedJt)
            if let bb = aabb(of: composedBaked) {
                print("Composed-jt LBS AABB: size=\(bb.max - bb.min)  min=\(bb.min)")
            }
        }
    }

    let expPos = expectedBakedPositions(from: source)
    print("Expected (LBS-baked) vertex count: \(expPos.count)")

    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }

    try GLTFRealityKitExporter().writeEntity(source, to: outURL)
    let loaded = try await GLTFRealityKitLoader.load(from: outURL)
    let ldPos = allPositions(from: loaded)
    print("Loaded (GLB) vertex count: \(ldPos.count)")

    #expect(!expPos.isEmpty, "No vertices found in source entity — left_hand.usdz may have no skinned geometry")
    #expect(!ldPos.isEmpty,  "No vertices in exported GLB — export produced empty file")

    // Sanity: baking should move vertices. Compare baked bounding box against bind-pose bounding box.
    // For a hand in its rest/fist pose the baked positions will differ measurably from bind pose.
    let bindPos = allPositions(from: source)
    if let bindBB = aabb(of: bindPos), let expBB2 = aabb(of: expPos) {
        let bindSize = bindBB.max - bindBB.min
        let expSize2 = expBB2.max - expBB2.min
        print("Bind-pose AABB: size=\(bindSize)")
        let maxDimDiff = max(abs(expSize2.x - bindSize.x), abs(expSize2.y - bindSize.y), abs(expSize2.z - bindSize.z))
        #expect(maxDimDiff > 0.001 || expPos == bindPos,
                "LBS baking produced no change from bind pose — jointTransforms may be identity or skinning is not running")
    }

    // Bounding box comparison: checks both size (shape) and position (origin offset).
    // A constant offset in baked positions passes a size-only check but fails a position check.
    if let expBB = aabb(of: expPos), let ldBB = aabb(of: ldPos) {
        let expSize = expBB.max - expBB.min
        let ldSize  = ldBB.max  - ldBB.min
        print("Expected AABB: min=\(expBB.min) max=\(expBB.max) size=\(expSize)")
        print("Loaded AABB:   min=\(ldBB.min)  max=\(ldBB.max)  size=\(ldSize)")

        let tol: Float = 0.05  // 5% relative or 1 mm absolute — whichever is larger
        for (axis, expLen, ldLen) in [("X", expSize.x, ldSize.x), ("Y", expSize.y, ldSize.y), ("Z", expSize.z, ldSize.z)] {
            let tolerance = max(abs(expLen) * tol, 0.001)
            #expect(abs(ldLen - expLen) < tolerance,
                    "AABB \(axis) size mismatch: expected=\(expLen) loaded=\(ldLen) — check for scale/units error or wrong joint-transform space in bakeSkinnedGeometry")
        }

        // Position check: if both AABBs have the same size but are offset, the shape is right but
        // the mesh is at the wrong location (e.g. wrong joint translation space).
        let minOff = length(expBB.min - ldBB.min)
        let maxOff = length(expBB.max - ldBB.max)
        print("AABB position offset: min_offset=\(minOff) m  max_offset=\(maxOff) m")
        #expect(minOff < 0.01,
                "AABB min position mismatch: expected=\(expBB.min) loaded=\(ldBB.min) — the mesh has the right shape but is at the wrong origin")
        #expect(maxOff < 0.01,
                "AABB max position mismatch: expected=\(expBB.max) loaded=\(ldBB.max) — the mesh has the right shape but is at the wrong origin")
    }

    #expect(ldPos.count == expPos.count,
            "Vertex count mismatch: expected=\(expPos.count) loaded=\(ldPos.count)")
}

// MARK: - Texture Sampler Round-Trip Tests

/// Verifies that texture wrapping modes (wrapS/wrapT) survive a GLB round-trip.
///
/// The DamagedHelmet's base color texture uses GL_REPEAT (wrapS/wrapT = 10497 = 0x2901)
/// because UV coordinates extend outside [0, 1]. GLTFAssetWriter.m currently hardcodes
/// CLAMP_TO_EDGE (33071 = 0x812F) for all exported samplers, which causes any UV < 0
/// or > 1 to sample the edge pixel instead of tiling — producing the near-black
/// appearance observed in the round-trip render.
///
/// This test is expected to FAIL until GLTFAssetWriter.m propagates the source
/// sampler's wrapS/wrapT instead of hardcoding CLAMP_TO_EDGE.
@Test @MainActor func testTextureWrappingModeRoundTrip() async throws {
    let data = try await HelmetCache.shared.data()
    let source = try await loadEntity(from: data)

    // Confirm UV coordinates go outside [0, 1] — this is why wrapping mode matters.
    let srcPairs = extractPositionUVPairs(from: source)
    let outOfRange = srcPairs.filter { $0.uv.x < 0 || $0.uv.x > 1 || $0.uv.y < 0 || $0.uv.y > 1 }
    #expect(!outOfRange.isEmpty, "Expected DamagedHelmet to have UV coordinates outside [0,1]; if not, wrapping mode does not affect rendering")

    // Load source as GLTFAsset to read the original sampler settings.
    let srcTempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    try data.write(to: srcTempURL)
    defer { try? FileManager.default.removeItem(at: srcTempURL) }

    let srcAsset = try GLTFAsset(url: srcTempURL)
    let srcSampler = firstBaseColorSampler(in: srcAsset)
    #expect(srcSampler != nil, "Source DamagedHelmet has no base color texture sampler")

    // Export via GLTFRealityKitExporter.
    let outURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).appendingPathExtension("glb")
    defer { try? FileManager.default.removeItem(at: outURL) }
    try GLTFRealityKitExporter().writeEntity(source, to: outURL)

    // Load exported GLB as GLTFAsset.
    let ldAsset = try GLTFAsset(url: outURL)
    let ldSampler = firstBaseColorSampler(in: ldAsset)
    #expect(ldSampler != nil, "Exported GLB has no base color texture sampler")

    if let src = srcSampler, let ld = ldSampler {
        let srcWrapS = src.wrapS.rawValue, srcWrapT = src.wrapT.rawValue
        let ldWrapS  = ld.wrapS.rawValue,  ldWrapT  = ld.wrapT.rawValue
        print("wrapS: src=\(srcWrapS), loaded=\(ldWrapS)  (REPEAT=10497, CLAMP=33071)")
        print("wrapT: src=\(srcWrapT), loaded=\(ldWrapT)  (REPEAT=10497, CLAMP=33071)")
        #expect(ld.wrapS == src.wrapS, "wrapS mismatch: src=\(srcWrapS) REPEAT=10497, loaded=\(ldWrapS) CLAMP=33071 — GLTFAssetWriter.m hardcodes CLAMP_TO_EDGE")
        #expect(ld.wrapT == src.wrapT, "wrapT mismatch: src=\(srcWrapT) REPEAT=10497, loaded=\(ldWrapT) CLAMP=33071 — GLTFAssetWriter.m hardcodes CLAMP_TO_EDGE")
    }
}

#endif
