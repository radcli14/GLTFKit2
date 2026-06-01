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

    #expect(abs(ldR - srcR) < 10, "Base color mean R: src=\(srcR), loaded=\(ldR)")
    #expect(abs(ldG - srcG) < 10, "Base color mean G: src=\(srcG), loaded=\(ldG)")
    #expect(abs(ldB - srcB) < 10, "Base color mean B: src=\(srcB), loaded=\(ldB)")
}

#endif
