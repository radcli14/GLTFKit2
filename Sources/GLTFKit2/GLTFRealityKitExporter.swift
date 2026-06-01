#if !os(tvOS) && compiler(>=5.6)

import GLTFKit2ObjC
import RealityKit
import Metal
import ImageIO

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

public enum GLTFExportError: LocalizedError {
    case noMeshData
    case metalDeviceUnavailable
    case textureReadbackFailed

    public var errorDescription: String? {
        switch self {
        case .noMeshData:                return "The entity tree contains no mesh data to export."
        case .metalDeviceUnavailable:    return "No Metal device available for texture readback."
        case .textureReadbackFailed:     return "Failed to read texture pixels from the GPU."
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

@available(macOS 12.0, iOS 15.0, *)
@MainActor
public class GLTFRealityKitExporter {

    public init() {}

    // MARK: - Public

    /// Writes the entity tree to a GLB file at `url`.
    public func writeEntity(_ entity: Entity, to url: URL) throws {
        let (rootNodes, materials) = try buildExportGraph(from: entity)
        try GLTFAssetWriter.writeGLB(to: url, rootNodes: rootNodes, materials: materials)
    }

    /// Returns GLB binary data for the entity tree.
    public func dataFromEntity(_ entity: Entity) throws -> Data {
        let (rootNodes, materials) = try buildExportGraph(from: entity)
        return try GLTFAssetWriter.glbData(forRootNodes: rootNodes, materials: materials)
    }

    // MARK: - Private: graph construction

    private func buildExportGraph(from entity: Entity) throws -> ([GLTFWriterNode], [GLTFWriterMaterial]) {
        var materials: [GLTFWriterMaterial] = []
        // Cache materials by identity so the same RK material becomes one glTF material.
        var materialCache: [ObjectIdentifier: Int] = [:]

        func processNode(_ e: Entity) throws -> GLTFWriterNode {
            let wn = GLTFWriterNode()
            wn.name = e.name.isEmpty ? nil : e.name

            let t = e.transform
            wn.translation = t.translation
            wn.rotation    = t.rotation
            wn.scale       = t.scale

            if let me = e as? ModelEntity, let model = me.model {
                let worldTransform = me.transformMatrix(relativeTo: nil)
                wn.meshParts = try buildMeshParts(
                    model: model,
                    worldTransform: worldTransform,
                    rkMaterials: model.materials,
                    materialList: &materials,
                    cache: &materialCache
                )
            }

            wn.light = extractLight(from: e)

            wn.children = try e.children.map { try processNode($0) }
            return wn
        }

        let rootNodes = try entity.children.map { try processNode($0) }
        return (rootNodes, materials)
    }

    // MARK: - Mesh parts

    private func buildMeshParts(
        model: ModelComponent,
        worldTransform: simd_float4x4,
        rkMaterials: [any Material],
        materialList: inout [GLTFWriterMaterial],
        cache: inout [ObjectIdentifier: Int]
    ) throws -> [GLTFWriterMeshPart] {

        var normalMatrix: simd_float3x3 {
            simd_float3x3(
                SIMD3<Float>(worldTransform[0].x, worldTransform[0].y, worldTransform[0].z),
                SIMD3<Float>(worldTransform[1].x, worldTransform[1].y, worldTransform[1].z),
                SIMD3<Float>(worldTransform[2].x, worldTransform[2].y, worldTransform[2].z)
            )
        }

        var parts: [GLTFWriterMeshPart] = []

        for rkModel in model.mesh.contents.models {
            for part in rkModel.parts {
                let positions = part.positions.elements
                guard !positions.isEmpty, let indexBuf = part.triangleIndices else { continue }

                let nm = normalMatrix
                let worldPositions = positions.map { p -> SIMD3<Float> in
                    let wp = worldTransform * SIMD4<Float>(p.x, p.y, p.z, 1)
                    return SIMD3<Float>(wp.x, wp.y, wp.z)
                }

                let wp = GLTFWriterMeshPart()
                // SIMD3<Float> has 16-byte stride in memory — must compact to 12-byte packed float3.
                wp.positionData = packFloat3(worldPositions)
                wp.vertexCount  = UInt(positions.count)

                if let normals = part.normals?.elements, normals.count == positions.count {
                    let worldNormals = normals.map { normalize(nm * $0) }
                    wp.normalData = packFloat3(worldNormals)
                }

                if let uvs = part.textureCoordinates?.elements, uvs.count == positions.count {
                    // Flip V: RealityKit uses top-left UV origin; glTF uses bottom-left.
                    // SIMD2<Float> is 8 bytes (no padding), so withUnsafeBytes is safe.
                    let flipped = uvs.map { SIMD2<Float>($0.x, 1 - $0.y) }
                    wp.texcoordData = flipped.withUnsafeBytes { Data($0) }
                }

                let indices = indexBuf.elements
                wp.indexData  = indices.withUnsafeBytes { Data($0) }
                wp.indexCount = UInt(indices.count)

                let matIdx = Int(part.materialIndex)
                let rkMat  = (matIdx < rkMaterials.count ? rkMaterials[matIdx] : rkMaterials.first) as? PhysicallyBasedMaterial
                wp.materialIndex = try resolveOrCreate(material: rkMat, list: &materialList, cache: &cache)

                parts.append(wp)
            }
        }
        return parts
    }

    // MARK: - Materials

    private func resolveOrCreate(
        material pbr: PhysicallyBasedMaterial?,
        list: inout [GLTFWriterMaterial],
        cache: inout [ObjectIdentifier: Int]
    ) throws -> Int {
        // No material → default material at index 0
        guard let pbr = pbr else {
            let mat = GLTFWriterMaterial()
            let idx = list.count
            list.append(mat)
            return idx
        }

        // Deduplication not reliable across separate model components, so we
        // always create a new entry per usage (avoids ObjectIdentifier hazards).
        let mat = GLTFWriterMaterial()

        // Base color tint
        var r: CGFloat = 1, g: CGFloat = 1, b: CGFloat = 1, a: CGFloat = 1
        #if os(macOS)
        (pbr.baseColor.tint.usingColorSpace(.sRGB) ?? pbr.baseColor.tint)
            .getRed(&r, green: &g, blue: &b, alpha: &a)
        #else
        pbr.baseColor.tint.getRed(&r, green: &g, blue: &b, alpha: &a)
        #endif
        mat.baseColorFactor = SIMD4<Float>(Float(r), Float(g), Float(b), Float(a))
        mat.metallicFactor  = pbr.metallic.scale
        mat.roughnessFactor = pbr.roughness.scale

        var er: CGFloat = 0, eg: CGFloat = 0, eb: CGFloat = 0
        #if os(macOS)
        (pbr.emissiveColor.color.usingColorSpace(.sRGB) ?? pbr.emissiveColor.color)
            .getRed(&er, green: &eg, blue: &eb, alpha: nil)
        #else
        pbr.emissiveColor.color.getRed(&er, green: &eg, blue: &eb, alpha: nil)
        #endif
        mat.emissiveFactor = SIMD3<Float>(Float(er), Float(eg), Float(eb))

        if case .transparent(let o) = pbr.blending {
            mat.baseColorFactor.w = o.scale
            mat.alphaMode = .blend
        }
        mat.isDoubleSided = pbr.faceCulling == .none

        // Textures
        mat.baseColorTexturePNG          = try encodePNG(pbr.baseColor.texture?.resource)
        mat.normalTexturePNG             = try encodePNG(pbr.normal.texture?.resource)
        mat.emissiveTexturePNG           = try encodePNG(pbr.emissiveColor.texture?.resource)
        mat.occlusionTexturePNG          = try encodePNG(pbr.ambientOcclusion.texture?.resource)
        // Pack metallic (B) and roughness (G) into one combined texture.
        mat.metallicRoughnessTexturePNG  = try encodeMetallicRoughnessPNG(
            metallicResource: pbr.metallic.texture?.resource,
            roughnessResource: pbr.roughness.texture?.resource
        )

        let idx = list.count
        list.append(mat)
        return idx
    }

    // MARK: - Lights

    private func extractLight(from entity: Entity) -> GLTFWriterLight? {
        if let comp = entity.components[PointLightComponent.self] {
            let l = GLTFWriterLight()
            l.type = .point
            l.color = simd_float3(from: comp.color)
            l.intensity = comp.intensity
            l.range     = comp.attenuationRadius
            return l
        }
        if let comp = entity.components[DirectionalLightComponent.self] {
            let l = GLTFWriterLight()
            l.type = .directional
            l.color = simd_float3(from: comp.color)
            l.intensity = comp.intensity
            return l
        }
        if let comp = entity.components[SpotLightComponent.self] {
            let l = GLTFWriterLight()
            l.type = .spot
            l.color = simd_float3(from: comp.color)
            l.intensity       = comp.intensity
            l.range           = comp.attenuationRadius
            l.innerConeAngle  = Float(comp.innerAngleInDegrees) * .pi / 180
            l.outerConeAngle  = Float(comp.outerAngleInDegrees) * .pi / 180
            return l
        }
        return nil
    }

    // MARK: - Texture encoding

    /// Copies a TextureResource to CPU memory and encodes it as PNG data.
    private func encodePNG(_ resource: TextureResource?) throws -> Data? {
        guard let resource = resource else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GLTFExportError.metalDeviceUnavailable
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: resource.width,
            height: resource.height,
            mipmapped: false
        )
        desc.usage       = .shaderWrite
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw GLTFExportError.textureReadbackFailed
        }
        try resource.copy(to: tex)

        #if os(macOS)
        if tex.storageMode == .managed {
            guard let q   = device.makeCommandQueue(),
                  let cmd = q.makeCommandBuffer(),
                  let enc = cmd.makeBlitCommandEncoder()
            else { throw GLTFExportError.textureReadbackFailed }
            enc.synchronize(resource: tex)
            enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        }
        #endif

        let bpr  = 4 * resource.width
        var bytes = [UInt8](repeating: 0, count: resource.height * bpr)
        bytes.withUnsafeMutableBytes { ptr in
            tex.getBytes(ptr.baseAddress!,
                         bytesPerRow: bpr,
                         from: MTLRegion(origin: .init(), size: MTLSize(width: resource.width, height: resource.height, depth: 1)),
                         mipmapLevel: 0)
        }
        return makePNG(bytes: bytes, width: resource.width, height: resource.height, bpr: bpr)
    }

    /// Build a combined ORM-style PNG: roughness in G, metallic in B.
    /// If only one resource is present the other channel is filled with 1.0.
    private func encodeMetallicRoughnessPNG(
        metallicResource: TextureResource?,
        roughnessResource: TextureResource?
    ) throws -> Data? {
        guard metallicResource != nil || roughnessResource != nil else { return nil }

        // Readback both (nil → filled with 255)
        let metallicBytes  = metallicResource  != nil ? try readbackBytes(metallicResource!)  : nil
        let roughnessBytes = roughnessResource != nil ? try readbackBytes(roughnessResource!) : nil

        let w = (metallicResource ?? roughnessResource)!.width
        let h = (metallicResource ?? roughnessResource)!.height
        let pixelCount = w * h
        var combined = [UInt8](repeating: 255, count: pixelCount * 4)

        for i in 0..<pixelCount {
            // glTF spec: B = metallic, G = roughness (R and A unused)
            combined[i * 4 + 2] = metallicBytes?[i * 4 + 0]  ?? 255 // B = metallic (from R)
            combined[i * 4 + 1] = roughnessBytes?[i * 4 + 0] ?? 255 // G = roughness (from R)
        }
        return makePNG(bytes: combined, width: w, height: h, bpr: w * 4)
    }

    private func readbackBytes(_ resource: TextureResource) throws -> [UInt8] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GLTFExportError.metalDeviceUnavailable
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: resource.width, height: resource.height, mipmapped: false)
        desc.usage = .shaderWrite; desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw GLTFExportError.textureReadbackFailed
        }
        try resource.copy(to: tex)
        #if os(macOS)
        if tex.storageMode == .managed {
            guard let q   = device.makeCommandQueue(),
                  let cmd = q.makeCommandBuffer(),
                  let enc = cmd.makeBlitCommandEncoder()
            else { throw GLTFExportError.textureReadbackFailed }
            enc.synchronize(resource: tex); enc.endEncoding()
            cmd.commit(); cmd.waitUntilCompleted()
        }
        #endif
        let bpr = 4 * resource.width
        var bytes = [UInt8](repeating: 0, count: resource.height * bpr)
        bytes.withUnsafeMutableBytes { ptr in
            tex.getBytes(ptr.baseAddress!, bytesPerRow: bpr,
                from: MTLRegion(origin: .init(), size: MTLSize(width: resource.width, height: resource.height, depth: 1)),
                mipmapLevel: 0)
        }
        return bytes
    }

    private func makePNG(bytes: [UInt8], width: Int, height: Int, bpr: Int) -> Data? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage  = CGImage(
                  width: width, height: height,
                  bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bpr,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }

        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data as CFMutableData, "public.png" as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pack [SIMD3<Float>] into tightly-packed 12-byte float3 data (no SIMD alignment padding).
private func packFloat3(_ values: [SIMD3<Float>]) -> Data {
    var data = Data(capacity: values.count * 12)
    for v in values {
        var xyz = (v.x, v.y, v.z)
        withUnsafeBytes(of: &xyz) { data.append(contentsOf: $0) }
    }
    return data
}

private extension simd_float3 {
    /// Extract linear-RGB float3 from a platform colour (NSColor / UIColor).
    init<C>(from platformColor: C) {
        var r: CGFloat = 1, g: CGFloat = 1, b: CGFloat = 1
        #if os(macOS)
        if let nc = platformColor as? NSColor {
            (nc.usingColorSpace(.sRGB) ?? nc).getRed(&r, green: &g, blue: &b, alpha: nil)
        }
        #else
        if let uc = platformColor as? UIColor {
            uc.getRed(&r, green: &g, blue: &b, alpha: nil)
        }
        #endif
        self = SIMD3<Float>(Float(r), Float(g), Float(b))
    }
}

#endif // !tvOS && compiler >= 5.6
