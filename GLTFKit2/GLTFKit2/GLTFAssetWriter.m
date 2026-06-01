#import "GLTFAssetWriter.h"

// Activate the cgltf writer implementation exactly once in this translation unit.
#define CGLTF_WRITE_IMPLEMENTATION
#include "cgltf_write.h"

// ---------------------------------------------------------------------------
// GLTFWriterMeshPart
// ---------------------------------------------------------------------------

@implementation GLTFWriterMeshPart
- (instancetype)init {
    if ((self = [super init])) {
        _materialIndex = -1;
    }
    return self;
}
@end

// ---------------------------------------------------------------------------
// GLTFWriterMaterial
// ---------------------------------------------------------------------------

@implementation GLTFWriterMaterial
- (instancetype)init {
    if ((self = [super init])) {
        _baseColorFactor = simd_make_float4(1, 1, 1, 1);
        _metallicFactor  = 1.0f;
        _roughnessFactor = 1.0f;
        _emissiveFactor  = simd_make_float3(0, 0, 0);
        _alphaCutoff     = 0.5f;
        _alphaMode       = GLTFAlphaModeOpaque;
    }
    return self;
}
@end

// ---------------------------------------------------------------------------
// GLTFWriterLight
// ---------------------------------------------------------------------------

@implementation GLTFWriterLight
- (instancetype)init {
    if ((self = [super init])) {
        _type             = GLTFLightTypePoint;
        _color            = simd_make_float3(1, 1, 1);
        _intensity        = 1.0f;
        _range            = 0.0f;
        _innerConeAngle   = 0.0f;
        _outerConeAngle   = (float)M_PI / 4.0f;
    }
    return self;
}
@end

// ---------------------------------------------------------------------------
// GLTFWriterNode
// ---------------------------------------------------------------------------

@implementation GLTFWriterNode
- (instancetype)init {
    if ((self = [super init])) {
        _rotation = simd_quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        _scale    = simd_make_float3(1, 1, 1);
        _children = @[];
    }
    return self;
}
@end

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void padTo4Bytes(NSMutableData *buf) {
    NSUInteger rem = buf.length % 4;
    if (rem > 0) {
        uint8_t zeros[3] = {0, 0, 0};
        [buf appendBytes:zeros length:4 - rem];
    }
}

static void flattenNodes(GLTFWriterNode *node, NSMutableArray<GLTFWriterNode *> *out) {
    [out addObject:node];
    for (GLTFWriterNode *child in node.children) {
        flattenNodes(child, out);
    }
}

// ---------------------------------------------------------------------------
// GLTFAssetWriter
// ---------------------------------------------------------------------------

@implementation GLTFAssetWriter

+ (nullable NSData *)glbDataForRootNodes:(NSArray<GLTFWriterNode *> *)rootNodes
                               materials:(NSArray<GLTFWriterMaterial *> *)materials
                                   error:(NSError **)outError
{
    NSString *tmpPath = [NSTemporaryDirectory()
        stringByAppendingPathComponent:[[NSUUID UUID].UUIDString stringByAppendingPathExtension:@"glb"]];
    NSURL *tmpURL = [NSURL fileURLWithPath:tmpPath];

    if (![self writeGLBToURL:tmpURL rootNodes:rootNodes materials:materials error:outError]) {
        return nil;
    }
    NSData *result = [NSData dataWithContentsOfURL:tmpURL options:0 error:outError];
    [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];
    return result;
}

+ (BOOL)writeGLBToURL:(NSURL *)url
            rootNodes:(NSArray<GLTFWriterNode *> *)rootNodes
            materials:(NSArray<GLTFWriterMaterial *> *)materials
                error:(NSError **)outError
{
    // -----------------------------------------------------------------------
    // 1. Flatten the node tree into a linear array.
    // -----------------------------------------------------------------------
    NSMutableArray<GLTFWriterNode *> *flatNodes = [NSMutableArray array];
    for (GLTFWriterNode *root in rootNodes) {
        flattenNodes(root, flatNodes);
    }
    NSUInteger nodeCount = flatNodes.count;

    // -----------------------------------------------------------------------
    // 2. Collect all mesh parts in tree order.
    // -----------------------------------------------------------------------
    NSMutableArray<GLTFWriterMeshPart *> *allParts = [NSMutableArray array];
    for (GLTFWriterNode *node in flatNodes) {
        for (GLTFWriterMeshPart *part in node.meshParts) {
            [allParts addObject:part];
        }
    }
    NSUInteger totalParts = allParts.count;

    // -----------------------------------------------------------------------
    // 3. Build the binary buffer in this order:
    //    positions → normals → texcoords → indices  (per-part)
    //    then image PNGs (per-material × slot)
    // -----------------------------------------------------------------------
    NSMutableData *binary = [NSMutableData data];

    NSMutableArray<NSNumber *> *positionOffsets = [NSMutableArray array];
    NSMutableArray<NSNumber *> *normalOffsets   = [NSMutableArray array];
    NSMutableArray<NSNumber *> *texcoordOffsets = [NSMutableArray array];
    NSMutableArray<NSNumber *> *indexOffsets    = [NSMutableArray array];

    for (GLTFWriterMeshPart *part in allParts) {
        [positionOffsets addObject:@(binary.length)];
        [binary appendData:part.positionData];
        padTo4Bytes(binary);

        if (part.normalData) {
            [normalOffsets addObject:@(binary.length)];
            [binary appendData:part.normalData];
            padTo4Bytes(binary);
        } else {
            [normalOffsets addObject:@(NSNotFound)];
        }

        if (part.texcoordData) {
            [texcoordOffsets addObject:@(binary.length)];
            [binary appendData:part.texcoordData];
            padTo4Bytes(binary);
        } else {
            [texcoordOffsets addObject:@(NSNotFound)];
        }

        [indexOffsets addObject:@(binary.length)];
        [binary appendData:part.indexData];
        padTo4Bytes(binary);
    }

    // Image slots: 0=baseColor 1=normal 2=metallicRoughness 3=emissive 4=occlusion
    const int kSlots = 5;
    NSUInteger matCount = materials.count;

    typedef struct { NSUInteger offset; NSUInteger size; } ImageEntry;
    ImageEntry *imgEntries = (ImageEntry *)calloc(matCount * kSlots, sizeof(ImageEntry));

    for (NSUInteger mi = 0; mi < matCount; mi++) {
        GLTFWriterMaterial *mat = materials[mi];
        NSData *pngs[kSlots] = {
            mat.baseColorTexturePNG,
            mat.normalTexturePNG,
            mat.metallicRoughnessTexturePNG,
            mat.emissiveTexturePNG,
            mat.occlusionTexturePNG,
        };
        for (int s = 0; s < kSlots; s++) {
            if (!pngs[s]) continue;
            imgEntries[mi * kSlots + s].offset = binary.length;
            imgEntries[mi * kSlots + s].size   = pngs[s].length;
            [binary appendData:pngs[s]];
            padTo4Bytes(binary);
        }
    }
    NSUInteger binaryLength = binary.length;

    // -----------------------------------------------------------------------
    // 4. Count cgltf array sizes.
    // -----------------------------------------------------------------------
    NSUInteger numImages = 0;
    for (NSUInteger mi = 0; mi < matCount; mi++) {
        for (int s = 0; s < kSlots; s++) {
            if (imgEntries[mi * kSlots + s].size > 0) numImages++;
        }
    }

    // Per-part attribute / buffer-view / accessor counts
    NSUInteger totalBVs    = 0;
    NSUInteger totalAccs   = 0;
    NSUInteger totalAttrs  = 0;
    for (GLTFWriterMeshPart *part in allParts) {
        NSUInteger a = 1; // position always present
        if (part.normalData)   a++;
        if (part.texcoordData) a++;
        totalAttrs += a;
        totalBVs   += a + 1; // +1 for indices
        totalAccs  += a + 1;
    }
    totalBVs += numImages; // image buffer views appended at end

    // Count nodes with mesh and nodes with lights
    NSUInteger numMeshes = 0;
    NSUInteger numLights = 0;
    for (GLTFWriterNode *node in flatNodes) {
        if (node.meshParts.count > 0) numMeshes++;
        if (node.light)              numLights++;
    }

    // -----------------------------------------------------------------------
    // 5. Allocate all cgltf arrays.
    // -----------------------------------------------------------------------
    cgltf_buffer      *buffers   = (cgltf_buffer *)calloc(1, sizeof(cgltf_buffer));
    cgltf_buffer_view *bvs       = totalBVs   ? (cgltf_buffer_view *)calloc(totalBVs,   sizeof(cgltf_buffer_view)) : NULL;
    cgltf_accessor    *accs      = totalAccs  ? (cgltf_accessor *)calloc(totalAccs,  sizeof(cgltf_accessor))       : NULL;
    cgltf_attribute   *attrs     = totalAttrs ? (cgltf_attribute *)calloc(totalAttrs, sizeof(cgltf_attribute))     : NULL;
    cgltf_primitive   *prims     = totalParts ? (cgltf_primitive *)calloc(totalParts, sizeof(cgltf_primitive))     : NULL;
    cgltf_mesh        *meshes    = numMeshes  ? (cgltf_mesh *)calloc(numMeshes,  sizeof(cgltf_mesh))               : NULL;
    cgltf_material    *mats      = matCount   ? (cgltf_material *)calloc(matCount,   sizeof(cgltf_material))       : NULL;
    cgltf_image       *images    = numImages  ? (cgltf_image *)calloc(numImages,  sizeof(cgltf_image))             : NULL;
    cgltf_sampler     *samplers  = numImages  ? (cgltf_sampler *)calloc(1, sizeof(cgltf_sampler))                  : NULL;
    cgltf_texture     *textures  = numImages  ? (cgltf_texture *)calloc(numImages,  sizeof(cgltf_texture))         : NULL;
    cgltf_node        *nodes     = nodeCount  ? (cgltf_node *)calloc(nodeCount,   sizeof(cgltf_node))              : NULL;
    cgltf_node       **rootPtrs  = rootNodes.count ? (cgltf_node **)calloc(rootNodes.count, sizeof(cgltf_node *)) : NULL;
    cgltf_scene       *scenes    = (cgltf_scene *)calloc(1, sizeof(cgltf_scene));
    cgltf_light       *lights    = numLights  ? (cgltf_light *)calloc(numLights, sizeof(cgltf_light))              : NULL;

    // Binary buffer data
    void *binaryBytes = binaryLength ? malloc(binaryLength) : NULL;
    if (binaryLength && !binaryBytes) {
        // OOM — free everything and bail
        free(buffers); free(bvs); free(accs); free(attrs); free(prims); free(meshes);
        free(mats); free(images); free(samplers); free(textures); free(nodes); free(rootPtrs);
        free(scenes); free(lights); free(imgEntries);
        if (outError) *outError = [NSError errorWithDomain:NSPOSIXErrorDomain code:ENOMEM userInfo:nil];
        return NO;
    }
    if (binaryBytes) memcpy(binaryBytes, binary.bytes, binaryLength);

    // -----------------------------------------------------------------------
    // 6. Single buffer (embedded in GLB — uri = NULL).
    // -----------------------------------------------------------------------
    buffers[0].size = binaryLength;
    buffers[0].data = binaryBytes;

    // -----------------------------------------------------------------------
    // 7. Default sampler.
    // -----------------------------------------------------------------------
    if (samplers) {
        samplers[0].wrap_s    = 33071; // GL_CLAMP_TO_EDGE
        samplers[0].wrap_t    = 33071;
        samplers[0].min_filter = 9729; // GL_LINEAR
        samplers[0].mag_filter = 9729;
    }

    // -----------------------------------------------------------------------
    // 8. Images, textures; per-material texture-index table.
    // -----------------------------------------------------------------------
    NSUInteger imageBVStart = totalBVs - numImages;
    NSUInteger imageIdx = 0;
    // matTextureIndices[mi * kSlots + s] = texture index, or NSNotFound
    NSUInteger *matTexIdx = (NSUInteger *)malloc(matCount * kSlots * sizeof(NSUInteger));
    for (NSUInteger k = 0; k < matCount * kSlots; k++) matTexIdx[k] = NSNotFound;

    for (NSUInteger mi = 0; mi < matCount; mi++) {
        for (int s = 0; s < kSlots; s++) {
            ImageEntry e = imgEntries[mi * kSlots + s];
            if (e.size == 0) continue;

            NSUInteger bvI = imageBVStart + imageIdx;
            bvs[bvI].buffer = &buffers[0];
            bvs[bvI].offset = e.offset;
            bvs[bvI].size   = e.size;
            bvs[bvI].type   = cgltf_buffer_view_type_invalid;

            images[imageIdx].buffer_view = &bvs[bvI];
            images[imageIdx].mime_type   = "image/png";

            textures[imageIdx].image     = &images[imageIdx];
            textures[imageIdx].sampler   = samplers ? &samplers[0] : NULL;

            matTexIdx[mi * kSlots + s] = imageIdx;
            imageIdx++;
        }
    }

    // -----------------------------------------------------------------------
    // 9. Materials.
    // -----------------------------------------------------------------------
    for (NSUInteger mi = 0; mi < matCount; mi++) {
        GLTFWriterMaterial *wm = materials[mi];
        cgltf_material *cm = &mats[mi];
        cm->has_pbr_metallic_roughness = 1;
        cm->pbr_metallic_roughness.base_color_factor[0] = wm.baseColorFactor.x;
        cm->pbr_metallic_roughness.base_color_factor[1] = wm.baseColorFactor.y;
        cm->pbr_metallic_roughness.base_color_factor[2] = wm.baseColorFactor.z;
        cm->pbr_metallic_roughness.base_color_factor[3] = wm.baseColorFactor.w;
        cm->pbr_metallic_roughness.metallic_factor  = wm.metallicFactor;
        cm->pbr_metallic_roughness.roughness_factor = wm.roughnessFactor;
        cm->emissive_factor[0] = wm.emissiveFactor.x;
        cm->emissive_factor[1] = wm.emissiveFactor.y;
        cm->emissive_factor[2] = wm.emissiveFactor.z;
        cm->double_sided  = wm.isDoubleSided ? 1 : 0;
        cm->alpha_cutoff  = wm.alphaCutoff;
        switch (wm.alphaMode) {
            case GLTFAlphaModeMask:  cm->alpha_mode = cgltf_alpha_mode_mask;  break;
            case GLTFAlphaModeBlend: cm->alpha_mode = cgltf_alpha_mode_blend; break;
            default:                 cm->alpha_mode = cgltf_alpha_mode_opaque; break;
        }

        NSUInteger bcIdx = matTexIdx[mi * kSlots + 0];
        NSUInteger nrIdx = matTexIdx[mi * kSlots + 1];
        NSUInteger mrIdx = matTexIdx[mi * kSlots + 2];
        NSUInteger emIdx = matTexIdx[mi * kSlots + 3];
        NSUInteger aoIdx = matTexIdx[mi * kSlots + 4];

        if (bcIdx != NSNotFound) {
            cm->pbr_metallic_roughness.base_color_texture.texture  = &textures[bcIdx];
            cm->pbr_metallic_roughness.base_color_texture.texcoord = 0;
            cm->pbr_metallic_roughness.base_color_texture.scale    = 1.0f;
        }
        if (mrIdx != NSNotFound) {
            cm->pbr_metallic_roughness.metallic_roughness_texture.texture  = &textures[mrIdx];
            cm->pbr_metallic_roughness.metallic_roughness_texture.texcoord = 0;
            cm->pbr_metallic_roughness.metallic_roughness_texture.scale    = 1.0f;
        }
        if (nrIdx != NSNotFound) {
            cm->normal_texture.texture  = &textures[nrIdx];
            cm->normal_texture.texcoord = 0;
            cm->normal_texture.scale    = 1.0f;
        }
        if (emIdx != NSNotFound) {
            cm->emissive_texture.texture  = &textures[emIdx];
            cm->emissive_texture.texcoord = 0;
            cm->emissive_texture.scale    = 1.0f;
        }
        if (aoIdx != NSNotFound) {
            cm->occlusion_texture.texture  = &textures[aoIdx];
            cm->occlusion_texture.texcoord = 0;
            cm->occlusion_texture.scale    = 1.0f;
        }
    }

    // -----------------------------------------------------------------------
    // 10. Buffer views, accessors, attributes, primitives, meshes.
    // -----------------------------------------------------------------------
    NSUInteger bvIdx   = 0;
    NSUInteger accIdx  = 0;
    NSUInteger attrIdx = 0;
    NSUInteger primIdx = 0;
    NSUInteger meshIdx = 0;
    NSUInteger partIdx = 0;

    // Map node → cgltf_mesh index (NSNotFound = no mesh)
    NSMutableArray<NSNumber *> *nodeMeshIdx = [NSMutableArray array];
    for (NSUInteger ni = 0; ni < nodeCount; ni++) [nodeMeshIdx addObject:@(NSNotFound)];

    for (NSUInteger ni = 0; ni < nodeCount; ni++) {
        GLTFWriterNode *wn = flatNodes[ni];
        if (wn.meshParts.count == 0) continue;

        nodeMeshIdx[ni] = @(meshIdx);
        meshes[meshIdx].primitives       = &prims[primIdx];
        meshes[meshIdx].primitives_count = wn.meshParts.count;

        for (GLTFWriterMeshPart *part in wn.meshParts) {
            NSUInteger vc = part.vertexCount;
            NSUInteger ic = part.indexCount;
            NSUInteger posOff = positionOffsets[partIdx].unsignedIntegerValue;
            NSUInteger nrmOff = normalOffsets[partIdx].unsignedIntegerValue;
            NSUInteger uvOff  = texcoordOffsets[partIdx].unsignedIntegerValue;
            NSUInteger idxOff = indexOffsets[partIdx].unsignedIntegerValue;

            NSUInteger firstAttr = attrIdx;
            NSUInteger numAttrs  = 0;

            // -- POSITION --
            bvs[bvIdx] = (cgltf_buffer_view){
                .buffer = &buffers[0], .offset = posOff,
                .size   = vc * 3 * sizeof(float), .type = cgltf_buffer_view_type_vertices };
            accs[accIdx] = (cgltf_accessor){
                .buffer_view    = &bvs[bvIdx],
                .component_type = cgltf_component_type_r_32f,
                .type           = cgltf_type_vec3,
                .count          = vc };
            // Compute min/max for spec compliance
            const float *pos = (const float *)part.positionData.bytes;
            if (vc > 0 && pos) {
                float mn[3] = {pos[0], pos[1], pos[2]};
                float mx[3] = {pos[0], pos[1], pos[2]};
                for (NSUInteger v = 1; v < vc; v++) {
                    for (int c = 0; c < 3; c++) {
                        if (pos[v*3+c] < mn[c]) mn[c] = pos[v*3+c];
                        if (pos[v*3+c] > mx[c]) mx[c] = pos[v*3+c];
                    }
                }
                accs[accIdx].has_min = accs[accIdx].has_max = 1;
                memcpy(accs[accIdx].min, mn, 3 * sizeof(float));
                memcpy(accs[accIdx].max, mx, 3 * sizeof(float));
            }
            attrs[attrIdx] = (cgltf_attribute){
                .name = "POSITION", .type = cgltf_attribute_type_position,
                .index = 0, .data = &accs[accIdx] };
            bvIdx++; accIdx++; attrIdx++; numAttrs++;

            // -- NORMAL (optional) --
            if (part.normalData && nrmOff != NSNotFound) {
                bvs[bvIdx] = (cgltf_buffer_view){
                    .buffer = &buffers[0], .offset = nrmOff,
                    .size   = vc * 3 * sizeof(float), .type = cgltf_buffer_view_type_vertices };
                accs[accIdx] = (cgltf_accessor){
                    .buffer_view    = &bvs[bvIdx],
                    .component_type = cgltf_component_type_r_32f,
                    .type           = cgltf_type_vec3,
                    .count          = vc };
                attrs[attrIdx] = (cgltf_attribute){
                    .name = "NORMAL", .type = cgltf_attribute_type_normal,
                    .index = 0, .data = &accs[accIdx] };
                bvIdx++; accIdx++; attrIdx++; numAttrs++;
            }

            // -- TEXCOORD_0 (optional) --
            if (part.texcoordData && uvOff != NSNotFound) {
                bvs[bvIdx] = (cgltf_buffer_view){
                    .buffer = &buffers[0], .offset = uvOff,
                    .size   = vc * 2 * sizeof(float), .type = cgltf_buffer_view_type_vertices };
                accs[accIdx] = (cgltf_accessor){
                    .buffer_view    = &bvs[bvIdx],
                    .component_type = cgltf_component_type_r_32f,
                    .type           = cgltf_type_vec2,
                    .count          = vc };
                attrs[attrIdx] = (cgltf_attribute){
                    .name = "TEXCOORD_0", .type = cgltf_attribute_type_texcoord,
                    .index = 0, .data = &accs[accIdx] };
                bvIdx++; accIdx++; attrIdx++; numAttrs++;
            }

            // -- Indices --
            bvs[bvIdx] = (cgltf_buffer_view){
                .buffer = &buffers[0], .offset = idxOff,
                .size   = ic * sizeof(uint32_t), .type = cgltf_buffer_view_type_indices };
            accs[accIdx] = (cgltf_accessor){
                .buffer_view    = &bvs[bvIdx],
                .component_type = cgltf_component_type_r_32u,
                .type           = cgltf_type_scalar,
                .count          = ic };
            bvIdx++; accIdx++;

            // -- Primitive --
            prims[primIdx].type             = cgltf_primitive_type_triangles;
            prims[primIdx].indices          = &accs[accIdx - 1]; // just-set index accessor
            prims[primIdx].attributes       = &attrs[firstAttr];
            prims[primIdx].attributes_count = numAttrs;
            if (part.materialIndex >= 0 && (NSUInteger)part.materialIndex < matCount) {
                prims[primIdx].material = &mats[part.materialIndex];
            }
            primIdx++;
            partIdx++;
        }
        meshIdx++;
    }

    // -----------------------------------------------------------------------
    // 11. Lights.
    // -----------------------------------------------------------------------
    NSUInteger lightIdx = 0;
    NSMutableArray<NSNumber *> *nodeLightIdx = [NSMutableArray array];
    for (NSUInteger ni = 0; ni < nodeCount; ni++) [nodeLightIdx addObject:@(NSNotFound)];

    for (NSUInteger ni = 0; ni < nodeCount; ni++) {
        GLTFWriterLight *wl = flatNodes[ni].light;
        if (!wl) continue;
        cgltf_light *cl = &lights[lightIdx];
        cl->color[0] = wl.color.x; cl->color[1] = wl.color.y; cl->color[2] = wl.color.z;
        cl->intensity = wl.intensity;
        cl->range     = wl.range;
        cl->spot_inner_cone_angle = wl.innerConeAngle;
        cl->spot_outer_cone_angle = wl.outerConeAngle;
        switch (wl.type) {
            case GLTFLightTypePoint:       cl->type = cgltf_light_type_point;       break;
            case GLTFLightTypeSpot:        cl->type = cgltf_light_type_spot;        break;
            case GLTFLightTypeDirectional: cl->type = cgltf_light_type_directional; break;
            default:                       cl->type = cgltf_light_type_point;       break;
        }
        nodeLightIdx[ni] = @(lightIdx);
        lightIdx++;
    }

    // -----------------------------------------------------------------------
    // 12. Nodes — build node map and per-node child pointer arrays.
    // -----------------------------------------------------------------------
    NSMutableDictionary<NSValue *, NSNumber *> *nodeMap = [NSMutableDictionary dictionary];
    for (NSUInteger ni = 0; ni < nodeCount; ni++) {
        nodeMap[[NSValue valueWithPointer:(__bridge void *)flatNodes[ni]]] = @(ni);
    }

    // Allocate per-node child-pointer arrays
    cgltf_node ***childArrays = (cgltf_node ***)calloc(nodeCount, sizeof(cgltf_node **));
    for (NSUInteger ni = 0; ni < nodeCount; ni++) {
        NSArray<GLTFWriterNode *> *children = flatNodes[ni].children;
        NSUInteger cc = children.count;
        if (cc == 0) continue;
        childArrays[ni] = (cgltf_node **)calloc(cc, sizeof(cgltf_node *));
        for (NSUInteger ci = 0; ci < cc; ci++) {
            NSNumber *childNI = nodeMap[[NSValue valueWithPointer:(__bridge void *)children[ci]]];
            if (childNI) childArrays[ni][ci] = &nodes[childNI.unsignedIntegerValue];
        }
    }

    for (NSUInteger ni = 0; ni < nodeCount; ni++) {
        GLTFWriterNode *wn = flatNodes[ni];
        cgltf_node *cn = &nodes[ni];

        if (wn.name.length > 0) cn->name = (char *)wn.name.UTF8String;

        cn->has_translation = 1;
        cn->translation[0] = wn.translation.x;
        cn->translation[1] = wn.translation.y;
        cn->translation[2] = wn.translation.z;

        cn->has_rotation = 1;
        cn->rotation[0]  = wn.rotation.vector.x;
        cn->rotation[1]  = wn.rotation.vector.y;
        cn->rotation[2]  = wn.rotation.vector.z;
        cn->rotation[3]  = wn.rotation.vector.w;

        cn->has_scale = 1;
        cn->scale[0] = wn.scale.x;
        cn->scale[1] = wn.scale.y;
        cn->scale[2] = wn.scale.z;

        NSUInteger mi = nodeMeshIdx[ni].unsignedIntegerValue;
        if (mi != NSNotFound) cn->mesh = &meshes[mi];

        NSUInteger li = nodeLightIdx[ni].unsignedIntegerValue;
        if (li != NSNotFound) cn->light = &lights[li];

        cn->children       = childArrays[ni];
        cn->children_count = flatNodes[ni].children.count;
    }

    // -----------------------------------------------------------------------
    // 13. Scene.
    // -----------------------------------------------------------------------
    for (NSUInteger ri = 0; ri < rootNodes.count; ri++) {
        NSNumber *ni = nodeMap[[NSValue valueWithPointer:(__bridge void *)rootNodes[ri]]];
        rootPtrs[ri] = ni ? &nodes[ni.unsignedIntegerValue] : NULL;
    }
    scenes[0].nodes       = rootPtrs;
    scenes[0].nodes_count = rootNodes.count;

    // -----------------------------------------------------------------------
    // 14. cgltf_data assembly.
    // -----------------------------------------------------------------------
    const char *kLightsPunctualExt = "KHR_lights_punctual";
    const char *extUsed[1]     = { kLightsPunctualExt };
    const char *extRequired[0] = {};

    cgltf_data data;
    memset(&data, 0, sizeof(data));
    data.file_type           = cgltf_file_type_glb;
    data.asset.version       = (char *)"2.0";
    data.asset.generator     = (char *)"GLTFKit2";
    data.buffers             = buffers;     data.buffers_count        = 1;
    data.buffer_views        = bvs;         data.buffer_views_count   = totalBVs;
    data.accessors           = accs;        data.accessors_count       = totalAccs;
    data.meshes              = meshes;      data.meshes_count          = numMeshes;
    data.materials           = mats;        data.materials_count       = matCount;
    data.images              = images;      data.images_count          = numImages;
    data.samplers            = samplers;    data.samplers_count        = numImages ? 1 : 0;
    data.textures            = textures;    data.textures_count        = numImages;
    data.nodes               = nodes;       data.nodes_count           = nodeCount;
    data.scenes              = scenes;      data.scenes_count          = 1;
    data.scene               = &scenes[0];
    data.lights              = lights;      data.lights_count          = numLights;
    // cgltf_write_file (GLB mode) reads data.bin / data.bin_size for the binary
    // chunk — NOT buffers[0].data. Both must be set; without them the BIN chunk
    // is omitted and the GLB is structurally invalid.
    data.bin                 = binaryBytes;
    data.bin_size            = binaryLength;

    if (numLights > 0) {
        data.extensions_used       = (char **)extUsed;
        data.extensions_used_count = 1;
    }

    cgltf_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.type = cgltf_file_type_glb;

    cgltf_result res = cgltf_write_file(&opts, url.fileSystemRepresentation, &data);

    // -----------------------------------------------------------------------
    // 15. Free all C allocations.
    // -----------------------------------------------------------------------
    free(binaryBytes);
    free(buffers); free(bvs); free(accs); free(attrs); free(prims); free(meshes);
    free(mats); free(images); free(samplers); free(textures);
    for (NSUInteger ni = 0; ni < nodeCount; ni++) free(childArrays[ni]);
    free(childArrays);
    free(nodes); free(rootPtrs); free(scenes); free(lights);
    free(imgEntries); free(matTexIdx);

    if (res != cgltf_result_success) {
        if (outError) {
            *outError = [NSError errorWithDomain:@"GLTFKit2"
                                            code:res
                                        userInfo:@{NSLocalizedDescriptionKey:
                            [NSString stringWithFormat:@"cgltf_write_file failed (code %d)", res]}];
        }
        return NO;
    }
    return YES;
}

@end
