#import <Foundation/Foundation.h>
#import <simd/simd.h>
#import "GLTFTypes.h"

NS_ASSUME_NONNULL_BEGIN

// ---------------------------------------------------------------------------
// Data-transfer objects — consumed by GLTFAssetWriter to produce a GLB file.
// ---------------------------------------------------------------------------

@interface GLTFWriterMeshPart : NSObject
/// Packed float3 array (3 × float per vertex, no padding).
@property (nonatomic, strong) NSData *positionData;
/// Packed float3 array; nil if not present.
@property (nonatomic, nullable, strong) NSData *normalData;
/// Packed float2 array; nil if not present.
@property (nonatomic, nullable, strong) NSData *texcoordData;
/// Packed uint32 array — triangle indices.
@property (nonatomic, strong) NSData *indexData;
@property (nonatomic, assign) NSUInteger vertexCount;
@property (nonatomic, assign) NSUInteger indexCount;
/// Index into the materials array passed to GLTFAssetWriter.
@property (nonatomic, assign) NSInteger materialIndex;
@end

@interface GLTFWriterMaterial : NSObject
@property (nonatomic, assign) simd_float4 baseColorFactor;    // rgba, linear
@property (nonatomic, assign) float metallicFactor;
@property (nonatomic, assign) float roughnessFactor;
@property (nonatomic, assign) simd_float3 emissiveFactor;     // linear rgb
@property (nonatomic, assign) float alphaCutoff;
@property (nonatomic, assign) GLTFAlphaMode alphaMode;
@property (nonatomic, assign) BOOL isDoubleSided;
/// PNG-encoded image bytes; nil → no texture for that slot.
@property (nonatomic, nullable, strong) NSData *baseColorTexturePNG;
@property (nonatomic, nullable, strong) NSData *normalTexturePNG;
/// Combined metallic (B channel) + roughness (G channel) PNG.
@property (nonatomic, nullable, strong) NSData *metallicRoughnessTexturePNG;
@property (nonatomic, nullable, strong) NSData *emissiveTexturePNG;
@property (nonatomic, nullable, strong) NSData *occlusionTexturePNG;
@end

@interface GLTFWriterLight : NSObject
@property (nonatomic, assign) GLTFLightType type;
@property (nonatomic, nullable, strong) NSString *name;
@property (nonatomic, assign) simd_float3 color;    // linear rgb, 0–1
@property (nonatomic, assign) float intensity;       // lux / lm/m2 / lm/sr
@property (nonatomic, assign) float range;           // 0 = infinite
@property (nonatomic, assign) float innerConeAngle;  // radians, spot only
@property (nonatomic, assign) float outerConeAngle;  // radians, spot only
@end

@interface GLTFWriterNode : NSObject
@property (nonatomic, nullable, strong) NSString *name;
@property (nonatomic, assign) simd_float3 translation;
@property (nonatomic, assign) simd_quatf rotation;
@property (nonatomic, assign) simd_float3 scale;
/// Mesh parts that belong to this node. nil or empty = no mesh on this node.
@property (nonatomic, nullable, strong) NSArray<GLTFWriterMeshPart *> *meshParts;
/// Nil = no light.
@property (nonatomic, nullable, strong) GLTFWriterLight *light;
@property (nonatomic, strong) NSArray<GLTFWriterNode *> *children;
@end

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

@interface GLTFAssetWriter : NSObject

/// Returns GLB binary data representing the scene, or nil on error.
+ (nullable NSData *)glbDataForRootNodes:(NSArray<GLTFWriterNode *> *)rootNodes
                               materials:(NSArray<GLTFWriterMaterial *> *)materials
                                   error:(NSError *_Nullable *_Nullable)outError;

/// Writes GLB to the file at `url`. Returns YES on success.
+ (BOOL)writeGLBToURL:(NSURL *)url
            rootNodes:(NSArray<GLTFWriterNode *> *)rootNodes
            materials:(NSArray<GLTFWriterMaterial *> *)materials
                error:(NSError *_Nullable *_Nullable)outError;

@end

NS_ASSUME_NONNULL_END
