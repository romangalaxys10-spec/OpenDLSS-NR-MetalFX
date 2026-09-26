import {BackSide, Camera, Color, DataTexture, FloatType, FrontSide, GLSL3, InstancedMesh,
    Matrix4, Mesh, MeshDepthMaterial, NearestFilter, NoBlending, RedFormat, RGBAFormat,
    Scene, ShaderMaterial, Side, Vector2, Vector3, Vector4, WebGLRenderTarget} from 'three'
import {FullScreenQuad} from 'three/examples/jsm/postprocessing/Pass'
import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {GBufferPlugin} from 'webgi/plugins/GBufferPlugin'
import {setThreeRendererMode} from 'webgi/helpers/threejs/threeUtils'
import {scatterBounds} from './nr-culled-instances'

export const vegetationQuadVertex = `void main() { gl_Position = vec4(position.xy, 0., 1.); }`
export const vegetationReduceDepth = `
uniform sampler2D sourceDepth;
uniform vec4 sourceRect;
uniform vec2 destinationOrigin;
out vec4 result;
float fetchDepth(ivec2 p) {
    if (any(greaterThanEqual(p, ivec2(sourceRect.zw)))) return 1.;
    return texelFetch(sourceDepth, p + ivec2(sourceRect.xy), 0).r;
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy - destinationOrigin) * 2;
    float d = max(max(fetchDepth(p), fetchDepth(p+ivec2(1,0))),
                  max(fetchDepth(p+ivec2(0,1)), fetchDepth(p+ivec2(1,1))));
    result = vec4(d, 0., 0., 1.);
}`
export const vegetationTestBounds = `
uniform sampler2D boundsTexture, depth0, depthA, depthB;
uniform mat4 viewProjectionBounds;
uniform vec4 levels[16];
uniform vec2 screenSize;
uniform float rasterPadding;
uniform int boundsWidth, outputWidth, instanceCount, lastLevel;
out vec4 result;
float depthAt(ivec2 p, int level) {
    vec4 rect = levels[level];
    if (any(lessThan(p,ivec2(0))) || any(greaterThanEqual(p,ivec2(rect.zw)))) return 1.;
    p += ivec2(rect.xy);
    if (level == 0) return texelFetch(depth0,p,0).r;
    if ((level & 1) == 1) return texelFetch(depthA,p,0).r;
    return texelFetch(depthB,p,0).r;
}
float visible(int index) {
    if (index >= instanceCount) return 1.;
    ivec2 address = ivec2((index*4) % boundsWidth,(index*4) / boundsWidth);
    vec4 center = texelFetch(boundsTexture,address,0);
    vec4 clip = viewProjectionBounds * vec4(center.xyz,1.);
    vec4 axisX = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(1,0),0).xyz,0.);
    vec4 axisY = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(2,0),0).xyz,0.);
    vec4 axisZ = viewProjectionBounds * vec4(texelFetch(boundsTexture,address+ivec2(3,0),0).xyz,0.);
    vec4 pad = (abs(viewProjectionBounds[0])+abs(viewProjectionBounds[1])+abs(viewProjectionBounds[2])) * center.w;
    vec4 extent = abs(axisX)+abs(axisY)+abs(axisZ)+pad;
    vec4 lo = clip-extent, hi = clip+extent;
    if (hi.w <= 0. || hi.z < -hi.w || lo.z > hi.w ||
        hi.x < -hi.w || lo.x > hi.w || hi.y < -hi.w || lo.y > hi.w) return 0.;
    if (lo.w <= 0. || lo.z <= -hi.w) return 1.;
    vec3 ndcLo = vec3(1e30), ndcHi = vec3(-1e30);
    // Project the transformed geometry box, with explicit float-rounding padding.
    for (int corner=0; corner<8; corner++) {
        vec3 signCorner = vec3((corner&1)==0 ? -1. : 1., (corner&2)==0 ? -1. : 1., (corner&4)==0 ? -1. : 1.);
        vec4 p = clip + signCorner.x*axisX + signCorner.y*axisY + signCorner.z*axisZ;
        vec4 a = p-pad, b = p+pad;
        ndcLo = min(ndcLo,min(min(a.xyz/a.w,a.xyz/b.w),min(b.xyz/a.w,b.xyz/b.w)));
        ndcHi = max(ndcHi,max(max(a.xyz/a.w,a.xyz/b.w),max(b.xyz/a.w,b.xyz/b.w)));
    }
    // Single-sample rasterization tests pixel centers. Include two units of
    // the implementation's reported subpixel precision around those centers.
    vec2 a = ceil((ndcLo.xy*.5+.5)*screenSize-.5-rasterPadding);
    vec2 b = floor((ndcHi.xy*.5+.5)*screenSize-.5+rasterPadding);
    if (any(greaterThan(a,b))) return 0.;
    if (any(lessThan(b,vec2(0.))) || any(greaterThanEqual(a,screenSize))) return 0.;
    a = max(a,vec2(0.)); b = min(b,screenSize-1.);
    int level = min(lastLevel,int(ceil(log2(max(1.,max(b.x-a.x,b.y-a.y)+1.)))));
    float scale = exp2(float(level));
    ivec2 p = ivec2(floor(a/scale)), q = ivec2(floor(b/scale));
    float farthest = max(max(depthAt(p,level),depthAt(ivec2(q.x,p.y),level)),
                         max(depthAt(ivec2(p.x,q.y),level),depthAt(q,level)));
    float nearest = ndcLo.z*.5+.5 - .000002;
    if (nearest > farthest) return 0.;
    // Refine the rectangle, not the geometry. Four coarse texels can include
    // a large amount of sky outside a thin plant's actual screen bounds.
    level = max(0,level-2); scale = exp2(float(level));
    p = ivec2(floor(a/scale)); q = ivec2(floor(b/scale));
    for (int y=0; y<5; y++) for (int x=0; x<5; x++) {
        ivec2 cell = p+ivec2(x,y);
        if (any(greaterThan(cell,q))) continue;
        if (nearest <= depthAt(cell,level)) return 1.;
    }
    return 0.;
}
void main() {
    int index = (int(gl_FragCoord.y)*outputWidth+int(gl_FragCoord.x))*4;
    result = vec4(visible(index),visible(index+1),visible(index+2),visible(index+3));
}`

/** Conservative opaque-depth culling. Geometry, materials and shadow fidelity are unchanged. */
export function createVegetationOcclusion(viewer: ViewerApp) {
    const renderer = viewer.renderer.rendererObject, gl = renderer.getContext() as WebGL2RenderingContext
    if (!renderer.capabilities.isWebGL2 || viewer.isAntialiased || renderer.capabilities.logarithmicDepthBuffer ||
        !gl.getExtension('EXT_color_buffer_float')) return undefined
    const mainOccluders = new Scene(), shadowOccluders = new Scene()
    const materials = new Map<number, MeshDepthMaterial>()
    const entries: {mesh: InstancedMesh; offset: number; count: number}[] = []
    let count = 0, occluderTriangles = 0
    const materialFor = (source: any, side: number) => {
        const key = source.id * 3 + side
        let material = materials.get(key)
        if (!material) {
            // The G-buffer discards empty map texels even on opaque materials.
            // Respect that coverage when deciding what can hide its geometry.
            material = new MeshDepthMaterial({side: side as Side, colorWrite: false,
                map: source.map, alphaTest: Math.max(.001, source.alphaTest)})
            materials.set(key, material)
        }
        return material
    }
    viewer.scene.modelRoot.updateWorldMatrix(true, true)
    viewer.scene.modelRoot.traverseVisible(object => {
        const mesh = object as Mesh
        if (!mesh.isMesh) return
        const scatter = scatterBounds.get(mesh as InstancedMesh)
        if (scatter) {
            const length = scatter.centers.length / 4
            entries.push({mesh: mesh as InstancedMesh, offset: count, count: length}); count += length
            return
        }
        if ((mesh as InstancedMesh).isInstancedMesh || Array.isArray(mesh.material)) return
        const m = mesh.material as any
        if (!m.visible || m.polygonOffset || m.userData.renderToDepth === false || m.userData.pluginsDisabled ||
            m.transparent || m.transmission || m.alphaHash || m.alphaTest || m.alphaMap || m.displacementMap ||
            m.depthWrite === false || m.clippingPlanes?.length || (mesh as any).isSkinnedMesh ||
            mesh.morphTargetInfluences?.length || mesh.customDepthMaterial) return
        const add = (scene: Scene, side: number) => {
            const proxy = new Mesh(mesh.geometry, materialFor(m, side))
            proxy.matrix.copy(mesh.matrixWorld); proxy.matrixAutoUpdate = false
            scene.add(proxy)
        }
        add(mainOccluders, m.side)
        if (mesh.castShadow) add(shadowOccluders,
            m.shadowSide ?? (m.side === FrontSide ? BackSide : m.side === BackSide ? FrontSide : m.side))
        occluderTriangles += (mesh.geometry.index?.count ?? 0) / 3
    })
    if (!count) return undefined
    const width = Math.min(renderer.capabilities.maxTextureSize, Math.max(2048, 2 ** Math.ceil(Math.log2(Math.sqrt(count * 4)))))
    const height = Math.ceil(count * 4 / width)
    if (height > renderer.capabilities.maxTextureSize) { materials.forEach(m => m.dispose()); return undefined }
    const bounds = new Float32Array(width * height * 4), point = new Vector3(), half = new Vector3(), center = new Vector3()
    const transform = new Matrix4()
    for (const entry of entries) {
        const mesh = entry.mesh, matrices = mesh.instanceMatrix.array
        mesh.geometry.computeBoundingBox()
        mesh.geometry.boundingBox!.getCenter(center)
        mesh.geometry.boundingBox!.getSize(half).multiplyScalar(.5)
        for (let i = 0; i < entry.count; i++) {
            transform.fromArray(matrices, i * 16).premultiply(mesh.matrixWorld)
            point.copy(center).applyMatrix4(transform)
            const k = (entry.offset + i) * 16, e = transform.elements
            bounds[k] = point.x; bounds[k + 1] = point.y; bounds[k + 2] = point.z
            bounds[k + 3] = .0001 + Math.max(Math.abs(point.x), Math.abs(point.y), Math.abs(point.z)) * 1e-6
            for (let j = 0; j < 3; j++) {
                bounds[k + 4 + j] = e[j] * half.x
                bounds[k + 8 + j] = e[4 + j] * half.y
                bounds[k + 12 + j] = e[8 + j] * half.z
            }
        }
    }
    const boundsTexture = new DataTexture(bounds, width, height, RGBAFormat, FloatType)
    boundsTexture.needsUpdate = true
    const maskWidth = width, maskHeight = Math.ceil(count / 4 / maskWidth)
    const makeTarget = (w: number, h: number, depth = false, float = false) => {
        const target = new WebGLRenderTarget(w, h, {depthBuffer: depth, format: float || depth ? RedFormat : RGBAFormat,
            type: float ? FloatType : undefined, minFilter: NearestFilter, magFilter: NearestFilter})
        return target
    }
    const maskTarget = makeTarget(maskWidth, maskHeight)
    const reduce = new ShaderMaterial({glslVersion: GLSL3, vertexShader: vegetationQuadVertex, fragmentShader: vegetationReduceDepth,
        depthTest: false, depthWrite: false, blending: NoBlending, uniforms: {
            sourceDepth: {value: null}, sourceRect: {value: new Vector4()}, destinationOrigin: {value: new Vector2()},
        }})
    const levels = Array.from({length: 16}, () => new Vector4())
    const test = new ShaderMaterial({glslVersion: GLSL3, vertexShader: vegetationQuadVertex, fragmentShader: vegetationTestBounds,
        depthTest: false, depthWrite: false, blending: NoBlending, uniforms: {
            boundsTexture: {value: boundsTexture}, depth0: {value: null}, depthA: {value: null}, depthB: {value: null},
            viewProjectionBounds: {value: new Matrix4()}, levels: {value: levels},
            screenSize: {value: new Vector2()}, boundsWidth: {value: width}, outputWidth: {value: maskWidth},
            rasterPadding: {value: 2 / 2 ** gl.getParameter(gl.SUBPIXEL_BITS)},
            instanceCount: {value: count}, lastLevel: {value: 0},
        }})
    const quad = new FullScreenQuad(reduce)
    const resources = new Map<string, {depthTarget: WebGLRenderTarget; atlas: WebGLRenderTarget[]}>()
    let revision = 0
    const cameras = new Map<Camera, {matrix: Matrix4; width: number; height: number; bytes: Uint8Array}>()
    const projection = new Matrix4(), previousColor = new Color(), scissor = new Vector4()
    const stats = {occluderTriangles, totalInstances: count, visibleInstances: 0, milliseconds: 0, updates: 0}
    const clearMasks = () => {
        for (const camera of cameras.keys()) for (const entry of entries) scatterBounds.get(entry.mesh)!.masks.delete(camera)
        cameras.clear()
    }
    const update = (camera: Camera, w: number, h: number, shadow = false) => {
        camera.updateMatrixWorld(true)
        projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
        let cache = cameras.get(camera)
        if (cache && cache.width === w && cache.height === h && cache.matrix.equals(projection)) return
        if (!cache) {
            cache = {matrix: new Matrix4(), width: w, height: h, bytes: new Uint8Array(maskWidth * maskHeight * 4)}
            cameras.set(camera, cache)
        }
        cache.matrix.copy(projection); cache.width = w; cache.height = h
        const key = shadow ? 'shadow' : 'camera'
        let targets = resources.get(key)
        if (targets && (targets.depthTarget.width !== w || targets.depthTarget.height !== h)) {
            targets.depthTarget.dispose(); targets.atlas.forEach(t => t.dispose()); targets = undefined
        }
        if (!targets) {
            const depthTarget = viewer.renderer.createTarget({size: {width: w, height: h}, depthBuffer: true,
                depthTexture: true, format: RedFormat, minFilter: NearestFilter, magFilter: NearestFilter}, false) as WebGLRenderTarget
            const atlas = [makeTarget(Math.ceil(w / 2), h, false, true), makeTarget(Math.ceil(w / 2), h, false, true)]
            targets = {depthTarget, atlas}; resources.set(key, targets)
        }
        const {depthTarget, atlas} = targets
        const start = performance.now(), oldTarget = renderer.getRenderTarget(), autoClear = renderer.autoClear
        const alpha = renderer.getClearAlpha(), scissorTest = renderer.getScissorTest()
        const userData = (renderer as any).userData
        const modes = {shadowMapRender: userData.shadowMapRender, backgroundRender: userData.backgroundRender,
            opaqueRender: userData.opaqueRender, transparentRender: userData.transparentRender,
            transmissionRender: userData.transmissionRender, mainRenderPass: userData.mainRenderPass}
        renderer.getClearColor(previousColor); renderer.getScissor(scissor)
        try {
            renderer.autoClear = false; renderer.setScissorTest(false)
            setThreeRendererMode(renderer, {shadowMapRender: false, backgroundRender: false, opaqueRender: true,
                transparentRender: false, transmissionRender: false, mainRenderPass: false}, () => {
                renderer.setRenderTarget(depthTarget)
                renderer.clear(false, true, false)
                renderer.render(shadow ? shadowOccluders : mainOccluders, camera)
                levels[0].set(0, 0, w, h)
                let lw = w, lh = h, level = 0
                const offsets = [0, 0]
                quad.material = reduce
                while ((lw > 1 || lh > 1) && level < 15) {
                    level++
                    const index = (level - 1) % 2, nw = Math.ceil(lw / 2), nh = Math.ceil(lh / 2), y = offsets[index]
                    reduce.uniforms.sourceDepth.value = level === 1 ? depthTarget!.depthTexture : atlas[1 - index].texture
                    reduce.uniforms.sourceRect.value.copy(levels[level - 1])
                    reduce.uniforms.destinationOrigin.value.set(0, y)
                    levels[level].set(0, y, nw, nh)
                    // Render-target viewports are physical pixels; setViewport
                    // would multiply these coordinates by the canvas pixel ratio.
                    atlas[index].viewport.set(0, y, nw, nh)
                    renderer.setRenderTarget(atlas[index])
                    quad.render(renderer)
                    offsets[index] += nh; lw = nw; lh = nh
                }
                test.uniforms.lastLevel.value = level
                test.uniforms.depth0.value = depthTarget!.depthTexture
                test.uniforms.depthA.value = atlas[0].texture; test.uniforms.depthB.value = atlas[1].texture
                test.uniforms.screenSize.value.set(w, h)
                test.uniforms.viewProjectionBounds.value.copy(cache!.matrix)
                quad.material = test
                renderer.setRenderTarget(maskTarget)
                quad.render(renderer)
                renderer.readRenderTargetPixels(maskTarget, 0, 0, maskWidth, maskHeight, cache!.bytes)
                revision++
                let visible = 0
                for (let i = 0; i < count; i++) if (cache!.bytes[i]) visible++
                for (const entry of entries) scatterBounds.get(entry.mesh)!.masks.set(camera,
                    {bytes: cache!.bytes, offset: entry.offset, revision, matrix: cache!.matrix, width: w, height: h})
                stats.visibleInstances = visible
            })
            stats.milliseconds = performance.now() - start; stats.updates++
        } finally {
            Object.assign(userData, modes)
            renderer.setRenderTarget(oldTarget); renderer.setScissor(scissor)
            renderer.setScissorTest(scissorTest); renderer.autoClear = autoClear; renderer.setClearColor(previousColor, alpha)
        }
    }
    const controller = {enabled: true, shadows: true, stats, update, clearMasks,
        probeChunks: async (levels = 3) => {
            const enabled = viewer.renderEnabled
            viewer.renderEnabled = false
            try {
                const camera = viewer.scene.activeCamera.cameraObject
                renderer.getDrawingBufferSize(size)
                update(camera, size.x, size.y)
                const {probeVegetationChunks} = await import('./nr-vegetation-chunk-probe')
                return await probeVegetationChunks(viewer, entries, camera, cameras.get(camera)!,
                    resources.get('camera')!, mainOccluders, levels)
            } finally { viewer.renderEnabled = enabled }
        }, dispose: () => {
        viewer.removeEventListener('preRender', preRender)
        clearMasks(); resources.forEach(({depthTarget, atlas}) => { depthTarget.dispose(); atlas.forEach(t => t.dispose()) })
        maskTarget.dispose()
        boundsTexture.dispose(); reduce.dispose(); test.dispose(); materials.forEach(m => m.dispose())
    }}
    const size = new Vector2()
    const preRender = () => {
        if (!controller.enabled) { if (cameras.size) clearMasks(); return }
        try {
            renderer.getDrawingBufferSize(size)
            const source = viewer.getPlugin(GBufferPlugin)?.getTarget()
            if (source) size.set(source.width, source.height)
            update(viewer.scene.activeCamera.cameraObject, size.x, size.y)
            if (controller.shadows && (renderer.shadowMap.autoUpdate || renderer.shadowMap.needsUpdate)) {
                viewer.scene.traverseVisible((object: any) => {
                    if (!object.isDirectionalLight || !object.castShadow) return
                    object.shadow.updateMatrices(object)
                    update(object.shadow.camera, object.shadow.mapSize.x, object.shadow.mapSize.y, true)
                })
            }
        } catch (error) {
            controller.enabled = false; clearMasks()
            console.warn('Vegetation occlusion unavailable; retaining all instances', error)
        }
    }
    viewer.addEventListener('preRender', preRender)
    return controller
}
