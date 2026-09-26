import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {AgXToneMapping, ACESFilmicToneMapping, NeutralToneMapping, Color, FogExp2, Mesh, MeshPhysicalMaterial,
    Vector3, BufferGeometry, BufferAttribute, PropertyBinding} from 'three'
import {DirectionalLight2} from 'webgi/core/threejs/Lights'
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls'
import {GroundPlugin} from 'webgi/plugins/GroundPlugin'
import {SSAOPlugin} from 'webgi/plugins/SSAOPlugin'
import {TonemapPlugin} from 'webgi/plugins/threejs/TonemapPlugin'
import {decodeGroomRings} from './nr-groom-rings'
import {createCulledInstances} from './nr-culled-instances'
import {createVegetationOcclusion} from './nr-vegetation-occlusion'
import {createOpaquePassFusion} from './nr-opaque-pass-fusion'
import {createStaticDrawGroups} from './nr-static-draw-groups'
import {assetUrl} from '../asset-url.js'

export const blendkitScenes: Record<string, string> = {
    'simple-lighting': 'Mustang · Suspended studio',
    'vege-packshot': 'Nature · Botanical packshot',
    arunthayan: 'Arunthayan',
    'selection-five': 'Cowboy Gramps',
    'selection-six': 'Highlands · Golden hour',
}

/** Each scene supplies its source camera, PBR assets and captured studio light. */
export async function loadBlendkitScene(viewer: ViewerApp, id: string, dispose: (fn: () => void) => void) {
    const base = assetUrl(`/scenes/blendkit/${id}/`)
    const response = await fetch(`${base}view.json`)
    if (!response.ok) throw new Error(`${blendkitScenes[id]} assets are unavailable`)
    const view = await response.json()
    await viewer.load(`${base}${view.model || 'scene.glb'}`, {importConfig: false, autoScale: false, autoCenter: false})
    if (view.instances) {
        ;(window as any).dlssLoading?.stage(`Preparing ${blendkitScenes[id]}`, 'Restoring vegetation instances')
        const [metadata, binary] = await Promise.all([
            fetch(`${base}instances.json`).then(r => { if (!r.ok) throw new Error('Missing instance manifest'); return r.json() }),
            fetch(`${base}instances.bin`).then(r => { if (!r.ok) throw new Error('Missing instance transforms'); return r.arrayBuffer() }),
        ])
        viewer.scene.modelRoot.updateWorldMatrix(true, true)
        for (const entry of metadata) {
            const source = viewer.scene.modelRoot.getObjectByName(PropertyBinding.sanitizeNodeName(entry.name))
            if (!source) throw new Error(`Missing instance geometry: ${entry.name}`)
            const matrices = new Float32Array(binary, entry.offset, entry.count * 16)
            const cells = new Map<string, number[]>()
            const size = view.instanceCellSize ?? 1
            for (let i = 0; i < entry.count; i++) {
                const key = `${Math.floor(matrices[i * 16 + 12] / size)},${Math.floor(matrices[i * 16 + 14] / size)}`
                const indices = cells.get(key)
                if (indices) indices.push(i)
                else cells.set(key, [i])
            }
            source.traverse(child => {
                const mesh = child as Mesh
                if (!mesh.isMesh) return
                // glTF primitives can be children of a group. Their local
                // transforms are retained independently of each scatter matrix.
                const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld)
                const batch = createCulledInstances(geometry, mesh.material, matrices,
                    [...cells.values()], viewer.renderer.rendererObject)
                batch.name = entry.name + ' instances'
                viewer.scene.modelRoot.add(batch)
                // Camera and shadow passes each compact their own visible set.
                // All source instances and mesh detail remain available.
                dispose(() => {
                    batch.removeFromParent(); batch.dispose()
                    geometry.dispose()
                })
            })
            source.visible = false
        }
    }
    if (view.groom) {
        ;(window as any).dlssLoading?.stage(`Preparing ${blendkitScenes[id]}`, 'Loading the original hair and beard groom')
        const response = await fetch(base + (typeof view.groom === 'string' ? view.groom : 'groom.json'))
        if (!response.ok) throw new Error('The portrait groom manifest is missing')
        const groom = await response.json()
        for (const [index, part] of groom.entries()) {
            const response = await fetch(base + part.file)
            if (!response.ok || !response.body) throw new Error(`Missing groom: ${part.name}`)
            let loaded = 0, updated = 0
            const total = Number(response.headers.get('content-length')) || 0
            const progress = new TransformStream<Uint8Array, Uint8Array>({transform(chunk, controller) {
                loaded += chunk.byteLength
                const now = performance.now()
                if (now - updated > 100 || loaded === total) {
                    ;(window as any).dlssLoading?.stage(`Loading ${blendkitScenes[id]}`,
                        `Hair detail ${index + 1} of ${groom.length} · ${part.name}`, {loaded, total})
                    updated = now
                }
                controller.enqueue(chunk)
            }})
            const bytes = await new Response(response.body.pipeThrough(progress)
                .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
            const geometry = new BufferGeometry()
            if (part.encoding === 'rings-v2') {
                const decoded = decodeGroomRings(bytes, part)
                geometry.setAttribute('position', new BufferAttribute(decoded.position, 3))
                geometry.setAttribute('normal', new BufferAttribute(decoded.normal, 3))
                geometry.setIndex(new BufferAttribute(decoded.index, 1))
                geometry.setAttribute('color', new BufferAttribute(decoded.color, 4, true))
            } else {
                geometry.setAttribute('position', new BufferAttribute(new Float32Array(bytes, 0, part.vertices * 3), 3))
                geometry.setAttribute('normal', new BufferAttribute(new Float32Array(bytes, part.vertices * 12, part.vertices * 3), 3))
                geometry.setIndex(new BufferAttribute(new Uint32Array(bytes, part.vertices * 24, part.indices), 1))
                if (part.vertexOpacity) geometry.setAttribute('color', new BufferAttribute(
                    new Uint8Array(bytes, part.vertices * 24 + part.indices * 4, part.vertices * 4), 4, true))
            }
            const material = new MeshPhysicalMaterial({color: part.colorLinear ? new Color().fromArray(part.colorLinear) : part.color,
                roughness: .48, metalness: 0, opacity: part.rasterOpacity ?? 1,
                vertexColors: !!part.vertexOpacity, alphaHash: !!part.vertexOpacity})
            const hair = new Mesh(geometry, material)
            hair.name = part.name
            await viewer.getManager()!.addImportedSingle(hair, {autoScale: false, autoCenter: false, addToRoot: true})
        }
    }
    await viewer.setEnvironmentMap(`${base}lighting.hdr`)
    const scene = viewer.scene
    scene.background = new Color(view.background)
    if (view.backgroundEnvironment) scene.background = scene.environment
    scene.backgroundIntensity = view.backgroundIntensity ?? 1
    scene.environmentIntensity = view.environmentIntensity
    if (view.fogDensity) scene.fog = new FogExp2(view.background, view.fogDensity)
    const tonemap = viewer.getPlugin(TonemapPlugin)
    if (tonemap) {
        tonemap.toneMapping = view.toneMapping === 'Neutral' ? NeutralToneMapping : view.toneMapping === 'AgX' ? AgXToneMapping : ACESFilmicToneMapping
        tonemap.exposure = view.exposure
        if (tonemap.config) {
            const previous = tonemap.config.tonemapBackground
            tonemap.config.tonemapBackground = !!view.backgroundEnvironment
            tonemap.config.setDirty()
            dispose(() => {
                if (tonemap.config) {
                    tonemap.config.tonemapBackground = previous
                    tonemap.config.setDirty()
                }
            })
        }
    }
    if (view.sun) {
        const light = new DirectionalLight2(new Color(view.sun.color), view.sun.intensity)
        const target = new Vector3().fromArray(view.camera.target)
        light.position.copy(target).addScaledVector(new Vector3().fromArray(view.sun.direction), 100)
        light.target.position.copy(target).sub(light.position)
        light.castShadow = true
        light.shadow.mapSize.set(4096, 4096)
        const extent = view.sun.shadowExtent ?? 35
        Object.assign(light.shadow.camera, {left: -extent, right: extent, top: extent, bottom: -extent, near: .1, far: 400})
        light.shadow.camera.updateProjectionMatrix()
        light.shadow.normalBias = view.sun.normalBias ?? .025
        scene.add(light)
    }
    scene.modelRoot.traverse(object => {
        const mesh = object as Mesh
        if (!mesh.isMesh) return
        mesh.castShadow = mesh.receiveShadow = true
        for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            const m = mat as MeshPhysicalMaterial
            if (id === 'arunthayan' && m.name === 'Body') {
                // The imported skin uses a full-strength clearcoat layer, which
                // reads as wet/plastic under the portrait HDR. Keep a restrained
                // natural highlight while preserving the authored texture maps.
                m.roughness = Math.max(m.roughness, .9)
                m.clearcoat = Math.min(m.clearcoat, .08)
                m.clearcoatRoughness = Math.max(m.clearcoatRoughness, .85)
                m.specularIntensity = Math.min(m.specularIntensity, .5)
                m.needsUpdate = true
            }
            if (view.instances && m.transparent && !m.transmission) {
                // Preserve soft foliage coverage while writing depth and
                // casting shadows; sorting an entire meadow as one blended
                // object produces incorrect transparency and excessive overdraw.
                m.transparent = false
                m.alphaHash = true
                // This WebGI Three fork's shadow pass copies alphaTest and
                // texture maps, but not alphaHash. Exclude fully empty texels
                // there while retaining fractional coverage in the main pass.
                m.alphaTest = Math.max(m.alphaTest, 1e-6)
                m.depthWrite = true
                m.needsUpdate = true
            }
            for (const key of ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'alphaMap', 'emissiveMap'] as const) {
                if (m[key]) m[key].anisotropy = 16
            }
        }
    })
    if (view.floor) {
        const original = scene.modelRoot.getObjectByName(view.floor) as Mesh
        if (!original?.isMesh) throw new Error('The studio floor is missing')
        original.updateWorldMatrix(true, false)
        const ground = await viewer.addPlugin(new GroundPlugin({autoAdjustTransform: false}))
        ground.bakedShadows = false
        ground.autoBakeShadows = false
        ground.groundReflection = true
        // The remaining suspended studio uses the VJSON's full, crisp
        // planar reflection finish.
        ground.physicalReflections = false
        ground.size = 1
        // Keep the authored floor geometry and placement.
        const geometry = original.geometry.clone().applyMatrix4(original.matrixWorld)
        geometry.computeBoundingBox()
        const floorCenter = geometry.boundingBox!.getCenter(new Vector3())
        // Reflector2 derives its plane from the mesh origin, not the vertices.
        // Put this horizontal floor on local Z=0 so that the reflection camera
        // mirrors across the actual floor instead of world Y=0.
        geometry.translate(-floorCenter.x, -floorCenter.y, -floorCenter.z)
        geometry.rotateX(Math.PI / 2)
        ground.setGeometry(geometry)
        ground.material!.color.set(0xa8a8a8)
        ground.material!.roughness = 0
        ground.material!.metalness = 1
        // GroundPlugin excludes this material from SSR; other surfaces keep SSR.
        original.visible = false
        ground.refreshOptions()
        ground.mesh!.modelObject.position.copy(floorCenter)
        ground.mesh!.modelObject.rotation.set(-Math.PI / 2, 0, 0)
        ground.mesh!.modelObject.updateMatrixWorld(true)
    }
    const ao = viewer.getPlugin(SSAOPlugin)!.passes.ssao.passObject.parameters
    ao.occlusionWorldRadius = view.giRadius ?? (id.startsWith('selection-') || id === 'arunthayan' ? .12 : 1)
    const camera = scene.activeCamera
    Object.assign(camera.cameraObject.userData, {autoNearFar: false, minNearPlane: view.camera.near, maxFarPlane: view.camera.far})
    camera.setCameraOptions(view.camera)
    const orbit = camera.getControls<OrbitControls>()
    if (orbit) {
        const distance = new Vector3().fromArray(view.camera.position).distanceTo(new Vector3().fromArray(view.camera.target))
        orbit.minDistance = distance * .12
        orbit.maxDistance = distance * 2.5
        orbit.maxPolarAngle = Math.PI * .75
        orbit.update()
    }
    if (view.instances) {
        const groups = createStaticDrawGroups(viewer)
        ;(window as any).dlssStaticDrawGroups = groups
        dispose(() => { groups.dispose(); delete (window as any).dlssStaticDrawGroups })
        const occlusion = createVegetationOcclusion(viewer)
        ;(window as any).dlssVegetationOcclusion = occlusion
        dispose(() => { occlusion?.dispose(); delete (window as any).dlssVegetationOcclusion })
        const fusion = createOpaquePassFusion(viewer)
        ;(window as any).dlssOpaquePassFusion = fusion
        dispose(() => { fusion?.dispose(); delete (window as any).dlssOpaquePassFusion })
    }
    viewer.setDirty()
}
