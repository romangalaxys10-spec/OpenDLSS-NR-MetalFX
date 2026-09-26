import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {DirectionalLight2} from 'webgi/core/threejs/Lights'
import {ProgressivePlugin} from 'webgi/plugins/threejs/ProgressivePlugin'
import {TonemapPlugin} from 'webgi/plugins/threejs/TonemapPlugin'
import {assetUrl} from '../asset-url.js'
import {SSAOPlugin} from 'webgi/plugins/SSAOPlugin'
import {SSContactShadows} from 'webgi/plugins/SSContactShadows'
import {Box3, CanvasTexture, Color, Float32BufferAttribute, Mesh, MeshPhysicalMaterial,
    PCFSoftShadowMap, SRGBColorSpace, Texture, Vector3} from 'three'
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls'

const convergenceFrames = 128

/** The Blender page shader was exported as a flat green swatch. */
function createPageTexture(): CanvasTexture {
    const canvas = document.createElement('canvas')
    canvas.width = 256
    canvas.height = 512
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Cannot create the courtyard page texture')
    const image = context.createImageData(canvas.width, canvas.height)
    let seed = 7281
    const random = () => {
        seed = Math.imul(1664525, seed) + 1013904223 >>> 0
        return seed / 4294967296
    }
    for (let y = 0; y < canvas.height; y++) {
        const line = 0.72 + random() * 0.28
        for (let x = 0; x < canvas.width; x++) {
            const grain = 0.95 + random() * 0.1
            const edge = 0.85 + 0.15 * Math.sin(Math.PI * x / canvas.width)
            const value = line * grain * edge
            const i = (y * canvas.width + x) * 4
            image.data.set([214 * value, 196 * value, 163 * value, 255], i)
        }
    }
    context.putImageData(image, 0, 0)
    const texture = new CanvasTexture(canvas)
    texture.name = 'Lone Monk aged page edges'
    texture.colorSpace = SRGBColorSpace
    texture.flipY = false
    return texture
}

/** Restore the per-tile variation lost from Blender's procedural roof. */
function colorRoofTiles(mesh: Mesh): void {
    const geometry = mesh.geometry
    const position = geometry.getAttribute('position')
    const parent = new Int32Array(position.count)
    const welded = new Map<string, number>()
    const root = (i: number): number => {
        while (parent[i] !== i) {
            parent[i] = parent[parent[i]]
            i = parent[i]
        }
        return i
    }
    // Join duplicate vertices across UV/normal seams, then find each separate tile.
    for (let i = 0; i < position.count; i++) {
        parent[i] = i
        const key = [position.getX(i), position.getY(i), position.getZ(i)]
            .map(v => Math.round(v * 10000)).join(',')
        const previous = welded.get(key)
        if (previous === undefined) welded.set(key, i)
        else parent[i] = root(previous)
    }
    const index = geometry.index
    const count = index?.count ?? position.count
    for (let i = 0; i < count; i += 3) {
        const a = index ? index.getX(i) : i
        const b = index ? index.getX(i + 1) : i + 1
        const c = index ? index.getX(i + 2) : i + 2
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    }
    const colors = new Float32Array(position.count * 3)
    const palette = ['#a6866b', '#c1a58a', '#c5b49b', '#8c7969', '#b8997a', '#b2a78c']
        .map(value => new Color(value))
    for (let i = 0; i < position.count; i++) {
        const hash = Math.imul(root(i) + 17, 2654435761) >>> 0
        palette[hash % palette.length].toArray(colors, i * 3)
    }
    geometry.setAttribute('color', new Float32BufferAttribute(colors, 3))
}

/** Fit shadow coverage to the complete courtyard for this light direction. */
function placeLight(light: DirectionalLight2, direction: Vector3, bounds: Box3): void {
    const center = bounds.getCenter(new Vector3())
    light.position.copy(center).addScaledVector(direction, bounds.getSize(new Vector3()).length())
    // DirectionalLight2's target is a child: use a local offset.
    light.target.position.copy(center).sub(light.position)
    light.updateMatrixWorld(true)
    const camera = light.shadow.camera
    camera.position.copy(light.position)
    camera.lookAt(center)
    camera.updateMatrixWorld(true)
    const lightBounds = bounds.clone().applyMatrix4(camera.matrixWorldInverse)
    camera.left = lightBounds.min.x - 1
    camera.right = lightBounds.max.x + 1
    camera.bottom = lightBounds.min.y - 1
    camera.top = lightBounds.max.y + 1
    camera.near = Math.max(0.1, -lightBounds.max.z - 1)
    camera.far = -lightBounds.min.z + 1
    camera.updateProjectionMatrix()
}

/** Preserve the unscaled architecture and rebuild missing material/lighting detail. */
export async function configureLoneMonkScene(viewer: ViewerApp, onCleanup?: (dispose: () => void) => void): Promise<void> {
    const {scene} = viewer
    const square = new URLSearchParams(location.search).get('sr') === '1'
    const renderer = viewer.renderer.rendererObject
    const materials = new Set<MeshPhysicalMaterial>()
    const textures = new Set<Texture>()
    const preparedGeometry = new Set<Mesh['geometry']>()
    const pageTexture = createPageTexture()
    scene.modelRoot.traverse(object => {
        const mesh = object as Mesh
        if (!mesh.isMesh) return
        const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
        if (list.some(material => material.name === 'sky1')) {
            mesh.visible = false
            mesh.userData.bboxVisible = false
        }
        if (list.some(material => material.name === 'Greenleaf Treeline 003')) {
            mesh.castShadow = false
            mesh.receiveShadow = false
            mesh.userData.bboxVisible = false
        }
        if (!preparedGeometry.has(mesh.geometry)) {
            if (list.some(material => material.name === 'paper - book')) {
                const geometry = mesh.geometry
                geometry.computeBoundingBox()
                const bounds = geometry.boundingBox
                if (!bounds) throw new Error('Courtyard page geometry has no bounds')
                const position = geometry.getAttribute('position')
                const uv = new Float32Array(position.count * 2)
                for (let i = 0; i < position.count; i++) {
                    uv[i * 2] = (position.getX(i) + position.getZ(i)) / 0.6 + 0.5
                    uv[i * 2 + 1] = (position.getY(i) - bounds.min.y) / Math.max(0.001, bounds.max.y - bounds.min.y)
                }
                geometry.setAttribute('uv', new Float32BufferAttribute(uv, 2))
            }
            if (list.some(material => material.name === 'roof')) colorRoofTiles(mesh)
            preparedGeometry.add(mesh.geometry)
        }
        list.forEach(material => materials.add(material as MeshPhysicalMaterial))
    })
    for (const material of materials) {
        // Keep the loader's tangent handedness while reducing exaggerated bump.
        material.normalScale.multiplyScalar(0.5)
        if (material.name === 'Greenleaf Treeline 003') material.emissiveIntensity = 1
        if (material.name.startsWith('wood -')) {
            material.metalness = 0
            material.color.set('#bba28b')
            material.roughnessMap = null
            material.roughness = 0.82
        }
        if (material.name.startsWith('leather - book - cover')) {
            const colors = ['#a78d72', '#887764', '#ac8e71']
            material.color.set(colors[Number(material.name.slice(-1)) - 1])
            material.roughnessMap = null
            material.roughness = 0.88
        }
        if (material.name === 'paper - book') {
            material.color.set('#ffffff')
            material.map = pageTexture
            material.roughness = 1
        }
        if (['column marble', 'brick marble', 'brick marble smooth'].includes(material.name)) {
            material.color.set('#fff5e2')
            material.normalScale.multiplyScalar(0.3)
        }
        if (['stone pavement', 'outdoor pavement', 'grass - ground'].includes(material.name)) {
            material.normalScale.multiplyScalar(0.4)
        }
        if (material.name.startsWith('plaster')) material.color.multiply(new Color('#fff4e4'))
        if (material.name === 'brick clay') material.color.set('#ead0b3')
        if (material.name === 'roof') {
            material.color.set('#c6a17c')
            material.vertexColors = true
            material.normalScale.multiplyScalar(0.4)
            material.roughnessMap = null
            material.roughness = 0.95
        }
        if (material.name === 'iron_nail') {
            // Includes the oxidized column straps, previously almost black.
            material.color.set('#8b897d')
            material.metalness = 0.4
            material.roughness = 0.8
        }
        if (material.name === 'glass far') {
            // Closed facade panes have no modeled rooms behind them. Restore a
            // reflective coating without exposing the empty building shells.
            material.color.set('#b3c5c9')
            material.metalness = 1
            material.roughness = 0.08
            material.envMapIntensity = 4
        }
        for (const texture of [material.map, material.normalMap, material.roughnessMap,
            material.metalnessMap, material.aoMap, material.emissiveMap]) {
            if (texture) textures.add(texture)
        }
        material.needsUpdate = true
    }
    for (const texture of textures) {
        texture.anisotropy = Math.min(16, renderer.capabilities.getMaxAnisotropy())
        texture.needsUpdate = true
    }

    await viewer.setEnvironmentMap(assetUrl('/scenes/lone-monk/kloofendal_48d_partly_cloudy_puresky_4k.hdr'),
        {setBackground: true})
    scene.environmentIntensity = 0.4
    scene.environmentRotation.y = 0.1
    scene.backgroundRotation.copy(scene.environmentRotation)
    const tonemap = viewer.getPlugin(TonemapPlugin)
    if (tonemap) tonemap.exposure = 1.2

    const bounds = scene.getModelBounds()
    // Direction of the HDR panorama's brightest sun pixel.
    const sunDirection = new Vector3(0.554743, 0.741466, 0.377476).normalize()
        .applyAxisAngle(new Vector3(0, 1, 0), scene.environmentRotation.y)
    const sun = new DirectionalLight2('#ffe0b1', 9)
    sun.name = 'Lone Monk sun'
    // A restrained cool fill preserves the sun's side lighting on the columns
    // and separates the shaded study from the sunlit courtyard.
    const sky = new DirectionalLight2('#c7dcff', 1.8)
    sky.name = 'Lone Monk sky fill'
    for (const light of [sun, sky]) {
        light.castShadow = true
        light.userData.bboxVisible = false
        const size = Math.min(light === sun ? 4096 : 2048, renderer.capabilities.maxTextureSize)
        light.shadow.mapSize.set(size, size)
        light.shadow.bias = -0.00005
        light.shadow.normalBias = 0.02
        scene.addLight(light, {addToRoot: true})
        placeLight(light, sunDirection, bounds)
    }
    renderer.shadowMap.type = PCFSoftShadowMap
    placeLight(sky, new Vector3(0, 1, 0), bounds)
    const tangent = new Vector3().crossVectors(sunDirection, new Vector3(0, 1, 0)).normalize()
    const bitangent = new Vector3().crossVectors(sunDirection, tangent)
    const direction = new Vector3()
    // Integrate shadowed sky fill and a soft solar disc across progressive frames.
    // Repeat deterministically: the bridge continues drawing after convergence.
    // SR accumulates its own history, so keep its light positions stable.
    const updateShadows = () => {
        const frame = viewer.renderer.frameCount % convergenceFrames
        let bits = frame, radical = 0, weight = 0.5
        while (bits > 0) {
            radical += (bits & 1) * weight
            bits >>= 1
            weight *= 0.5
        }
        const azimuth = frame * 2.399963229728653
        const height = 0.04 + 0.92 * radical
        const radius = Math.sqrt(1 - height * height)
        direction.set(Math.cos(azimuth) * radius, height, Math.sin(azimuth) * radius)
        placeLight(sky, direction, bounds)
        const sunRadius = 0.018 * Math.sqrt(radical)
        direction.copy(sunDirection).addScaledVector(tangent, Math.cos(azimuth) * sunRadius)
            .addScaledVector(bitangent, Math.sin(azimuth) * sunRadius).normalize()
        placeLight(sun, direction, bounds)
        viewer.renderer.resetShadows()
    }
    if (!square) {
        viewer.addEventListener('preRender', updateShadows)
        onCleanup?.(() => viewer.removeEventListener('preRender', updateShadows))
    }

    viewer.getPlugin(SSAOPlugin)!.passes.ssao.passObject.parameters.occlusionWorldRadius = 0.8
    const contact = viewer.getPlugin(SSContactShadows)
    if (contact) contact.enabled = false
    const progressive = viewer.getPlugin(ProgressivePlugin)
    if (progressive) progressive.maxFrameCount = convergenceFrames

    const camera = scene.activeCamera
    // Lift the imported view toward the arch and desk, keeping the authored
    // near clipping through the wall behind the camera.
    Object.assign(camera.cameraObject.userData, {autoNearFar: false, minNearPlane: 1.12, maxFarPlane: 200})
    camera.setCameraOptions({position: [20, 0.9, -7.512925], target: [20, 2.35, -20],
        fov: square ? 65 : 55.795, near: 1.12, far: 200})
    const controls = camera.getControls<OrbitControls>()
    if (controls) {
        controls.minDistance = 2
        controls.maxDistance = 45
        controls.maxPolarAngle = Math.PI * 0.95
    }
    viewer.renderer.resetShadows()
    scene.setDirty()
}
