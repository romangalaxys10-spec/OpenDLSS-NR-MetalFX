import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {SSAOPlugin} from 'webgi/plugins/SSAOPlugin'
import {DirectionalLight, Mesh, MeshStandardMaterial, SpotLight, Texture, Vector3} from 'three'

/** Configure lighting for the unscaled, metre-sized Bistro exterior. */
export async function configureBistroLighting(viewer: ViewerApp): Promise<void> {
    const {scene} = viewer
    const capabilities = viewer.renderer.rendererObject.capabilities
    const materials = new Set<MeshStandardMaterial>()
    const textures = new Set<Texture>()
    const cafeLamps: Vector3[] = []
    scene.modelRoot.traverse(object => {
        const mesh = object as Mesh
        if (!mesh.isMesh) return
        const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
        meshMaterials.forEach(material => materials.add(material as MeshStandardMaterial))
        if (meshMaterials.some(material => material.name === 'Emissive_StreetLight')) {
            mesh.geometry.computeBoundingBox()
            const position = mesh.geometry.boundingBox!.getCenter(new Vector3()).applyMatrix4(mesh.matrixWorld)
            // The five pendant globes beneath the cafe awnings. Their emissive
            // textures alone do not illuminate the surrounding rasterized scene.
            if (position.y > 3 && position.y < 4.2 && position.x > -4 && position.x < 14
                && position.z > -14 && position.z < 9) cafeLamps.push(position)
        }
    })
    for (const material of materials) {
        // Bistro v5.2 leaves R black in all 201 packed Specular maps. It is an
        // unused channel, not glTF AO (where zero blocks all indirect light).
        // Also repair previously exported GLBs; new exports omit this binding.
        if (material.aoMap?.name.endsWith('_Specular')) {
            material.aoMap = null
            material.needsUpdate = true
        }
        // Leaf cards need cutout shadows and depth, rather than blended rectangles.
        if (material.transparent && material.name.startsWith('Foliage_')) {
            material.transparent = false
            material.alphaTest = 0.5
            material.depthWrite = true
            material.needsUpdate = true
        }
        for (const texture of [material.map, material.normalMap, material.roughnessMap,
            material.metalnessMap, material.aoMap]) {
            if (texture) textures.add(texture)
        }
    }
    for (const texture of textures) {
        texture.anisotropy = Math.min(16, capabilities.getMaxAnisotropy())
        texture.needsUpdate = true
    }

    const sun = scene.modelRoot.getObjectByName('directionalLight1') as DirectionalLight | undefined
    if (sun?.isDirectionalLight) {
        const bounds = scene.getModelBounds()
        const center = bounds.getCenter(new Vector3())
        const distance = bounds.getSize(new Vector3()).length()
        // Remove the FBX parent scale before positioning the sun in world units.
        scene.attach(sun)
        scene.attach(sun.target)
        sun.target.position.copy(center)
        sun.position.copy(center).addScaledVector(new Vector3(-0.62, 0.85, -0.4).normalize(), distance)
        sun.color.set('#ffedd4')
        sun.intensity = 5.5
        sun.castShadow = true
        sun.updateMatrixWorld(true)
        sun.target.updateMatrixWorld(true)

        // The imported 512px, 10m shadow frustum misses almost all of the street.
        const {shadow} = sun
        const camera = shadow.camera
        camera.position.copy(sun.position)
        camera.lookAt(center)
        camera.updateMatrixWorld(true)
        const lightBounds = bounds.clone().applyMatrix4(camera.matrixWorldInverse)
        camera.left = lightBounds.min.x - 2
        camera.right = lightBounds.max.x + 2
        camera.bottom = lightBounds.min.y - 2
        camera.top = lightBounds.max.y + 2
        camera.near = Math.max(0.1, -lightBounds.max.z - 2)
        camera.far = -lightBounds.min.z + 2
        camera.updateProjectionMatrix()
        const shadowSize = Math.min(4096, capabilities.maxTextureSize)
        shadow.mapSize.set(shadowSize, shadowSize)
        shadow.bias = -0.0001
        shadow.normalBias = 0.04
        shadow.radius = 1.5
    }

    // The sun provides shape; restrained HDR sky fill keeps the shaded stone
    // readable without lifting the entire image or changing the display curve.
    scene.environmentIntensity = 0.8
    for (const [index, position] of cafeLamps.entries()) {
        const lamp = new SpotLight('#ffd09a', 24, 5, Math.PI / 3, 0.75, 2)
        lamp.name = `Bistro cafe pendant ${index + 1}`
        lamp.position.copy(position).add(new Vector3(0, -0.12, 0))
        lamp.target.position.copy(position).add(new Vector3(0, -3, 0))
        scene.add(lamp, lamp.target)
    }
    viewer.getPlugin(SSAOPlugin)!.passes.ssao.passObject.parameters.occlusionWorldRadius = 1.2
    viewer.renderer.resetShadows()
    scene.setDirty()
}
