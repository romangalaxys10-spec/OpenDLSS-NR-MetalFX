import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {DlssBridgePlugin} from 'webgi/plugins/DlssBridgePlugin'
import {Mesh, Light} from 'three'
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls'
import {configureBistroLighting} from './nr-bistro-lighting'
import {SSAOPlugin} from 'webgi/plugins/SSAOPlugin'
import {configureLoneMonkScene} from './nr-lone-monk-scene'
import {blendkitScenes, loadBlendkitScene} from './nr-blendkit-scenes'
import {assetUrl} from '../asset-url.js'

// Cowboy Gramps is the only scene this build ships assets for.
const sceneNames: Record<string, string> = {'selection-five': blendkitScenes['selection-five']}
const modelExtensions = new Set(['glb', 'gltf', 'drc', 'obj', 'fbx', 'stl', '3dm', 'zip'])
const defaultLocalEnvironment = 'studio-small-08'
const localEnvironments: Record<string, {label: string, path?: string}> = {
    embedded: {label: 'Embedded in model'},
    'studio-small-08': {label: 'Neutral studio · soft', path: assetUrl('/environments/studio_small_08_1k.hdr')},
    'studio-small-09': {label: 'Neutral studio · directional', path: assetUrl('/environments/studio_small_09_1k.hdr')},
    'white-studio-04': {label: 'Bright product studio', path: assetUrl('/environments/white_studio_04_1k.hdr')},
    'cloud-layers': {label: 'Outdoor · bright midday', path: assetUrl('/environments/cloud_layers_1k.hdr')},
    'venice-sunset': {label: 'Outdoor · warm sunset', path: assetUrl('/environments/venice_sunset_1k.hdr')},
    'autumn-forest-02': {label: 'Forest · soft overcast', path: assetUrl('/environments/autumn_forest_02_1k.hdr')},
}

/** One viewer; each selection disposes the old models and scene-owned lighting. */
export async function mountDemoScenes(viewer: ViewerApp, bridge: DlssBridgePlugin) {
    const globals = window as any
    const {scene} = viewer
    const roots = new Set(scene.children)
    const ambientOcclusion = await viewer.getOrAddPlugin(SSAOPlugin)
    ambientOcclusion.enabled = true
    const defaultAoRadius = ambientOcclusion.passes.ssao.passObject.parameters.occlusionWorldRadius
    const sharedPlugins = new Set(Object.values(viewer.plugins))
    const camera = scene.activeCamera
    const cameraDefaults = camera.getCameraOptions()
    const cameraData = {...camera.cameraObject.userData}
    const orbit = camera.getControls<OrbitControls>()
    const orbitDefaults = orbit ? {minDistance: orbit.minDistance, maxDistance: orbit.maxDistance,
        minPolarAngle: orbit.minPolarAngle, maxPolarAngle: orbit.maxPolarAngle} : null
    const tonemap = viewer.getPlugin('Tonemap') as any
    const defaultExposure = tonemap?.exposure
    const defaultToneMapping = tonemap?.toneMapping
    const progressive = viewer.getPlugin('Progressive') as any
    const defaultFrames = progressive?.maxFrameCount
    const shadowType = viewer.renderer.rendererObject.shadowMap.type
    let disposers: (() => void)[] = []
    let changing = false
    let loadingName = 'scene'
    let localEnvironmentId = defaultLocalEnvironment
    let embeddedEnvironment: any = null
    const localEnvironmentTextures = new Map<string, any>()
    viewer.getManager()?.importer?.addEventListener('importFile', (event: any) => {
        if (!changing || event.state !== 'downloading') return
        const asset = /\.(hdr|exr)(\?|$)/i.test(event.path) ? 'Environment' : '3D model'
        globals.dlssLoading?.stage(`Loading ${loadingName}`, `Downloading ${asset.toLowerCase()}`, {
            loaded: event.loadedBytes || 0, total: event.totalBytes || 0,
        })
        if (event.totalBytes > 0 && event.loadedBytes >= event.totalBytes) {
            globals.dlssLoading?.stage(`Preparing ${loadingName}`, `${asset} downloaded · preparing materials and textures`)
        }
    })
    viewer.getManager()?.importer?.addEventListener('processFileStart', () => {
        if (changing) globals.dlssLoading?.stage(`Preparing ${loadingName}`, 'Preparing scene materials and textures')
    })

    async function clearScene() {
        disposers.forEach(dispose => dispose())
        disposers = []
        // Dependencies are added first; remove scene plugins in reverse order.
        // Preserve the viewer and DLSS runtime, including mode-specific buffers.
        for (const plugin of Object.values(viewer.plugins).reverse()) {
            if (!sharedPlugins.has(plugin)) await viewer.removePlugin(plugin)
        }
        // Every demo uses the shared SSAO pass. Only its scene-scale radius varies.
        ambientOcclusion.enabled = true
        ambientOcclusion.passes.ssao.passObject.parameters.occlusionWorldRadius = defaultAoRadius
        scene.disposeSceneModels()
        for (const object of [...scene.children]) {
            if (roots.has(object)) continue
            object.traverse(child => {
                const light = child as Light & {shadow?: {dispose(): void}}
                light.shadow?.dispose()
                const mesh = child as Mesh
                mesh.geometry?.dispose()
                if (mesh.material) for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose()
            })
            object.removeFromParent()
        }
        const environmentResources = new Set<any>([scene.environment, ...localEnvironmentTextures.values()])
        if ((scene.background as any)?.isTexture) environmentResources.add(scene.background)
        if (embeddedEnvironment?.environment) environmentResources.add(embeddedEnvironment.environment)
        if (embeddedEnvironment?.background?.isTexture) environmentResources.add(embeddedEnvironment.background)
        scene.environment = null
        scene.background = null
        for (const resource of environmentResources) resource?.dispose?.()
        localEnvironmentTextures.clear()
        embeddedEnvironment = null
        localEnvironmentId = defaultLocalEnvironment
        globals.dlssEmbeddedEnvironment = false
        scene.fog = null
        scene.environmentIntensity = 1
        scene.backgroundIntensity = 1
        scene.environmentRotation.set(0, 0, 0)
        scene.backgroundRotation.set(0, 0, 0)
        if (tonemap) { tonemap.exposure = defaultExposure; tonemap.toneMapping = defaultToneMapping }
        if (progressive) progressive.maxFrameCount = defaultFrames
        viewer.renderer.rendererObject.shadowMap.type = shadowType
        for (const key of ['autoNearFar', 'minNearPlane', 'maxFarPlane']) {
            if (key in cameraData) camera.cameraObject.userData[key] = cameraData[key]
            else delete camera.cameraObject.userData[key]
        }
        camera.setCameraOptions(cameraDefaults)
        if (orbit && orbitDefaults) Object.assign(orbit, orbitDefaults)
    }

    async function applyLocalEnvironment(id: string) {
        const selection = localEnvironments[id]
        if (!selection) throw new Error('Unknown HDR environment')
        if (id === 'embedded') {
            if (!embeddedEnvironment) throw new Error('This model does not contain an embedded WebGI environment')
            scene.environment = embeddedEnvironment.environment
            scene.background = embeddedEnvironment.background
            scene.environmentIntensity = embeddedEnvironment.environmentIntensity
            scene.backgroundIntensity = embeddedEnvironment.backgroundIntensity
            scene.environmentRotation.copy(embeddedEnvironment.environmentRotation)
            scene.backgroundRotation.copy(embeddedEnvironment.backgroundRotation)
        } else {
            let texture = localEnvironmentTextures.get(id)
            if (!texture) {
                texture = await viewer.setEnvironmentMap(selection.path!, {setBackground: true})
                if (!texture) throw new Error(`Unable to load ${selection.label}`)
                localEnvironmentTextures.set(id, texture)
            } else {
                scene.environment = texture
                scene.background = texture
            }
            scene.environmentIntensity = 1
            scene.backgroundIntensity = 1
            scene.environmentRotation.set(0, 0, 0)
            scene.backgroundRotation.set(0, 0, 0)
        }
        localEnvironmentId = id
        globals.dlssLocalEnvironment = id
        window.dispatchEvent(new CustomEvent('dlss-local-environment-changed', {detail: {id}}))
    }

    async function changeScene(id: string) {
        if (!sceneNames[id]) throw new Error('Unknown demo scene')
        if (changing) throw new Error('A scene is already loading')
        changing = true
        globals.dlssSceneLoading = true
        document.body.dataset.sceneLoading = 'true'
        globals.dlssSceneReady = false
        loadingName = sceneNames[id]
        globals.dlssLoading?.begin(`Loading ${loadingName}`, 'Preparing the scene')
        try {
            await globals.dlssDemoUi?.beforeSceneChange()
            viewer.renderEnabled = false
            document.querySelector('#dlssWebGpuOutput')?.classList.remove('visible')
            const status = document.querySelector('#dlssWebGpuStatus') as HTMLElement
            status.textContent = `Loading ${sceneNames[id]}…`
            status.dataset.state = 'active'
            await clearScene()
            if (blendkitScenes[id]) {
                await loadBlendkitScene(viewer, id, dispose => disposers.push(dispose))
            } else if (id === 'bistro') {
                const response = await fetch(assetUrl('/scenes/bistro/view.json'))
                if (!response.ok) throw new Error('Bistro camera data is unavailable')
                const view = await response.json()
                await viewer.setEnvironmentMap(assetUrl('/scenes/bistro/san_giuseppe_bridge_4k.hdr'), {setBackground: true})
                await viewer.load(assetUrl('/scenes/bistro/BistroExterior.glb'), {importConfig: false, autoScale: false})
                await configureBistroLighting(viewer)
                camera.setCameraOptions({position: view.position, target: view.target, fov: view.fov, near: 0.05, far: 1000})
            } else if (id === 'lone-monk') {
                await viewer.load(assetUrl('/scenes/lone-monk/lone-monk.glb'), {importConfig: false, autoScale: false})
                await configureLoneMonkScene(viewer, dispose => disposers.push(dispose))
            }
            globals.dlssSceneId = id
            globals.dlssSceneLabel = sceneNames[id]
            globals.dlssSceneReady = true
            const query = new URLSearchParams(location.search)
            const mode = query.get('sr') === '1' ? (query.get('srChain') === '1' ? 'SR + Neural rendering' : 'Super resolution') : 'Neural rendering'
            document.title = `${sceneNames[id]} · DLSS 5 WebGPU`
            const heading = document.querySelector('h1')
            if (heading) heading.textContent = `${sceneNames[id]} / ${mode}`
            document.querySelector('#mcanvas')?.setAttribute('aria-label', `Interactive ${sceneNames[id]} viewer`)
            bridge.resetHistory()
            viewer.renderer.refreshPipeline()
            viewer.renderer.resetShadows()
            // The loading state replaces the scene transition. Do not feed a
            // crossfade containing the disposed scene into the next DLSS frame.
            ;(viewer.getPlugin('FrameFade') as any)?.stopTransition()
            scene.setDirty({sceneUpdate: true, frameFade: false})
            viewer.renderEnabled = true
            viewer.setDirty()
            globals.dlssLoading?.stage('Preparing first frame', `${loadingName} loaded · preparing the rendered view`)
            globals.dlssDemoUi?.afterSceneChange()
            window.dispatchEvent(new CustomEvent('dlss-scene-ready', {detail: {id, label: sceneNames[id]}}))
        } catch (error) {
            globals.dlssLoading?.fail(error)
            await clearScene()
            throw error
        } finally {
            changing = false
            globals.dlssSceneLoading = false
            delete document.body.dataset.sceneLoading
            globals.dlssLoading?.releaseScene?.()
        }
    }
    async function importLocalFiles(files: Map<string, File>) {
        if (changing) throw new Error('A scene is already loading')
        const rootFiles = [...files.keys()].filter(path => modelExtensions.has(path.split('.').pop()?.toLowerCase() ?? ''))
        if (!rootFiles.length) {
            const supported = [...modelExtensions].map(extension => `.${extension}`).join(', ')
            throw new Error(`No supported 3D model found. Supported formats: ${supported}`)
        }
        changing = true
        globals.dlssSceneLoading = true
        document.body.dataset.sceneLoading = 'true'
        globals.dlssSceneReady = false
        const filename = rootFiles[0].split(/[\\/]/).pop() || 'Local model'
        loadingName = filename.replace(/\.[^.]+$/, '') || filename
        globals.dlssLoading?.begin(`Loading ${loadingName}`, 'Preparing local files')
        try {
            await globals.dlssDemoUi?.beforeSceneChange()
            viewer.renderEnabled = false
            document.querySelector('#dlssWebGpuOutput')?.classList.remove('visible')
            const status = document.querySelector('#dlssWebGpuStatus') as HTMLElement
            status.textContent = `Loading ${loadingName}…`
            status.dataset.state = 'active'
            // This removes scene models, custom meshes/lights, environments,
            // backgrounds, fog and scene-owned plugins before import begins.
            await clearScene()
            const manager = viewer.getManager()
            // Do not suppress importConfig here: WebGI-authored GLBs can carry
            // viewer settings, cameras, lighting and embedded environment maps.
            const options = {autoScale: true, autoCenter: true, autoScaleRadius: 2}
            const imported = await manager?.importer?.importFiles(files as any, options)
            const assets = [...imported?.values() ?? []].flat(2).filter(Boolean)
            if (!assets.length) throw new Error(`WebGI could not import ${filename}`)
            const added = manager?.addProcessedAssets(assets as any, options as any).filter(Boolean) ?? []
            if (!added.length) throw new Error(`WebGI did not add ${filename} to the scene`)
            if (scene.environment) {
                embeddedEnvironment = {
                    environment: scene.environment,
                    background: scene.background,
                    environmentIntensity: scene.environmentIntensity,
                    backgroundIntensity: scene.backgroundIntensity,
                    environmentRotation: scene.environmentRotation.clone(),
                    backgroundRotation: scene.backgroundRotation.clone(),
                }
                globals.dlssEmbeddedEnvironment = true
                await applyLocalEnvironment('embedded')
            } else {
                globals.dlssEmbeddedEnvironment = false
                await applyLocalEnvironment(defaultLocalEnvironment)
            }
            globals.dlssSceneId = 'local-file'
            globals.dlssSceneLabel = loadingName
            globals.dlssSceneReady = true
            const query = new URLSearchParams(location.search)
            const mode = query.get('sr') === '1' ? (query.get('srChain') === '1' ? 'SR + Neural rendering' : 'Super resolution') : 'Neural rendering'
            document.title = `${loadingName} · DLSS 5 WebGPU`
            const heading = document.querySelector('h1')
            if (heading) heading.textContent = `${loadingName} / ${mode}`
            document.querySelector('#mcanvas')?.setAttribute('aria-label', `Interactive ${loadingName} viewer`)
            bridge.resetHistory()
            viewer.renderer.refreshPipeline()
            viewer.renderer.resetShadows()
            ;(viewer.getPlugin('FrameFade') as any)?.stopTransition()
            scene.setDirty({sceneUpdate: true, frameFade: false})
            viewer.renderEnabled = true
            viewer.setDirty()
            globals.dlssLoading?.stage('Preparing first frame', `${loadingName} loaded · preparing the rendered view`)
            globals.dlssDemoUi?.afterSceneChange()
            window.dispatchEvent(new CustomEvent('dlss-scene-ready', {detail: {id: 'local-file', label: loadingName}}))
        } catch (error) {
            globals.dlssLoading?.fail(error)
            await clearScene()
            throw error
        } finally {
            changing = false
            globals.dlssSceneLoading = false
            delete document.body.dataset.sceneLoading
            globals.dlssLoading?.releaseScene?.()
        }
    }
    async function setLocalEnvironment(id: string) {
        if (globals.dlssSceneId !== 'local-file' || !globals.dlssSceneReady) return
        if (id === localEnvironmentId || changing) return
        changing = true
        globals.dlssSceneLoading = true
        globals.dlssSceneReady = false
        document.body.dataset.sceneLoading = 'true'
        const label = localEnvironments[id]?.label || 'HDR environment'
        globals.dlssLoading?.begin(`Loading ${label}`, 'Preparing environment lighting')
        try {
            await globals.dlssDemoUi?.beforeSceneChange()
            viewer.renderEnabled = false
            await applyLocalEnvironment(id)
            bridge.resetHistory()
            viewer.renderer.refreshPipeline()
            viewer.renderer.resetShadows()
            scene.setDirty({sceneUpdate: true, frameFade: false})
            globals.dlssSceneReady = true
            viewer.renderEnabled = true
            viewer.setDirty()
            globals.dlssLoading?.stage('Preparing first frame', `${label} loaded · preparing the rendered view`)
            globals.dlssDemoUi?.afterSceneChange()
        } catch (error) {
            globals.dlssLoading?.fail(error)
            globals.dlssSceneReady = true
            throw error
        } finally {
            changing = false
            globals.dlssSceneLoading = false
            delete document.body.dataset.sceneLoading
            globals.dlssLoading?.releaseScene?.()
        }
    }
    let initial = 'selection-five'
    try {
        const saved = sessionStorage.getItem('dlss-demo-scene')
        sessionStorage.removeItem('dlss-demo-scene')
        if (saved && sceneNames[saved]) initial = saved
    } catch { /* Storage is optional. */ }
    globals.dlssChangeScene = changeScene
    globals.dlssImportFiles = importLocalFiles
    globals.dlssSetLocalEnvironment = setLocalEnvironment
    globals.dlssLocalEnvironment = localEnvironmentId
    globals.dlssEmbeddedEnvironment = false
    globals.dlssLocalEnvironments = Object.fromEntries(Object.entries(localEnvironments)
        .map(([id, environment]) => [environment.label, id]))
    await changeScene(initial)
}
