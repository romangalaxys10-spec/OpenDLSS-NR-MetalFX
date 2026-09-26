import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {addBasePlugins} from 'webgi/extras/viewer/CoreViewerApp'
import {KTX2LoadPlugin} from 'webgi/extras/asset_manager/importer/threejs/generators/ktx2'
import {DlssBridgePlugin} from 'webgi/plugins/DlssBridgePlugin'
import {VelocityBufferPlugin} from 'webgi/plugins/VelocityBufferPlugin'
import {DiamondPlugin} from 'webgi/extras/ijewel/diamondPlugin'
import {GemRefractionPlugin} from 'webgi/plugins/GemRefractionPlugin'
import {GLTFMeshOptPlugin} from 'webgi/plugins/GLTFMeshOptPlugin'
import {mountDemoScenes} from './scenes/nr-demo-scenes'

const query = new URLSearchParams(location.search)

// Deliberately standalone: no try-on SDK, backend, hand image, or imported viewer preset.
async function main() {
    ;(window as any).dlssLoading?.stage('Preparing viewer', 'Connecting to your GPU and loading the viewer')
    const canvas = document.querySelector<HTMLCanvasElement>('#mcanvas')
    if (!canvas) throw new Error('Viewer canvas is missing')
    const viewer = new ViewerApp({canvas, useRgbm: false, useGBufferDepth: false})
    viewer.renderEnabled = false
    const diagnostic = ['srReferenceHead', 'srReferenceEnc0Down', 'srReferenceDec5',
        'srReferenceDec4', 'srReferenceDec0', 'nrReference', 'srDec0Sweep']
        .some(key => query.get(key) === '1')
    await viewer.addPlugin(new VelocityBufferPlugin(true, true, true, true, false))
    KTX2LoadPlugin.TRANSCODER_LIBRARY_PATH = '/three/examples/jsm/libs/basis/'
    GLTFMeshOptPlugin.DECODER_URL = '/three/examples/jsm/libs/meshopt_decoder.module.js'
    await viewer.addPlugin(GLTFMeshOptPlugin)
    await addBasePlugins(viewer, {ground: false, bloom: false, depthTonemap: false,
        enableDrop: false, interactionPrompt: false})
    // addBasePlugins includes dormant scene effects. Let each demo scene own
    // these plugins so their settings, targets and listeners cannot carry over.
    for (const name of ['SSGI', 'SSContactShadows', 'RandomizedDirectionalLight', 'HDRiGroundPlugin', 'WatchHandsPlugin']) {
        const plugin = viewer.getPlugin(name)
        if (plugin) await viewer.removePlugin(plugin)
    }
    await viewer.getOrAddPlugin(DiamondPlugin)
    await viewer.getOrAddPlugin(GemRefractionPlugin)
    const bridge = await viewer.addPlugin(new DlssBridgePlugin(1, false, true, 1, false))
    bridge.srOnly = false
    bridge.nrTemporal = !diagnostic
    viewer.renderer.renderScale = 1
    await mountDemoScenes(viewer, bridge)
    viewer.scene.activeCamera.interactionsEnabled = true
    viewer.renderer.refreshPipeline()
    viewer.renderEnabled = true
    ;(window as any).dlssViewer = viewer
    window.dispatchEvent(new CustomEvent('dlss-viewer-ready', {detail: viewer}))
    viewer.setDirty()
}

main().catch(error => {
    ;(window as any).dlssLoading?.fail(error)
    console.error(error)
    const status = document.querySelector('#dlssWebGpuStatus')
    if (status) {
        status.textContent = `Scene loading failed: ${error.message || error}`
        status.setAttribute('data-state', 'error')
    }
})
