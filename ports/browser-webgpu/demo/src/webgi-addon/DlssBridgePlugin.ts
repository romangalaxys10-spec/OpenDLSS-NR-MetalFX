import {GenericFilterPlugin} from 'webgi/plugins/GenericFilterPlugin'
import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {GBufferPlugin} from 'webgi/plugins/GBufferPlugin'
import {VelocityBufferPlugin} from 'webgi/plugins/VelocityBufferPlugin'
import {ProgressivePlugin} from 'webgi/plugins/threejs/ProgressivePlugin'
import {TemporalAAPlugin} from 'webgi/plugins/TemporalAAPlugin'
import {
    Color,
    FloatType,
    HalfFloatType,
    LinearFilter,
    Matrix4,
    Mesh,
    NearestFilter,
    NoBlending,
    NoColorSpace,
    PerspectiveCamera,
    RGBAFormat,
    RGFormat,
    RedFormat,
    ShaderMaterial,
    UnsignedByteType,
    Vector2,
    Vector4,
    WebGLRenderer,
    WebGLRenderTarget,
} from 'three'
import {CopyShader} from 'three/examples/jsm/shaders/CopyShader'
import {FullScreenQuad, Pass} from 'three/examples/jsm/postprocessing/Pass'
import {updateBit} from 'webgi/helpers/mathutil'

export interface DlssBridgeFrame {
    renderer: WebGLRenderer
    color: WebGLRenderTarget
    depth?: WebGLRenderTarget
    motion?: WebGLRenderTarget
    reactive?: WebGLRenderTarget
    control?: WebGLRenderTarget
    renderWidth: number
    renderHeight: number
    outputWidth: number
    outputHeight: number
    jitter: {x: number, y: number}
    previousJitter: {x: number, y: number}
    reset: boolean
}

export type DlssBridgeFrameConsumer = (frame: DlssBridgeFrame) => void

/**
 * GPU-only WebGI -> native DLSS bridge contract.
 *
 * Each export target has the same useful WxH viewport and a distinct number of
 * unused rows. The native ReShade add-on can therefore find the four ANGLE D3D11
 * resources without private Chromium APIs or CPU image readback:
 *
 *   color    RGBA16F  W x (H + 4)   linear, pre-tonemap
 *   depth    R32F     W x (H + 3)   hardware depth
 *   motion   RG16F    W x (H + 2)   jitter-free UV motion
 *   reactive R8       W x (H + 1)   adaptive current-colour confidence
 *   control  R8       OW x (OH + 5)  full-resolution DLSSNR material mask
 *
 * W/H are the reduced engine-render extent and OW/OH are the presentation
 * extent. Only the useful rows contain image data; signature rows stay zero.
 */
export class DlssBridgePlugin extends GenericFilterPlugin<DlssBridgeExportPass, 'dlssBridgeExport', ''> {
    static readonly PluginType = 'DlssBridge'
    passId: 'dlssBridgeExport' = 'dlssBridgeExport'

    protected _beforeFilters = ['combinedPost']
    protected _afterFilters = ['render', 'taa', 'progressive', 'bloom']
    protected _requiredFilters = ['render', 'combinedPost']

    private _frameIndex = 0
    private _camera?: PerspectiveCamera
    private _nrCamera?: PerspectiveCamera
    private _unjitteredProjection = new Matrix4()
    private _jitter = new Vector2()
    private _previousJitter = new Vector2()
    private _serializationIgnoresAdded: string[] = []
    private _frameConsumer?: DlssBridgeFrameConsumer
    /** Frame generation: publish full-scale depth and motion export targets each frame (the presented
     *  colour is taken from the canvas by the FG runtime); no SR scaling, no NR temporal history. */
    fgMode = false
    /** Browser Performance/J consumes color, depth and motion only. */
    srOnly = false
    /** Feature 18 consumes color and motion; its default pre-kernel skips depth. */
    nrTemporal = false
    private _updateGBuffer = (object: any, data: Vector4) => {
        if (!(object instanceof Mesh) || !object.material) return
        const materials = Array.isArray(object.material) ? object.material : [object.material]
        const isDiamond = materials.some((material: any) => material?.isDiamondMaterial)
        // Preserve WebGI's existing inverted bit-4 diamond convention so all
        // G-buffer consumers agree even when GemRefractionPlugin is absent.
        data.w = updateBit(data.w, 4, isDiamond ? 0 : 1)
    }

    constructor(public renderScale = 2 / 3, public jitterEnabled = true,
                public preserveWebGiTemporal = false,
                public displayScale = renderScale,
                public colorOnly = false) {
        super()
    }

    passCtor(viewer: ViewerApp): DlssBridgeExportPass {
        return new DlssBridgeExportPass(viewer)
    }

    protected _update(viewer: ViewerApp): boolean {
        // Imported GLBs may carry a serialized ViewerApp/plugin preset. Those
        // settings are applied after this plugin is installed, and can otherwise
        // disable the required velocity pass or re-enable WebGI temporal history.
        // The DLSS bridge owns these states for as long as it is active.
        const velocity = viewer.getPlugin(VelocityBufferPlugin)
        if (!this.colorOnly && velocity && !velocity.enabled) {
            velocity.enabled = true
            console.info('[dlss-bridge] restored velocity buffer after scene import')
        }
        const progressive = viewer.getPlugin(ProgressivePlugin)
        const taa = viewer.getPlugin(TemporalAAPlugin)
        if (this.preserveWebGiTemporal) {
            if (progressive && !progressive.enabled) progressive.enabled = true
            if (taa && !taa.enabled) taa.enabled = true
        } else {
            if (progressive?.enabled) {
                progressive.enabled = false
                console.info('[dlss-bridge] disabled progressive history restored by scene import')
            }
            if (taa?.enabled) {
                taa.enabled = false
                console.info('[dlss-bridge] disabled WebGI TAA restored by scene import')
            }
        }
        return super._update(viewer)
    }

    async onAdded(viewer: ViewerApp): Promise<void> {
        await super.onAdded(viewer)

        // A temporal upscaler needs a newly rendered engine frame for every
        // presentation. Keep the renderer running through the pass dirty flag,
        // not the plugin dirty flag: ViewerApp treats a dirty plugin as a scene
        // change and resets BaseRenderer.frameCount to zero every refresh. That
        // repeatedly restarts WebGI's own temporal passes and eventually poisons
        // the scene colour/history.
        if (this.pass) {
            Object.defineProperty(this.pass, 'dirty', {
                configurable: true,
                get: () => this.enabled,
                set: () => undefined,
            })
        }

        if (viewer.useRgbm) throw new Error('DlssBridgePlugin requires useRgbm=false for a linear RGBA16F source')

        const progressive = viewer.getPlugin(ProgressivePlugin)
        if (progressive) {
            progressive.jitter = this.preserveWebGiTemporal
            progressive.enabled = this.preserveWebGiTemporal
        }
        const taa = viewer.getPlugin(TemporalAAPlugin)
        if (taa) taa.enabled = this.preserveWebGiTemporal

        const velocity = viewer.getPlugin(VelocityBufferPlugin)
        if (velocity && !this.colorOnly) velocity.enabled = true
        const gbuffer = viewer.getPlugin(GBufferPlugin)
        gbuffer?.registerGBufferUpdater(this._updateGBuffer)
        // The stock 0.22.1 bundle uses linear filtering for the byte-packed
        // material flags. The bridge samples them as discrete bits.
        const flags = gbuffer?.getFlagsTexture()
        if (flags) {
            flags.minFilter = NearestFilter
            flags.magFilter = NearestFilter
            flags.needsUpdate = true
        }

        const ownedPluginTypes = this.preserveWebGiTemporal
            ? [VelocityBufferPlugin.PluginType]
            : [VelocityBufferPlugin.PluginType, ProgressivePlugin.PluginType, TemporalAAPlugin.PluginType]
        for (const type of ownedPluginTypes) {
            if (!viewer.serializePluginsIgnored.includes(type)) {
                viewer.serializePluginsIgnored.push(type)
                this._serializationIgnoresAdded.push(type)
            }
        }

        viewer.renderer.displayCanvasScaling = this.displayScale
        viewer.addEventListener('preRender', this._applyCameraJitter)
        viewer.addEventListener('preFrame', this._captureNrProjection)
        viewer.addEventListener('postRender', this._clearCameraJitter)
    }

    async onRemove(viewer: ViewerApp): Promise<void> {
        viewer.removeEventListener('preRender', this._applyCameraJitter)
        viewer.removeEventListener('preFrame', this._captureNrProjection)
        viewer.removeEventListener('postRender', this._clearCameraJitter)
        ;(viewer.getPlugin(GBufferPlugin) as any)?.unregisterGBufferUpdater?.(this._updateGBuffer)
        for (const type of this._serializationIgnoresAdded) {
            const index = viewer.serializePluginsIgnored.indexOf(type)
            if (index >= 0) viewer.serializePluginsIgnored.splice(index, 1)
        }
        this._serializationIgnoresAdded = []
        this._clearCameraJitter()
        return super.onRemove(viewer)
    }

    setRenderScale(scale: number) {
        const next = Math.max(0.25, Math.min(1, scale))
        if (Math.abs(next - this.renderScale) < 0.0001) return
        this.renderScale = next
        this._frameIndex = 0
        if (this._viewer) {
            if (!this.preserveWebGiTemporal) this.displayScale = next
            this._viewer.renderer.displayCanvasScaling = this.displayScale
            this._viewer.setDirty(this)
        }
    }

    get jitter() {
        return this._jitter
    }

    get previousJitter() {
        return this._previousJitter
    }

    /**
     * Installs an alternate consumer for the exact resources exported for the
     * native feeder. The Electron path does not install one and is unchanged.
     */
    setFrameConsumer(consumer?: DlssBridgeFrameConsumer) {
        this._frameConsumer = consumer
    }

    /** Discard the preceding scene's temporal inputs after an in-place load. */
    resetHistory() {
        this.pass?.passObject.resetHistory()
        this._frameIndex = 0
        this._jitter.set(0, 0)
        this._previousJitter.set(0, 0)
    }

    publishFrame(frame: DlssBridgeFrame) {
        if (!this._frameConsumer) return
        try {
            this._frameConsumer(frame)
        } catch (error) {
            console.error('[dlss-bridge] frame consumer failed', error)
        }
    }

    private _captureNrProjection = () => {
        if (!this.nrTemporal || !this._viewer) return
        const camera = this._viewer.scene.activeCamera.cameraObject
        if (!(camera instanceof PerspectiveCamera)) return
        // Capture before ProgressivePlugin adds its sampling jitter in preRender.
        // NR receives resolved color, so sampling jitter is not scene motion.
        this._nrCamera = camera
        this._unjitteredProjection.copy(camera.projectionMatrix)
        camera.userData.dlssUnjitteredProjectionMatrix = this._unjitteredProjection
    }

    private _applyCameraJitter = () => {
        if (!this.jitterEnabled || !this._viewer) return
        const camera = this._viewer.scene.activeCamera.cameraObject
        if (!(camera instanceof PerspectiveCamera)) return

        // BaseRenderer floors scaled composer extents; use the identical rule so
        // camera jitter is expressed in the exact DLSS input-pixel grid.
        const width = Math.max(1, Math.floor(this._viewer.renderer.renderSize.width * this.displayScale))
        const height = Math.max(1, Math.floor(this._viewer.renderer.renderSize.height * this.displayScale))
        this._camera = camera
        this._unjitteredProjection.copy(camera.projectionMatrix)
        camera.userData.dlssUnjitteredProjectionMatrix = this._unjitteredProjection

        // Halton(2,3), centred on the pixel. The native side uses the same
        // sequence and advances only when these export targets are submitted.
        const sampleIndex = (this._frameIndex++ % 32) + 1
        this._previousJitter.copy(this._jitter)
        this._jitter.set(halton(sampleIndex, 2) - 0.5, halton(sampleIndex, 3) - 0.5)
        camera.setViewOffset(width, height, this._jitter.x, this._jitter.y, width, height)
    }

    private _clearCameraJitter = () => {
        if (this._nrCamera) {
            delete this._nrCamera.userData.dlssUnjitteredProjectionMatrix
            this._nrCamera = undefined
        }
        if (!this._camera) return
        this._camera.clearViewOffset()
        delete this._camera.userData.dlssUnjitteredProjectionMatrix
        this._camera = undefined
    }
}

function halton(index: number, base: number) {
    let fraction = 1
    let result = 0
    while (index > 0) {
        fraction /= base
        result += fraction * (index % base)
        index = Math.floor(index / base)
    }
    return result
}

class DlssBridgeExportPass extends Pass {
    resetHistory() {
        this._historyValid = false
    }
    private readonly _viewer: ViewerApp
    private readonly _colorMaterial: ShaderMaterial
    private readonly _depthMaterial: ShaderMaterial
    private readonly _motionMaterial: ShaderMaterial
    private readonly _reactiveMaterial: ShaderMaterial
    private readonly _controlMaterial: ShaderMaterial
    private readonly _quad: FullScreenQuad

    private _colorTarget?: WebGLRenderTarget
    private _depthTarget?: WebGLRenderTarget
    private _motionTarget?: WebGLRenderTarget
    private _reactiveTarget?: WebGLRenderTarget
    private _controlTarget?: WebGLRenderTarget
    private _width = 0
    private _height = 0
    private _outputWidth = 0
    private _outputHeight = 0
    private _historyValid = false

    constructor(viewer: ViewerApp) {
        super()
        this._viewer = viewer
        this.needsSwap = false
        this.clear = false

        this._colorMaterial = makeMaterial(`
            uniform sampler2D tInput;
            varying vec2 vUv;
            void main() { gl_FragColor = vec4(texture2D(tInput, vUv).rgb, 1.0); }
        `, {tInput: {value: null}})

        this._depthMaterial = makeMaterial(`
            uniform sampler2D tDepth;
            varying vec2 vUv;
            void main() { gl_FragColor = vec4(texture2D(tDepth, vUv).r, 0.0, 0.0, 1.0); }
        `, {tDepth: {value: null}})

        this._motionMaterial = makeMaterial(`
            uniform sampler2D tVelocity;
            varying vec2 vUv;
            void main() {
                gl_FragColor = vec4(texture2D(tVelocity, vUv).xy, 0.0, 1.0);
            }
        `, {tVelocity: {value: null}})

        this._reactiveMaterial = makeMaterial(`
            uniform sampler2D tInput;
            uniform sampler2D tPrevious;
            uniform sampler2D tDepth;
            uniform sampler2D tFlags;
            uniform sampler2D tVelocity;
            uniform vec2 jitterDeltaUv;
            uniform float historyValid;
            varying vec2 vUv;

            vec3 compressHdr(vec3 value) {
                value = max(value, vec3(0.0));
                return value / (vec3(1.0) + value);
            }

            void main() {
                vec4 color = texture2D(tInput, vUv);
                float depth = texture2D(tDepth, vUv).r;
                float flagByte = floor(texture2D(tFlags, vUv).a * 255.0 + 0.5);
                // GemRefractionPlugin stores an inverted diamond flag in bit 4:
                // zero for diamond geometry, one for ordinary geometry/background.
                float diamond = 1.0 - mod(floor(flagByte / 16.0), 2.0);
                float covered = depth < 0.999999 ? 1.0 : 0.0;
                float transparency = covered * clamp(1.0 - color.a, 0.0, 1.0);

                // A diamond is view-dependent, but that does not make every
                // diamond pixel invalid every frame. Reproject the previous
                // pre-tonemap colour with the real motion vector and only bias
                // DLSS toward the current frame where the refracted/reflected
                // result actually changed. The explicit jitter delta aligns the
                // two jittered colour buffers; velocity itself is jitter-free.
                vec2 previousUv = vUv + texture2D(tVelocity, vUv).xy + jitterDeltaUv;
                float outside = step(previousUv.x, 0.0) + step(1.0, previousUv.x)
                    + step(previousUv.y, 0.0) + step(1.0, previousUv.y);
                float diamondBias = diamond;
                if (historyValid > 0.5) {
                    vec3 previous = texture2D(tPrevious, clamp(previousUv, 0.0, 1.0)).rgb;
                    vec3 delta = abs(compressHdr(color.rgb) - compressHdr(previous));
                    float change = max(delta.r, max(delta.g, delta.b));
                    diamondBias *= max(step(0.5, outside), smoothstep(0.035, 0.22, change));
                }

                gl_FragColor = vec4(max(transparency, diamondBias), 0.0, 0.0, 1.0);
            }
        `, {
            tInput: {value: null},
            tPrevious: {value: null},
            tDepth: {value: null},
            tFlags: {value: null},
            tVelocity: {value: null},
            jitterDeltaUv: {value: new Vector2()},
            historyValid: {value: 0},
        })

        // DLSSNR.ControlMask is independent from DLSS Super Resolution's
        // current-colour bias mask above. The signed runtime uses it as the
        // original-colour retention weight: zero accepts the neural result and
        // one preserves the source. Keep rendered geometry at zero and protect
        // the untouched canvas background with one.
        this._controlMaterial = makeMaterial(`
            uniform sampler2D tDepth;
            varying vec2 vUv;
            void main() {
                float depth = texture2D(tDepth, vUv).r;
                float preserveOriginal = step(0.999999, depth);
                gl_FragColor = vec4(preserveOriginal, 0.0, 0.0, 1.0);
            }
        `, {tDepth: {value: null}})

        this._quad = new FullScreenQuad(this._colorMaterial)
    }

    render(renderer: WebGLRenderer, _writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget) {
        const colorBridge = this._viewer.getPlugin(DlssBridgePlugin)
        if (colorBridge?.nrTemporal) {
            const motion = this._viewer.getPlugin(VelocityBufferPlugin)?.getVelocityBuffer()
            if (!motion) throw new Error('Temporal NR requires the velocity buffer')
            colorBridge.publishFrame({renderer, color: readBuffer, motion: motion as WebGLRenderTarget,
                renderWidth: readBuffer.width, renderHeight: readBuffer.height,
                outputWidth: readBuffer.width, outputHeight: readBuffer.height,
                jitter: {x: 0, y: 0}, previousJitter: {x: 0, y: 0}, reset: !this._historyValid})
            this._historyValid = true
            return
        }
        if (colorBridge?.colorOnly) {
            // The consumer reads synchronously before the composer reuses this
            // resolved pre-tonemap target. No extra textures or export passes.
            colorBridge.publishFrame({renderer, color: readBuffer,
                renderWidth: readBuffer.width, renderHeight: readBuffer.height,
                outputWidth: readBuffer.width, outputHeight: readBuffer.height,
                jitter: {x: 0, y: 0}, previousJitter: {x: 0, y: 0}, reset: false})
            return
        }
        const gbuffer = this._viewer.getPlugin(GBufferPlugin)
        const velocity = this._viewer.getPlugin(VelocityBufferPlugin)
        const depthTexture = gbuffer?.getDepthTexture()
        const flagsTexture = gbuffer?.getFlagsTexture()
        const velocityTexture = velocity?.getVelocityBuffer()?.texture
        if (!depthTexture || !flagsTexture || !velocityTexture) return

        const bridge = this._viewer.getPlugin(DlssBridgePlugin)
        // Feature 18 is native-resolution. When the browser page deliberately
        // runs NR without SR, its contract is the physical WebGI render itself,
        // including the stock WebVTO 2x display pixel ratio.
        const outputWidth = bridge?.preserveWebGiTemporal
            ? readBuffer.width
            : Math.max(1, Math.floor(this._viewer.renderer.renderSize.width))
        const outputHeight = bridge?.preserveWebGiTemporal
            ? readBuffer.height
            : Math.max(1, Math.floor(this._viewer.renderer.renderSize.height))
        this._ensureTargets(readBuffer.width, readBuffer.height, outputWidth, outputHeight)
        if ((bridge?.srOnly || bridge?.fgMode) && this._colorTarget && this._depthTarget && this._motionTarget) {
            this._colorMaterial.uniforms.tInput.value = readBuffer.texture
            this._depthMaterial.uniforms.tDepth.value = depthTexture
            this._motionMaterial.uniforms.tVelocity.value = velocityTexture
            const oldTarget = renderer.getRenderTarget()
            this._renderTarget(renderer, this._colorTarget, this._colorMaterial)
            this._renderTarget(renderer, this._depthTarget, this._depthMaterial)
            this._renderTarget(renderer, this._motionTarget, this._motionMaterial)
            renderer.setRenderTarget(oldTarget)
            bridge.publishFrame({renderer, color: this._colorTarget, depth: this._depthTarget,
                motion: this._motionTarget, renderWidth: this._width, renderHeight: this._height,
                outputWidth, outputHeight, jitter: {x: bridge.jitter.x, y: bridge.jitter.y},
                previousJitter: {x: bridge.previousJitter.x, y: bridge.previousJitter.y},
                reset: !this._historyValid})
            this._historyValid = true
            return
        }
        if (!this._colorTarget || !this._depthTarget || !this._motionTarget ||
            !this._reactiveTarget || !this._controlTarget) return

        this._colorMaterial.uniforms.tInput.value = readBuffer.texture
        this._depthMaterial.uniforms.tDepth.value = depthTexture
        this._motionMaterial.uniforms.tVelocity.value = velocityTexture
        this._reactiveMaterial.uniforms.tInput.value = readBuffer.texture
        this._reactiveMaterial.uniforms.tPrevious.value = this._colorTarget.texture
        this._reactiveMaterial.uniforms.tDepth.value = depthTexture
        this._reactiveMaterial.uniforms.tFlags.value = flagsTexture
        this._reactiveMaterial.uniforms.tVelocity.value = velocityTexture
        this._controlMaterial.uniforms.tDepth.value = depthTexture
        const jitter = bridge?.jitter
        const previousJitter = bridge?.previousJitter
        this._reactiveMaterial.uniforms.jitterDeltaUv.value.set(
            ((jitter?.x ?? 0) - (previousJitter?.x ?? 0)) / this._width,
            -((jitter?.y ?? 0) - (previousJitter?.y ?? 0)) / this._height,
        )
        this._reactiveMaterial.uniforms.historyValid.value = this._historyValid ? 1 : 0

        const oldTarget = renderer.getRenderTarget()
        const oldClearColor = renderer.getClearColor(new Color())
        const oldClearAlpha = renderer.getClearAlpha()
        renderer.setClearColor(0, 0)

        const reset = !this._historyValid

        // Initialise the history texture deterministically on the first frame.
        // On later frames it must remain untouched until the bias pass samples it.
        if (!this._historyValid) this._renderTarget(renderer, this._colorTarget, this._colorMaterial)
        this._renderTarget(renderer, this._depthTarget, this._depthMaterial)
        this._renderTarget(renderer, this._motionTarget, this._motionMaterial)
        this._renderTarget(renderer, this._reactiveTarget, this._reactiveMaterial)
        this._renderTarget(renderer, this._controlTarget, this._controlMaterial, this._outputWidth, this._outputHeight)
        // Keep the prior exported colour alive until after the confidence mask
        // has consumed it, then replace it with this frame's DLSS colour input.
        if (this._historyValid) this._renderTarget(renderer, this._colorTarget, this._colorMaterial)
        this._historyValid = true

        const exportedFrame: DlssBridgeFrame = {
            renderer,
            color: this._colorTarget,
            depth: this._depthTarget,
            motion: this._motionTarget,
            reactive: this._reactiveTarget,
            control: this._controlTarget,
            renderWidth: this._width,
            renderHeight: this._height,
            outputWidth: this._outputWidth,
            outputHeight: this._outputHeight,
            jitter: {x: jitter?.x ?? 0, y: jitter?.y ?? 0},
            previousJitter: {x: previousJitter?.x ?? 0, y: previousJitter?.y ?? 0},
            reset,
        }

        renderer.setRenderTarget(oldTarget)
        renderer.setClearColor(oldClearColor, oldClearAlpha)
        bridge?.publishFrame(exportedFrame)
    }

    private _renderTarget(renderer: WebGLRenderer, target: WebGLRenderTarget, material: ShaderMaterial,
                          usefulWidth = this._width, usefulHeight = this._height) {
        // RenderTarget.viewport is already in physical framebuffer pixels.
        // WebGLRenderer.setViewport() is deliberately not used here because it
        // multiplies by displayCanvasScaling. The composer dimensions passed to
        // this bridge are already scaled, so using setViewport() would apply the
        // DLSS render scale twice and leave most of the export texture unwritten.
        target.scissorTest = false
        target.viewport.set(0, 0, target.width, target.height)
        renderer.setRenderTarget(target)
        renderer.clear(true, false, false)

        // Keep the signature rows clear while drawing the useful WxH image at
        // its exact physical extent.
        target.viewport.set(0, 0, usefulWidth, usefulHeight)
        renderer.setRenderTarget(target)
        this._quad.material = material
        this._quad.render(renderer)
    }

    private _ensureTargets(width: number, height: number, outputWidth: number, outputHeight: number) {
        if (!this._colorTarget) {
            this._colorTarget = this._createTarget(width, height + 4, HalfFloatType, RGBAFormat, 'WEBGI_DLSS_COLOR_V1')
            this._colorTarget.texture.minFilter = LinearFilter
            this._colorTarget.texture.magFilter = LinearFilter
            this._depthTarget = this._createTarget(width, height + 3, FloatType, RedFormat, 'WEBGI_DLSS_DEPTH_V1')
            this._motionTarget = this._createTarget(width, height + 2, HalfFloatType, RGFormat, 'WEBGI_DLSS_MOTION_V1')
            if (!this._viewer.getPlugin(DlssBridgePlugin)?.srOnly) {
                this._reactiveTarget = this._createTarget(width, height + 1, UnsignedByteType, RedFormat, 'WEBGI_DLSS_REACTIVE_V1')
                this._controlTarget = this._createTarget(outputWidth, outputHeight + 5, UnsignedByteType, RedFormat, 'WEBGI_DLSS_NR_CONTROL_V1')
            }
        } else if (width !== this._width || height !== this._height ||
                   outputWidth !== this._outputWidth || outputHeight !== this._outputHeight) {
            this._colorTarget.setSize(width, height + 4)
            this._depthTarget!.setSize(width, height + 3)
            this._motionTarget!.setSize(width, height + 2)
            this._reactiveTarget?.setSize(width, height + 1)
            this._controlTarget?.setSize(outputWidth, outputHeight + 5)
            this._historyValid = false
        }
        this._width = width
        this._height = height
        this._outputWidth = outputWidth
        this._outputHeight = outputHeight
    }

    private _createTarget(width: number, height: number, type: number, format: number, name: string) {
        const target = this._viewer.renderer.createTarget({
            size: {width, height},
            type: type as any,
            format,
            colorSpace: NoColorSpace,
            depthBuffer: false,
            generateMipmaps: false,
            minFilter: NearestFilter,
            magFilter: NearestFilter,
        }) as WebGLRenderTarget
        target.texture.name = name
        return target
    }

    setSize() {
        // Explicit target sizes are derived from the actual composer read buffer.
    }

    dispose() {
        for (const target of [this._colorTarget, this._depthTarget, this._motionTarget,
            this._reactiveTarget, this._controlTarget]) {
            if (target) this._viewer.renderer.disposeTarget(target as any)
        }
        this._quad.dispose()
        this._colorMaterial.dispose()
        this._depthMaterial.dispose()
        this._motionMaterial.dispose()
        this._reactiveMaterial.dispose()
        this._controlMaterial.dispose()
        super.dispose()
    }
}

function makeMaterial(fragmentShader: string, uniforms: Record<string, {value: any}>) {
    return new ShaderMaterial({
        vertexShader: CopyShader.vertexShader,
        fragmentShader,
        uniforms,
        blending: NoBlending,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
    })
}
