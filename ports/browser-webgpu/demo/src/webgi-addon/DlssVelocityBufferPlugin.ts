import {IRenderTarget, IShaderPropertiesUpdater, IViewerPluginAsync, UiObjectConfig} from 'webgi/interfaces'
import {
    HalfFloatType,
    IUniform,
    Material,
    RGFormat,
    ShaderMaterial,
    UnsignedByteType,
    WebGLRenderer,
    WebGLRenderTarget,
} from 'three'
import {DlssVelocityPass} from './passes/DlssVelocityPass'
import {GenericFilterPlugin} from 'webgi/plugins/GenericFilterPlugin'
import {setThreeRendererMode} from 'webgi/helpers/threejs/threeUtils'
import {ViewerApp} from 'webgi/viewer/ViewerApp'
import type {DebugPlugin} from 'webgi/plugins/DebugPlugin'

export class DlssVelocityBufferPlugin extends GenericFilterPlugin<DlssVelocityPass, 'velocityBuffer', ''> implements IViewerPluginAsync, IShaderPropertiesUpdater {
    passId: 'velocityBuffer' = 'velocityBuffer'

    protected _beforeFilters = ['render']
    protected _afterFilters = []
    protected _requiredFilters = ['render']

    static readonly PluginType = 'VelocityBuffer'
    private _velocityBuffers: IRenderTarget[] = []
    // private _velocityBufferPass?: IFilter<SSVelocityPass>

    passCtor(v: ViewerApp): DlssVelocityPass {
        const rawOutput = this.rawOutput
        const target = v.renderer.createTarget({
            depthBuffer: true,
            type: rawOutput ? HalfFloatType : UnsignedByteType,
            ...rawOutput ? {format: RGFormat} : {},
        })
        target.texture.name = 'velocityBuffer'
        this._velocityBuffers.push(target)

        const debug = v.getPluginByType<DebugPlugin>('debug')
        if (debug) {
            // debug.addTexture('velocityBuffer', ()=> target.texture, [40, 50, 400, 200])
        }

        const transparentMats = new Set<Material>()
        const transmissiveMats = new Set<[Material, number]>()
        const includeTransparent = this.includeTransparent
        const continuous = this.continuous
        return new class DlssVelocityPass2 extends DlssVelocityPass {
            render(renderer: WebGLRenderer, writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget, deltaTime: number, maskActive: boolean) {
                if (!continuous && v.renderer.frameCount > 0) return

                const t = renderer.getRenderTarget()
                const activeCubeFace = renderer.getActiveCubeFace()
                const activeMipLevel = renderer.getActiveMipmapLevel()
                // renderer.setRenderTarget(target)

                this.scene?.traverse(({material}: any) => {
                    if (!material) return
                    const forceRender = includeTransparent || material.userData.renderToDepth
                        && !material.userData.pluginsDisabled && !material.userData[DlssVelocityBufferPlugin.PluginType]?.disabled
                    // && material.colorWrite === true
                    const doNotRender = material.userData.renderToDepth === false
                        || material.userData.pluginsDisabled || material.userData[DlssVelocityBufferPlugin.PluginType]?.disabled
                    // || material.colorWrite === false
                    if (
                        material.transparent && forceRender || // transparent and render to depth
                        !material.transparent && !material.transmission && doNotRender // opaque and render to depth
                    ) {
                        transparentMats.add(material)
                        material.transparent = !material.transparent
                    }
                    if (
                        Math.abs(material.transmission || 0) > 0 && forceRender // transmission and render to depth
                    ) {
                        transmissiveMats.add([material, material.transmission])
                        material.transmission = 0
                    }
                    // if (!material.transparent && !material.transmission &&
                    //     (material.userData.pluginsDisabled ||
                    //         material.userData[VelocityBufferPlugin.PluginType]?.disabled)) {
                    //     transparentMats.add(material)
                    //     material.transparent = true
                    // }
                })

                // todo; copy double sided.

                setThreeRendererMode(renderer, {
                    shadowMapRender: false,
                    backgroundRender: false,
                    opaqueRender: true,
                    transparentRender: false,
                    transmissionRender: false,
                    mainRenderPass: false,
                }, ()=> super.render(renderer, writeBuffer, target as any, deltaTime, maskActive))

                transparentMats.forEach(m => m.transparent = !m.transparent)
                transparentMats.clear()

                transmissiveMats.forEach(([m, tr]: [any, number]) => m.transmission = tr)
                transmissiveMats.clear()

                renderer.setRenderTarget(t, activeCubeFace, activeMipLevel)

            }
        }(undefined, undefined, undefined, rawOutput)
    }

    protected _update(v: ViewerApp): boolean {
        if (!super._update(v)) return false
        if (!this.continuous && v.renderer.frameCount > 0) return false
        const pass = this.pass!.passObject
        pass.scene = v.scene.modelObject
        v.scene.renderCamera.updateShaderProperties(pass.overrideMaterial as ShaderMaterial)
        pass.camera = v.scene.renderCamera.cameraObject!

        return true
    }

    constructor(
        enabled = true,
        public includeTransparent = false,
        public continuous = false,
        public rawOutput = false,
        public exposeToPostProcessing = true,
    ) {
        super()
        this.enabled = enabled
    }

    getVelocityBuffer() {
        return this._velocityBuffers.length > 0 ? this._velocityBuffers[0] : undefined
    }

    async onDispose(viewer: ViewerApp): Promise<void> {
        return
    }

    async onRemove(viewer: ViewerApp): Promise<void> {
        this._velocityBuffers.forEach(value => viewer.renderer.disposeTarget(value?.dispose?.() as any))
        return super.onRemove(viewer)
    }

    updateShaderProperties(material: {defines: Record<string, string | number | undefined>; uniforms: {[p: string]: IUniform}}): this {
        // NR's signed motion is consumed by its bridge. Existing WebGI effects
        // expect the encoded velocity format and keep their prior input path.
        if (!this.exposeToPostProcessing) return this
        if (material.uniforms.tVelocity) material.uniforms.tVelocity.value = this.enabled ? this.getVelocityBuffer()?.texture ?? null : null
        else console.warn('BaseRenderer: no uniform: tVelocity')
        return this
    }
    public get uiConfig(): UiObjectConfig | undefined {
        return this.pass?.passObject.uiConfig
        // return {}
    }

}

// Scene modules still import the conventional WebGI name. The standalone
// build maps that name to this local implementation and preserves PluginType.
export {DlssVelocityBufferPlugin as VelocityBufferPlugin}
