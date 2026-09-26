import {IEvent, onChange} from 'ts-browser-helpers'
import {IViewerPluginAsync, UiObjectConfig} from 'webgi/interfaces'
import {
    BasicShadowMap,
    BufferGeometry,
    LinearFilter,
    LinearMipmapLinearFilter,
    Mesh,
    MeshStandardMaterial,
    NoColorSpace,
    PCFShadowMap,
    PCFSoftShadowMap,
    RGBAFormat,
    UnsignedByteType,
    VSMShadowMap,
    WebGLRenderTarget,
} from 'three'
import {DebugPlugin} from 'webgi/plugins/DebugPlugin'
import {ShadowMapBaker} from 'webgi/extras/ShadowMapBaker'
import {DlssReflector2} from 'webgi/helpers/threejs/Reflector2'
import type {SSRPlugin} from 'webgi/plugins/SSRPlugin'
import {serialize} from 'webgi/helpers/serialize'
import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {BaseGroundPlugin, GroundOptions} from 'webgi/plugins/BaseGroundPlugin'
import {SSBevelPlugin} from 'webgi/plugins/SSBevelPlugin'

export class DlssGroundPlugin<TMesh extends Mesh<BufferGeometry, MeshStandardMaterial> | DlssReflector2 = Mesh<BufferGeometry, MeshStandardMaterial> | DlssReflector2> extends BaseGroundPlugin<TMesh> implements IViewerPluginAsync {
    get shadowBaker(): ShadowMapBaker | undefined {
        return this._shadowBaker
    }

    // set mesh(value: IModel<TMesh>|undefined) {
    //     this._iMesh = value
    // }
    static readonly PluginType = 'Ground'

    @onChange(DlssGroundPlugin.prototype.refreshOptions)
    @serialize() bakedShadows = true
    @onChange(DlssGroundPlugin.prototype.refreshOptions)
    @serialize() groundReflection = false
    @onChange(DlssGroundPlugin.prototype.refreshOptions)
    @serialize() physicalReflections = false
    @onChange(DlssGroundPlugin.prototype.refreshOptions)
    @serialize() autoFrustumSize = true

    @serialize('shadowBaker')
    private _shadowBaker?: ShadowMapBaker
    private _showDebug: boolean

    /**
     * autoBakeShadows - when true, shadows are baked automatically on scene update(whenever any object in the scene changes), set it to `false` to trigger baking manually with {@see bakeShadows()}
     */
    autoBakeShadows = true

    /**
     * bake shadows manually, to be used with {@see autoBakeShadows} set to false
     */
    bakeShadows() {
        this._shadowBaker?.reset()
    }

    constructor(options: Partial<GroundOptions> = {}, showDebug = false) {
        super(options)
        this._showDebug = showDebug
        if (showDebug) this.dependencies.push(DebugPlugin)
        this._onSceneUpdate = this._onSceneUpdate.bind(this)

        // this.refreshOptions()
    }

    protected _createMesh(): Mesh {
        const reflector = new DlssReflector2(this._geometry, this._viewer!.renderer.createTarget({
            // type: HalfFloatType,
            type: UnsignedByteType,
            format: RGBAFormat,
            colorSpace: NoColorSpace, // todo: we can do rgbm if only opaque objects will be reflected
            size: {width: 1024, height: 1024},
            generateMipmaps: true,
            depthBuffer: true,
            minFilter: LinearMipmapLinearFilter,
            magFilter: LinearFilter,
            // isAntialiased: this._viewer.isAntialiased,
        }) as WebGLRenderTarget)
        const superOnBeforeRender = reflector.onBeforeRender
        reflector.onBeforeRender = (...params) => {
            let ssr = this._viewer?.getPluginByType<SSRPlugin>('SSReflection')?.passes.ssr.passObject
            if (ssr && !ssr.enabled) ssr = undefined
            if (ssr) ssr.enabled = false // todo: do we need to disable ssao also?

            let ssbevel = this._viewer?.getPluginByType<SSBevelPlugin>('SSBevelPlugin')?.pass?.passObject
            if (ssbevel && !ssbevel.enabled) ssbevel = undefined
            if (ssbevel) ssbevel.enabled = false // todo: do we need to disable ssao also?

            superOnBeforeRender(...params)

            if (ssr) ssr.enabled = true
            if (ssbevel) ssbevel.enabled = true
        }
        return reflector

    }

    async onAdded(viewer: ViewerApp): Promise<void> {
        await super.onAdded(viewer)
        if (this._showDebug) {
            viewer.getPlugin(DebugPlugin)?.addTexture('bake_ground_1', () => {
                return (this._shadowBaker?.light.shadow.map as WebGLRenderTarget|undefined)?.texture
            }, [100, 100, 200, 200])
            viewer.getPlugin(DebugPlugin)?.addTexture('bake_ground_2', () => {
                return (this._shadowBaker?.target as WebGLRenderTarget)?.texture
            }, [100, 400, 400, 400], 'texel = vec4(vec3(unpackRGBAToDepth(texel)), 1.0);')
        }
    }

    protected _postFrame() {
        super._postFrame()
        if (!this._viewer) return
        if (!this.enabled) return
        if (this._shadowBaker && this.bakedShadows) {
            this._shadowBaker.autoUpdateShadow()
        }
    }

    protected _preRender() {
        super._preRender()
        if (!this._viewer) return
        ;(this._mesh as DlssReflector2).reflectionTargetNeedsUpdate = this._viewer.renderer.frameCount < 1
    }

    async onDispose(viewer: ViewerApp): Promise<void> {
        return super.onDispose(viewer)
    }

    async onRemove(viewer: ViewerApp): Promise<void> {
        return super.onRemove(viewer)
    }

    protected _removeMaterial() {
        if (!this._material) return
        if (this._shadowBaker && (this._material as any).groundMatExtension) {
            this._material.unregisterMaterialExtensions?.([this._shadowBaker.materialExtension])
            delete (this._material as any).groundMatExtension
        }
        if ((this._material as any).reflectorMatExtension) {
            const ext = (this._mesh as DlssReflector2).materialExtension
            if (!ext) console.warn('WebGi GroundPlugin: unable to find the extension to unregister')
            this._material.unregisterMaterialExtensions?.([ext])
            delete (this._material as any).reflectorMatExtension
        }
        // todo: remove map or render target thats assigned

        super._removeMaterial()
    }

    protected _onSceneUpdate(event: IEvent<string>) {
        super._onSceneUpdate(event)
        if (event.geometryChanged === false) return
        if (this.autoBakeShadows) this._shadowBaker?.reset()
    }

    public refreshOptions(): void {
        if (!this._viewer) return
        if (this.bakedShadows && !this._shadowBaker) {
            this._shadowBaker = new ShadowMapBaker(this._viewer)
            this._shadowBaker.attachedMesh = this._mesh
        } else if (!this.bakedShadows && this._shadowBaker) {
            this._shadowBaker.reset()
            this._shadowBaker.cleanupMaterial()
        }
        const ref = this._mesh as DlssReflector2
        if (ref.isReflector2) {
            ref.enabled = this.groundReflection
            ref.reflectorModePhysical = this.physicalReflections
        }
        super.refreshOptions()
        this._viewer.setDirty(this)
    }

    protected _refreshTransform() {
        if (this.autoFrustumSize) {
            const baker = this.shadowBaker
            if (baker) {
                const fs = this.size / 2
                if (fs !== baker.light.shadowParams.frustumSize) {
                    baker.light.shadowParams.frustumSize = fs
                    baker.light.updateShadowParams()
                    baker.reset()
                }
                // ground.bakeShadows()
            }
        }
        super._refreshTransform()
    }

    // see BaseGroundPlugin
    fromJSON(data: any, meta?: any): this | null {
        if (!super.fromJSON(data, meta)) return null
        if (data.autoFrustumSize === undefined) this.autoFrustumSize = false // for files which were saved before this option was added.
        return this
    }

    protected _refreshMaterial(): boolean {
        if (!this._viewer) return false
        const isNewMaterial = super._refreshMaterial()
        if (!this._material) return isNewMaterial
        // if (isNewMaterial) this._material.transparent = true
        if (this.groundReflection && (this._mesh as DlssReflector2).isReflector2 && !(this._material as any).reflectorMatExtension) {
            const ext = (this._mesh as DlssReflector2).materialExtension
            ext.updaters = [this._viewer.scene, this._viewer.renderer]
            this._material.registerMaterialExtensions?.([ext])
            ;(this._material as any).reflectorMatExtension = true
        }
        if (this.bakedShadows && this._shadowBaker && !(this._material as any).groundMatExtension) {
            this._material.registerMaterialExtensions?.([this._shadowBaker.materialExtension])
            ;(this._material as any).groundMatExtension = true
        }

        this._material.materialObject.userData.ssreflDisabled = this.groundReflection
        this._material.materialObject.userData.ssreflNonPhysical = !this.physicalReflections
        this._viewer.setDirty(this)
        return isNewMaterial
    }

    protected _extraUiConfig(): (UiObjectConfig | (() => UiObjectConfig|UiObjectConfig[]))[] {
        return [
            {
                label: 'Baked Shadows',
                type: 'checkbox',
                property: [this, 'bakedShadows'],
            },
            {
                label: 'Shadow Frames',
                type: 'input',
                hidden: ()=> !this._shadowBaker,
                stepSize: 1,
                bounds: [1, 1000],
                property: [this._shadowBaker, 'maxFrameNumber'],
            },
            {
                label: 'Alpha Vignette',
                type: 'checkbox',
                hidden: ()=> !this._material || this._material.transmission < 0.0001 && !this._material.transparent,
                property: [this._shadowBaker, 'alphaVignette'],
                limitedUi: true,
                onChange: ()=>this._uiConfig?.uiRefresh?.('postFrame', true),
            },
            {
                label: 'Alpha Vignette Axis',
                type: 'dropdown',
                hidden: ()=> !this._shadowBaker?.alphaVignette || !this._material || this._material.transmission < 0.0001 && !this._material.transparent,
                property: [this._shadowBaker, 'alphaVignetteAxis'],
                children: ['x', 'y', 'xy'].map(v => ({label: v, value: v})),
                limitedUi: true,
            },
            {
                label: 'Planar Reflections',
                type: 'checkbox',
                property: [this, 'groundReflection'],
            },
            {
                label: 'Auto Frustum Size',
                type: 'checkbox',
                property: [this, 'autoFrustumSize'],
            },
            {
                label: 'Physical Reflections',
                type: 'checkbox',
                // hidden: ()=> !this._options.groundReflection || !(this._mesh as Reflector2).isReflector2,
                // property: [this._mesh as Reflector2, 'reflectorModePhysical'],
                property: [this, 'physicalReflections'],
                limitedUi: true,
            },
            {
                label: 'Shadow type',
                type: 'dropdown',
                hidden: ()=> !this._shadowBaker,
                property: [this._shadowBaker, 'groundMapMode'],
                children: [
                    {label: 'aoMap'},
                    {label: 'map'},
                    {label: 'alphaMap'},
                ],
                limitedUi: true,
            },
            {
                label: 'Smooth Shadow',
                type: 'checkbox',
                property: [this._shadowBaker, 'smoothShadow'],
            },
            {
                label: 'Baked shadow type',
                type: 'dropdown',
                children: [['Basic', BasicShadowMap], ['PCF', PCFShadowMap], ['PCFSoft', PCFSoftShadowMap], ['VSM', VSMShadowMap]].map((v) => ({label: v[0].toString(), value: v[1]})),
                property: [this._shadowBaker, 'shadowMapType'],
            },
            {
                type: 'folder',
                label: 'Randomized Light',
                hidden: ()=> !this._shadowBaker,
                limitedUi: true,
                children: [
                    {
                        type: 'color',
                        label: 'Color',
                        property: [this._shadowBaker?.light, 'color'],
                    },
                    {
                        type: 'slider',
                        label: 'Intensity',
                        bounds: [0, 100],
                        property: [this._shadowBaker?.light, 'intensity'],
                    },
                    {
                        type: 'checkbox',
                        label: 'Shadow Enabled',
                        property: [this._shadowBaker?.light?.shadowParams, 'enabled'],
                        onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate],
                    },
                    {
                        type: 'slider',
                        bounds: [0.0, 1],
                        property: [this._shadowBaker?.light?.randomParams, 'focus'],
                        onChange: [this._onSceneUpdate],
                    },
                    {
                        type: 'slider',
                        bounds: [0.0, 1],
                        property: [this._shadowBaker?.light?.randomParams, 'spread'],
                        onChange: [this._onSceneUpdate],
                        limitedUi: true,
                    },
                    {
                        type: 'slider',
                        bounds: [0.01, 60],
                        property: [this._shadowBaker?.light?.randomParams, 'distanceScale'],
                        onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate],
                    },
                    {
                        type: 'vec3',
                        bounds: [-1, 1],
                        property: [this._shadowBaker?.light?.randomParams, 'direction'],
                        onChange: [this._onSceneUpdate],
                        limitedUi: true,
                    },
                    {
                        type: 'vec3',
                        bounds: [-1, 1],
                        property: [this._shadowBaker?.light?.randomParams, 'normalDirection'],
                        onChange: [this._onSceneUpdate],
                        limitedUi: true,
                    },
                    {
                        type: 'slider',
                        bounds: [0.01, 10],
                        property: [this._shadowBaker?.light?.shadowParams, 'radius'],
                        onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate],
                    },
                    {
                        type: 'input',
                        property: [this._shadowBaker?.light?.shadowParams, 'frustumSize'],
                        hidden: ()=> this.autoFrustumSize,
                        onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate],
                    },
                    {
                        type: 'slider',
                        bounds: [-0.1, 0.1],
                        property: [this._shadowBaker?.light?.shadowParams, 'bias'],
                        onChange: [this._shadowBaker?.light?.updateShadowParams, this._onSceneUpdate],
                    },
                ],
            },
            ...super._extraUiConfig(),
        ]
    }
}

export {DlssGroundPlugin as GroundPlugin}

