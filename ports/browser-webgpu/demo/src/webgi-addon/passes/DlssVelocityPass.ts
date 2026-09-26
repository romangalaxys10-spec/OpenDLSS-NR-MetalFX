import {
    BufferGeometry,
    Camera,
    Color,
    DoubleSide, Group,
    IUniform,
    Material,
    Matrix3,
    Matrix4,
    Object3D,
    Scene,
    ShaderMaterial,
    Vector2,
    WebGLRenderer,
    WebGLRenderTarget,
} from 'three'
import {RenderPass} from 'three/examples/jsm/postprocessing/RenderPass'
import ssVelocityVert from 'webgi/plugins/shaders/ssVelocityVert.glsl'
import ssVelocityFrag from 'webgi/plugins/shaders/ssVelocityFrag.glsl'
import {UiObjectConfig} from 'webgi/interfaces'
import {uiFolder, uiToggle} from 'webgi/ui/uiConfigDecorators'

@uiFolder('Velocity Buffer (TAA)')
export class DlssVelocityPass extends RenderPass { // todo: extend from jittered

    @uiToggle('Enabled')
    enabled = true

    constructor(scene?: Scene, camera?: Camera, overrideMaterial?: Material, rawOutput = false) {
        super(
            scene,
            camera,
            overrideMaterial ?? new SSVelocityMaterial(rawOutput),
            rawOutput ? new Color(0, 0, 0) : new Color(0.5, 0.5, 0.5),
            1,
        ) // encoded output uses 0.5 for zero; signed RG16F output uses 0.
    }

    private _firstCall = true
    render(renderer: WebGLRenderer, writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget, deltaTime: number, maskActive: boolean) {
        if (!this.enabled || !this.camera) return
        const mat = this.overrideMaterial as ShaderMaterial
        // OrbitControls updates the camera after the preceding render. Refresh
        // matrixWorldInverse before building the current PV matrix; otherwise
        // velocity lags the colour render by one frame during camera movement.
        this.camera.updateMatrixWorld(true)
        const projection = this.camera.userData.dlssUnjitteredProjectionMatrix ?? this.camera.projectionMatrix
        mat.uniforms.currentProjectionViewMatrix.value.copy(projection).multiply(this.camera.matrixWorldInverse)
        if (this._firstCall) {
            mat.uniforms.lastProjectionViewMatrix.value.copy(mat.uniforms.currentProjectionViewMatrix.value)
            this._firstCall = false
        }
        // The override material remains bound across consecutive scene renders.
        // Three.js can therefore reuse the program without refreshing ordinary
        // ShaderMaterial uniforms. These two matrices change every frame and
        // must be uploaded even when the material/program did not change.
        mat.uniformsNeedUpdate = true
        super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive)
        mat.uniforms.lastProjectionViewMatrix.value.copy(mat.uniforms.currentProjectionViewMatrix.value)
    }

    uiConfig?: UiObjectConfig
}

export {DlssVelocityPass as SSVelocityPass}
class SSVelocityMaterial extends ShaderMaterial {

    constructor(rawOutput = false) {
        super({
            vertexShader: ssVelocityVert,
            fragmentShader: ssVelocityFrag,
            uniforms: {
                cameraNearFar: {value: new Vector2(0.1, 1000)},
                alphaMap: {value: null},
                alphaTest: {value: null},
                alphaMapTransform: {value: /* @__PURE__*/ new Matrix3()},
                currentProjectionViewMatrix: {value: new Matrix4()},
                lastProjectionViewMatrix: {value: new Matrix4()},
            },
            defines: rawOutput ? {VELOCITY_RAW: 1} : {},
        })
    }

    extraUniformsToUpload: Record<string, IUniform> = {
        modelMatrixPrevious: {value: new Matrix4().identity()},
    }

    private _previousWorldMatrices: Record<string, Matrix4> = {}

    // this gets called for each object.
    onBeforeRender(renderer: WebGLRenderer, s: Scene, c: Camera, geometry: BufferGeometry, object: Object3D, group: Group) {
        super.onBeforeRender(renderer, s, c, geometry, object, group)

        const prevMatrix = this._previousWorldMatrices[object.uuid]
        this.extraUniformsToUpload.modelMatrixPrevious.value.copy(prevMatrix ?? object.matrixWorld)

        // todo: make sure all objects are only rendered once.
        if (prevMatrix) {
            prevMatrix.copy(object.matrixWorld)
        } else {
            this._previousWorldMatrices[object.uuid] = object.matrixWorld.clone()
        }

        // todo: add support for all this in the shaders.
        let mat = (object as any).material

        if (Array.isArray(mat)) { // todo: add support for multi materials.
            mat = mat[0]
        }
        this.uniforms.alphaMap.value = mat?.alphaMap ?? null
        this.uniforms.alphaTest.value = !mat || !mat.alphaTest || mat.alphaTest < 0.0000001 ? 0.001 : mat.alphaTest

        let x = this.uniforms.alphaMap.value ? 1 : undefined
        if (x !== this.defines.USE_ALPHAMAP) {
            if (x === undefined) {
                delete this.defines.USE_ALPHAMAP
                delete this.defines.ALPHAMAP_UV
            } else {
                this.defines.USE_ALPHAMAP = x
                // Required by three.js's shared uv_vertex shader chunk.
                this.defines.ALPHAMAP_UV = 'uv'
            }
            this.needsUpdate = true
        }
        x = mat.userData.ALPHA_I_RGBA_PACKING ? 1 : undefined
        if (x !== this.defines.ALPHA_I_RGBA_PACKING) {
            if (x === undefined) delete this.defines.ALPHA_I_RGBA_PACKING
            else this.defines.ALPHA_I_RGBA_PACKING = x
            this.needsUpdate = true
        }

        this.side = mat.side ?? DoubleSide


    }
}
