import {
    BufferGeometry,
    MathUtils,
    Matrix4,
    Mesh,
    MeshPhysicalMaterial,
    PerspectiveCamera,
    Plane,
    Vector2,
    Vector3,
    Vector4,
    WebGLRenderTarget,
} from 'three'
import {MaterialExtension} from 'webgi/extras/asset_manager/threejs/MaterialExtender'
import {onChange} from 'ts-browser-helpers'
import {MeshStandardMaterial2} from 'webgi/extras/asset_manager/threejs/MeshStandardMaterial2'
import {IMaterial} from 'webgi/interfaces'
import {setThreeRendererMode} from 'webgi/helpers/threejs/threeUtils'

import reflectorSample from 'webgi/plugins/shaders/reflectorSample.glsl'
import randomHelpers from 'webgi/plugins/shaders/randomHelpers.glsl'
import poissonDiskSamples from 'webgi/plugins/shaders/poissonDiskSamples.glsl'
import {shaderReplaceString} from 'webgi/helpers/material'

export class DlssReflector2 extends Mesh<BufferGeometry, MeshPhysicalMaterial> {
    public type = 'Reflector'
    public readonly isReflector2 = true
    private _renderTarget: WebGLRenderTarget
    // private _renderTarget2: WebGLRenderTarget
    public readonly textureMatrix: Matrix4

    @onChange(DlssReflector2.prototype._updateExtension)
    public enabled = true

    @onChange(DlssReflector2.prototype._updateExtension)
    public reflectorModePhysical = true // shows envmap in reflection

    public reflectionTargetNeedsUpdate = true
    public transparentReflectionBackground = true // todo change to reflectBackground or something

    private _updateExtension() {
        this.transparentReflectionBackground = this.reflectorModePhysical
        this.materialExtension?.setDirty?.()
    }

    constructor(geometry: BufferGeometry, renderTarget: WebGLRenderTarget, clipBias = 0) {
        super(geometry)
        this.material = undefined as any // todo: test this...

        this._renderTarget = renderTarget
        // todo: do we need to dispose this on RT dispose??
        // this._renderTarget.depthTexture = new DepthTexture(renderTarget.width / 2, renderTarget.height / 2, UnsignedByteType)

        const reflectorPlane = new Plane()
        const normal = new Vector3()
        const reflectorWorldPosition = new Vector3()
        const cameraWorldPosition = new Vector3()
        const rotationMatrix = new Matrix4()
        const lookAtPosition = new Vector3(0, 0, -1)
        const clipPlane = new Vector4()

        const view = new Vector3()
        const target = new Vector3()
        const q = new Vector4()

        const textureMatrix = new Matrix4()
        const virtualCamera = new PerspectiveCamera()

        if (!MathUtils.isPowerOfTwo(renderTarget.texture.image.width) || !MathUtils.isPowerOfTwo(renderTarget.texture.image.height))
            this._renderTarget.texture.generateMipmaps = false

        this.onBeforeRender = (renderer, scene, camera) => {

            if (!this.enabled || !(renderer as any).userData.mainRenderPass) return // mainRenderPass is set in RenderPass2

            if (!this.reflectionTargetNeedsUpdate) {
                return
            }

            const viewOffset = (camera as PerspectiveCamera).view ? Object.assign({}, (camera as PerspectiveCamera).view) : null
            // clear camera jitter
            viewOffset && (camera as PerspectiveCamera).clearViewOffset && (camera as PerspectiveCamera).clearViewOffset()

            reflectorWorldPosition.setFromMatrixPosition(this.matrixWorld)
            cameraWorldPosition.setFromMatrixPosition(camera.matrixWorld)

            rotationMatrix.extractRotation(this.matrixWorld)

            normal.set(0, 0, 1)
            normal.applyMatrix4(rotationMatrix)

            view.subVectors(reflectorWorldPosition, cameraWorldPosition)

            // Avoid rendering when reflector is facing away

            if (view.dot(normal) > 0) return

            view.reflect(normal).negate()
            view.add(reflectorWorldPosition)

            rotationMatrix.extractRotation(camera.matrixWorld)

            lookAtPosition.set(0, 0, -1)
            lookAtPosition.applyMatrix4(rotationMatrix)
            lookAtPosition.add(cameraWorldPosition)

            target.subVectors(reflectorWorldPosition, lookAtPosition)
            target.reflect(normal).negate()
            target.add(reflectorWorldPosition)

            virtualCamera.position.copy(view)
            virtualCamera.up.set(0, 1, 0)
            virtualCamera.up.applyMatrix4(rotationMatrix)
            virtualCamera.up.reflect(normal)
            virtualCamera.lookAt(target)

            virtualCamera.far = 2. // Used in WebGLBackground
            virtualCamera.near = 0. // Used in WebGLBackground

            virtualCamera.updateMatrixWorld()
            virtualCamera.projectionMatrix.copy(camera.projectionMatrix)

            // Update the texture matrix
            textureMatrix.set(
                0.5, 0.0, 0.0, 0.5,
                0.0, 0.5, 0.0, 0.5,
                0.0, 0.0, 0.5, 0.5,
                0.0, 0.0, 0.0, 1.0
            )
            textureMatrix.multiply(virtualCamera.projectionMatrix)
            textureMatrix.multiply(virtualCamera.matrixWorldInverse)
            textureMatrix.multiply(this.matrixWorld)

            // Now update projection matrix with new clip plane, implementing code from: http://www.terathon.com/code/oblique.html
            // Paper explaining this technique: http://www.terathon.com/lengyel/Lengyel-Oblique.pdf
            reflectorPlane.setFromNormalAndCoplanarPoint(normal, reflectorWorldPosition)
            reflectorPlane.applyMatrix4(virtualCamera.matrixWorldInverse)

            clipPlane.set(reflectorPlane.normal.x, reflectorPlane.normal.y, reflectorPlane.normal.z, reflectorPlane.constant)

            const projectionMatrix = virtualCamera.projectionMatrix

            q.x = (Math.sign(clipPlane.x) + projectionMatrix.elements[ 8 ]) / projectionMatrix.elements[ 0 ]
            q.y = (Math.sign(clipPlane.y) + projectionMatrix.elements[ 9 ]) / projectionMatrix.elements[ 5 ]
            q.z = -1.0
            q.w = (1.0 + projectionMatrix.elements[ 10 ]) / projectionMatrix.elements[ 14 ]

            // Calculate the scaled plane vector
            clipPlane.multiplyScalar(2.0 / clipPlane.dot(q))

            // Replacing the third row of the projection matrix
            projectionMatrix.elements[ 2 ] = clipPlane.x
            projectionMatrix.elements[ 6 ] = clipPlane.y
            projectionMatrix.elements[ 10 ] = clipPlane.z + 1.0 - clipBias
            projectionMatrix.elements[ 14 ] = clipPlane.w

            // Render
            this.visible = false

            const currentRenderTarget = renderer.getRenderTarget()

            const currentXrEnabled = renderer.xr.enabled
            const currentShadowAutoUpdate = renderer.shadowMap.autoUpdate

            renderer.xr.enabled = false // Avoid camera modification
            renderer.shadowMap.autoUpdate = false // Avoid re-computing shadows

            renderer.setRenderTarget(this._renderTarget)

            renderer.state.buffers.depth.setMask(true) // make sure the depth buffer is writable so it can be properly cleared, see #18897

            if (renderer.autoClear === false) renderer.clear()

            const sceneBackground = scene.background as any
            // console.log((sceneBackground as any)?.uuid)
            // backgroundRender below suppresses the background for this pass.
            // Mutating RootScene.background here emits a scene update and starts
            // FrameFade on every reflection, preventing progressive convergence.

            const renderBackground = !this.transparentReflectionBackground
            if (sceneBackground?.isTexture && renderBackground) {
                if (!sceneBackground.userData) sceneBackground.userData = {}
                sceneBackground.userData.flipX = !sceneBackground.userData.flipX
            }

            setThreeRendererMode(renderer, {
                shadowMapRender:false,
                backgroundRender: renderBackground,
                opaqueRender:true,
                transparentRender:true,
                transmissionRender:false, // todo: render transmissive objects somehow
                screenSpaceRendering:false,
            }, ()=> renderer.render(scene, virtualCamera))

            if (sceneBackground?.isTexture && renderBackground) {
                sceneBackground.userData.flipX = !sceneBackground.userData.flipX || undefined
            }

            renderer.xr.enabled = currentXrEnabled
            renderer.shadowMap.autoUpdate = currentShadowAutoUpdate

            renderer.setRenderTarget(currentRenderTarget)

            if (viewOffset?.enabled && (camera as PerspectiveCamera).setViewOffset)
                (camera as PerspectiveCamera).setViewOffset(viewOffset.fullWidth, viewOffset.fullHeight, viewOffset.offsetX, viewOffset.offsetY, viewOffset.width, viewOffset.height)

            // Restore viewport

            const viewport = (camera as any).viewport

            if (viewport !== undefined) {

                renderer.state.viewport(viewport)

            }

            this.visible = true

            this.reflectionTargetNeedsUpdate = false

        }

        this.textureMatrix = textureMatrix
        this.materialExtension.extraUniforms!.tRefDiffuse.value = this._renderTarget.texture
        this.materialExtension.extraUniforms!.tRefDiffuseSize.value = new Vector2(this._renderTarget.width, this._renderTarget.height)
        // this.materialExtension.extraUniforms!.tRefDepth.value = this._renderTarget.depthTexture
        this.materialExtension.extraUniforms!.refTextureMatrix.value = textureMatrix

    }

    getRenderTarget() {
        return this._renderTarget
    }

    readonly materialExtension: MaterialExtension = {
        extraUniforms: {
            // tRefDepth: {value: null},
            tRefDiffuse: {value: null},
            tRefDiffuseSize: {value: new Vector2()},
            refTextureMatrix: {value: null},
            frameCount: {value: 0},
            sceneBoundingRadius: {value: 0},
        },
        extraDefines: {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            USE_UV: '',
        },
        updaters: [],
        shaderExtender: (shader, material, renderer) => {
            if (this.enabled) {
                shader.vertexShader = shaderReplaceString(shader.vertexShader, 'void main() {', 'void main() {\nvRefUv = refTextureMatrix * vec4( position, 1.0 );')
                // if (!shader.fragmentShader.includes('#include <map_fragment>')) console.error('Shader modified before this.')
                const ls = '#glMarker beforeModulation'
                shader.fragmentShader = shaderReplaceString(shader.fragmentShader, ls, `
                    if(roughnessFactor < 0.95) {
                        float d = 0.;//textureProj(tRefDepth, vRefUv).r;
                        // d = min(2., max(0., (d-0.06) * ((7./3.-ior)) * sceneBoundingRadius));
                        vec4 refBaseColor = getReflectionColor(material.roughness, material.roughness * d);
                        // refBaseColor.rgb = vec3(refBaseColor.a);
                        // refBaseColor.a *= 1.0 - clamp(material.roughness * .3, 0., 1.);
                        `
                    + (this.reflectorModePhysical ? `
                        #if !defined(SSR_ENABLED) || SSR_ENABLED < 1 
                        vec3 specularColor = EnvironmentBRDF(geometryNormal, geometryViewDir, material.specularColor.rgb, material.specularF90, material.roughness);
                        #endif
                        reflectedLight.indirectSpecular = mix(vec3(reflectedLight.indirectSpecular), saturate(specularColor.rgb * refBaseColor.rgb), refBaseColor.a);
                        ` : `
                        reflectedLight.indirectSpecular = saturate(diffuseColor.rgb * refBaseColor.rgb);
                        diffuseColor.a *= refBaseColor.a;
                        `) +
                    '}\n' + ls) // todo: this is a hacky way to handle transparent object, use a MeshBasicMaterial instead            // ;(shader as any).vertexUvs = true
            }

        },
        parsVertexSnippet: ()=>!this.enabled ? '' : `
		uniform mat4 refTextureMatrix;
		varying vec4 vRefUv;
`,
        parsFragmentSnippet: ()=>this.enabled ? poissonDiskSamples + '\n' + randomHelpers + '\n' + reflectorSample : '',
        computeCacheKey: (material: IMaterial)=>{
            return this.enabled + ' ' + material.materialObject.transparent + ' ' + this.reflectorModePhysical + ' '
        },
        onObjectRender: (object, {materialObject}) => {
            if (materialObject.userData.__lastTransparent !== materialObject.transparent) {
                materialObject.needsUpdate = true
                materialObject.userData.__lastTransparent = materialObject.transparent
            }
        },
        isCompatible: material => {
            return (material as MeshStandardMaterial2).isMeshStandardMaterial2
        },
    }
}

export {DlssReflector2 as Reflector2}

(DlssReflector2.prototype as any).isReflector = true
