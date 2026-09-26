import {InstancedMesh, Material, Mesh, Scene, ShaderMaterial, UnsignedIntType, WebGLRenderTarget} from 'three'
import {ViewerApp} from 'webgi/viewer/ViewerApp'
import {GBufferPlugin} from 'webgi/plugins/GBufferPlugin'
import {VelocityBufferPlugin} from 'webgi/plugins/VelocityBufferPlugin'
import velocityFragment from 'webgi/plugins/shaders/ssVelocityFrag.glsl'

/** Share opaque scatter rasterization between depth/normal and motion outputs. */
export function createOpaquePassFusion(viewer: ViewerApp) {
    const renderer = viewer.renderer.rendererObject, gl = renderer.getContext() as WebGL2RenderingContext
    const gb = viewer.getPlugin(GBufferPlugin), velocity = viewer.getPlugin(VelocityBufferPlugin)
    const gp = gb?.pass?.passObject, vp = velocity?.pass?.passObject
    const gt = gb?.getTarget(), vt = velocity?.getVelocityBuffer() as WebGLRenderTarget | undefined
    if (!renderer.capabilities.isWebGL2 || !gp || !vp || !gt || !vt || gt.textures.length !== 2 ||
        gt.samples || vt.samples || gt.depthTexture && gt.depthTexture.type !== UnsignedIntType ||
        renderer.capabilities.logarithmicDepthBuffer) return undefined
    // Alpha extrema checked in the packaged source images, including all pixels.
    // This path is scoped to these immutable demo assets, never arbitrary models.
    const certified = new Set(['Foliage001_2K-JPG_Color', 'Grass procedural .003', 'Brown grass.003',
        'Bark.001', 'Stem', 'Flower.001', 'Flower.002'])
    const eligible = new Set<Mesh>()
    viewer.scene.modelRoot.traverseVisible(object => {
        const mesh = object as InstancedMesh, m = mesh.material as any
        if (!mesh.isInstancedMesh || Array.isArray(m) || !certified.has(m.name) || m.transparent ||
            m.alphaHash || m.alphaMap || m.alphaTest || m.transmission || m.displacementMap || m.polygonOffset ||
            m.clippingPlanes?.length || m.depthWrite === false || m.userData.pluginsDisabled ||
            m.userData.renderToDepth === false || mesh.morphTargetInfluences?.length) return
        eligible.add(mesh)
    })
    if (!eligible.size) return undefined
    const gm = gp.overrideMaterial as ShaderMaterial & {extraUniformsToUpload: Record<string, any>}
    const vm = vp.overrideMaterial as ShaderMaterial & {extraUniformsToUpload: Record<string, any>}
    const original = {gRender: gp.render, vRender: vp.render, compile: gm.onBeforeCompile,
        key: gm.customProgramCacheKey, before: gm.onBeforeRender}
    const previousUniform = gm.extraUniformsToUpload.modelMatrixPrevious
    const fused = gt.clone()
    fused.textures.push(vt.texture.clone())
    fused.textures[2].name = 'opaqueScatterMotion'
    let enabled = false, fusedFrame = false, drawingOpaque = false
    const motionBody = velocityFragment.slice(0, velocityFragment.indexOf('void main()'))
    const motionWrite = velocity.rawOutput ? 'gScatterMotion = vec4(clamp(computeScreenSpaceVelocity2(),-1.0,1.0),0.,1.);' : `
        vec2 scatterVelocity=clamp(computeScreenSpaceVelocity2(),-1.0,1.0);
        scatterVelocity=sign(scatterVelocity)*pow(abs(scatterVelocity),vec2(1./4.));
        gScatterMotion=vec4(scatterVelocity*.5+.5,1.,1.);`
    gm.onBeforeCompile = function(this: Material, shader, r) {
        original.compile.call(this, shader, r)
        if (!enabled) return
        shader.uniforms.currentProjectionViewMatrix = vm.uniforms.currentProjectionViewMatrix
        shader.uniforms.lastProjectionViewMatrix = vm.uniforms.lastProjectionViewMatrix
        shader.vertexShader = `varying vec3 vWorldPosition; varying vec3 vWorldPositionPrevious;
            uniform mat4 modelMatrixPrevious;\n` + shader.vertexShader.replace('#include <project_vertex>', `
            vec4 scatterPosition = vec4(transformed,1.);
            #ifdef USE_INSTANCING
            scatterPosition = instanceMatrix * scatterPosition;
            #endif
            vWorldPosition = (modelMatrix * scatterPosition).xyz;
            vWorldPositionPrevious = (modelMatrixPrevious * scatterPosition).xyz;
            #include <project_vertex>`)
        shader.fragmentShader = 'layout(location=2) out vec4 gScatterMotion;\n' + motionBody +
            shader.fragmentShader.replace(/}\s*$/, motionWrite + '\n}')
    }
    gm.customProgramCacheKey = function(this: Material) { return original.key.call(this) + '|scatter-motion:' + enabled }
    gm.onBeforeRender = function(r, s, c, geometry, object, group) {
        if (drawingOpaque) {
            vm.onBeforeRender(r, s, c, geometry, object, group)
            gm.extraUniformsToUpload.modelMatrixPrevious = vm.extraUniformsToUpload.modelMatrixPrevious
        }
        original.before.call(this, r, s, c, geometry, object, group)
    }
    const select = (opaque: boolean, callback: () => void) => {
        const hidden: Mesh[] = []
        viewer.scene.modelRoot.traverseVisible(object => {
            const mesh = object as Mesh
            if (mesh.isMesh && eligible.has(mesh) !== opaque) { hidden.push(mesh); mesh.visible = false }
        })
        try { callback() } finally { for (const mesh of hidden) mesh.visible = true }
    }
    const copy = (source: WebGLFramebuffer, destination: WebGLRenderTarget, attachment: number, output: number, depth: boolean) => {
        renderer.setRenderTarget(destination)
        const oldRead = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING), oldDraw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING)
        const buffers = destination.textures.map((_, i) => gl.COLOR_ATTACHMENT0 + i)
        try {
            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, source)
            gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachment)
            gl.drawBuffers(buffers.map((b, i) => i === output ? b : gl.NONE))
            gl.blitFramebuffer(0, 0, fused.width, fused.height, 0, 0, destination.width, destination.height,
                gl.COLOR_BUFFER_BIT | (depth ? gl.DEPTH_BUFFER_BIT : 0), gl.NEAREST)
            gl.readBuffer(gl.COLOR_ATTACHMENT0)
            gl.drawBuffers(buffers)
        } finally {
            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, oldRead); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, oldDraw)
        }
    }
    vp.render = function(...args) {
        fusedFrame = false
        if (!enabled || !gb.enabled || gt.width !== vt.width || gt.height !== vt.height || gt.samples || vt.samples) {
            return original.vRender.apply(this, args)
        }
        fused.setSize(gt.width, gt.height)
        gb.pass?.update?.()
        const render = renderer.render
        renderer.render = function(scene, camera) {
            if (drawingOpaque) {
                const zero = velocity.rawOutput ? 0 : .5
                gl.clearBufferfv(gl.COLOR, 2, new Float32Array([zero, zero, 1, 1]))
                return render.call(this, scene, camera)
            }
            if ((scene as Scene).overrideMaterial !== vm) return render.call(this, scene, camera)
            const old = {target: gp.target, scene: gp.scene, camera: gp.camera, clear: gp.clear}
            try {
                drawingOpaque = true
                gp.target = fused; gp.scene = scene as Scene; gp.camera = camera; gp.clear = true
                select(true, () => original.gRender.call(gp, renderer, null))
                renderer.setRenderTarget(fused)
                const framebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING)
                copy(framebuffer, vt, 2, 0, true)
                copy(framebuffer, gt, 0, 0, true)
                copy(framebuffer, gt, 1, 1, false)
                fusedFrame = true
            } finally {
                drawingOpaque = false
                gp.target = old.target; gp.scene = old.scene; gp.camera = old.camera; gp.clear = old.clear
                renderer.setRenderTarget(vt)
            }
            select(false, () => render.call(this, scene, camera))
        }
        try { return original.vRender.apply(this, args) } finally { renderer.render = render }
    }
    gp.render = function(...args) {
        if (!fusedFrame) return original.gRender.apply(this, args)
        const clear = this.clear
        try { this.clear = false; select(false, () => original.gRender.apply(this, args)) }
        finally { this.clear = clear; fusedFrame = false }
    }
    const setEnabled = (value: boolean) => { enabled = value; gm.needsUpdate = true; viewer.setDirty() }
    setEnabled(true)
    return {setEnabled, get enabled() { return enabled }, eligibleMeshes: eligible.size, dispose: () => {
        gp.render = original.gRender; vp.render = original.vRender
        gm.onBeforeCompile = original.compile; gm.customProgramCacheKey = original.key; gm.onBeforeRender = original.before
        if (previousUniform) gm.extraUniformsToUpload.modelMatrixPrevious = previousUniform
        else delete gm.extraUniformsToUpload.modelMatrixPrevious
        gm.needsUpdate = true; fused.dispose()
    }}
}
