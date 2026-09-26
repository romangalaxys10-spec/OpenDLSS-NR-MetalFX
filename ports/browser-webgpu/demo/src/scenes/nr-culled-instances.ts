import {Box3, BufferGeometry, Camera, DynamicDrawUsage, Frustum, InstancedMesh,
    Material, Matrix4, Sphere, Vector3, WebGLRenderer} from 'three'

interface Visibility {
    matrix: Matrix4
    matrices: Float32Array
    count: number
    revision: number
    maskRevision: number
}
interface OcclusionMask {bytes: Uint8Array; offset: number; revision: number; matrix: Matrix4; width: number; height: number}
export const scatterBounds = new WeakMap<InstancedMesh, {centers: Float64Array;
    masks: WeakMap<Camera, OcclusionMask>}>()

/** Static scatter: keep every instance, submit only those intersecting this pass's camera. */
export function createCulledInstances(geometry: BufferGeometry, material: Material | Material[],
    original: Float32Array, groups: number[][], renderer: WebGLRenderer) {
    const total = original.length / 16
    const mesh = new InstancedMesh(geometry, material, total)
    const matrices = mesh.instanceMatrix.array as Float32Array
    mesh.instanceMatrix.setUsage(DynamicDrawUsage)
    geometry.computeBoundingSphere()
    const source = geometry.boundingSphere!
    const centers = new Float64Array(total * 4)
    const clusters: {start: number; end: number; box: Box3; sphere: Sphere}[] = []
    const box = new Box3(), fullBox = new Box3(), point = new Vector3()
    let cursor = 0
    for (const group of groups) {
        const start = cursor
        box.makeEmpty()
        for (const index of group) {
            const j = index * 16, d = cursor * 16, k = cursor * 4
            for (let c = 0; c < 16; c++) matrices[d + c] = original[j + c]
            const x = original[j] * source.center.x + original[j + 4] * source.center.y + original[j + 8] * source.center.z + original[j + 12]
            const y = original[j + 1] * source.center.x + original[j + 5] * source.center.y + original[j + 9] * source.center.z + original[j + 13]
            const z = original[j + 2] * source.center.x + original[j + 6] * source.center.y + original[j + 10] * source.center.z + original[j + 14]
            const dot = (a: number, b: number) => original[j + a] * original[j + b] +
                original[j + a + 1] * original[j + b + 1] + original[j + a + 2] * original[j + b + 2]
            const xy = Math.abs(dot(0, 4)), xz = Math.abs(dot(0, 8)), yz = Math.abs(dot(4, 8))
            // Bound the largest singular value, including sheared transforms.
            const scale = Math.sqrt(Math.max(dot(0, 0) + xy + xz, dot(4, 4) + xy + yz, dot(8, 8) + xz + yz))
            const r = source.radius * scale + 1e-5
            centers[k] = x; centers[k + 1] = y; centers[k + 2] = z; centers[k + 3] = r
            box.expandByPoint(point.set(x - r, y - r, z - r))
            box.expandByPoint(point.set(x + r, y + r, z + r))
            cursor++
        }
        fullBox.union(box)
        clusters.push({start, end: cursor, box: box.clone(), sphere: box.getBoundingSphere(new Sphere())})
    }
    if (cursor !== total) throw new Error('Incomplete scatter partition')
    mesh.boundingBox = fullBox
    const masks = new WeakMap<Camera, OcclusionMask>()
    scatterBounds.set(mesh, {centers, masks})
    mesh.boundingSphere = fullBox.getBoundingSphere(new Sphere())
    const frustum = new Frustum(), projection = new Matrix4()
    const cameras = new WeakMap<Camera, Visibility>()
    const gl = renderer.getContext()
    let buffer: WebGLBuffer | null = null
    let uploaded: Visibility | null = null, uploadedRevision = -1
    // Three's supported upload callback runs while this attribute's buffer is
    // bound. Keep its allocation/VAO ownership; replace only the draw's prefix.
    mesh.instanceMatrix.onUpload(() => {
        buffer = gl.getParameter(gl.ARRAY_BUFFER_BINDING)
        uploaded = null
    })
    const prepare = (camera: Camera) => {
        if (!buffer) return
        if (!mesh.frustumCulled) {
            if (uploaded) {
                const previous = gl.getParameter(gl.ARRAY_BUFFER_BINDING)
                gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
                gl.bufferData(gl.ARRAY_BUFFER, matrices, gl.DYNAMIC_DRAW)
                gl.bindBuffer(gl.ARRAY_BUFFER, previous)
                uploaded = null
            }
            mesh.count = total
            return
        }
        projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
        const candidateMask = masks.get(camera)
        // A pass may adjust its projection after preRender. Never apply a mask
        // measured for a different view (including a different jitter sample).
        const target = renderer.getRenderTarget()
        const mask = candidateMask?.matrix.equals(projection) && target && !target.samples &&
            candidateMask.width === target.width && candidateMask.height === target.height ? candidateMask : undefined
        projection.multiply(mesh.matrixWorld)
        let visible = cameras.get(camera)
        if (!visible) {
            visible = {matrix: new Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                matrices: new Float32Array(matrices.length), count: 0, revision: 0, maskRevision: -1}
            cameras.set(camera, visible)
        }
        const maskRevision = mask?.revision ?? -1
        if (!visible.matrix.equals(projection) || visible.maskRevision !== maskRevision) {
            visible.matrix.copy(projection)
            visible.maskRevision = maskRevision
            frustum.setFromProjectionMatrix(projection)
            const planes = frustum.planes
            let count = 0
            for (const cluster of clusters) {
                if (!frustum.intersectsSphere(cluster.sphere)) continue
                const c = cluster.sphere.center, r = cluster.sphere.radius
                const inside = planes.every(p => p.distanceToPoint(c) >= r)
                if (inside && !mask) {
                    visible.matrices.set(matrices.subarray(cluster.start * 16, cluster.end * 16), count * 16)
                    count += cluster.end - cluster.start
                    continue
                }
                for (let i = cluster.start; i < cluster.end; i++) {
                    if (mask && !mask.bytes[mask.offset + i]) continue
                    const k = i * 4, x = centers[k], y = centers[k + 1], z = centers[k + 2], radius = centers[k + 3]
                    let keep = true
                    for (const p of planes) if (p.normal.x * x + p.normal.y * y + p.normal.z * z + p.constant < -radius) {
                        keep = false
                        break
                    }
                    if (!keep) continue
                    const a = i * 16, b = count++ * 16
                    for (let j = 0; j < 16; j++) visible.matrices[b + j] = matrices[a + j]
                }
            }
            visible.count = count
            visible.revision++
        }
        mesh.count = visible.count
        if (uploaded !== visible || uploadedRevision !== visible.revision) {
            const previous = gl.getParameter(gl.ARRAY_BUFFER_BINDING)
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
            // Orphan the previous store: a shadow pass can still be reading it.
            // Rewriting that in-use store forces a long CPU/GPU synchronization.
            gl.bufferData(gl.ARRAY_BUFFER, Math.max(64, visible.count * 64), gl.DYNAMIC_DRAW)
            if (visible.count) gl.bufferSubData(gl.ARRAY_BUFFER, 0, visible.matrices, 0, visible.count * 16)
            gl.bindBuffer(gl.ARRAY_BUFFER, previous)
            uploaded = visible
            uploadedRevision = visible.revision
        }
    }
    mesh.onBeforeRender = (_renderer, _scene, camera) => prepare(camera)
    mesh.onBeforeShadow = (_renderer, _object, _camera, shadowCamera) => prepare(shadowCamera)
    // Picking and bounds continue to see the complete CPU-side scatter.
    mesh.onAfterRender = mesh.onAfterShadow = () => { mesh.count = total }
    return mesh
}
