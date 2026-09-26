import {BufferAttribute, Material, Mesh, Object3D} from 'three'
import {mergeGeometries} from 'three/examples/jsm/utils/BufferGeometryUtils'
import {ViewerApp} from 'webgi/viewer/ViewerApp'

/** Concatenate identical draw states without transforming or simplifying vertices. */
export function createStaticDrawGroups(viewer: ViewerApp) {
    const groups = new Map<string, Mesh[]>()
    // Viewer lifecycle helpers are bound to each imported object. They do not
    // describe draw state and must not be copied onto a different mesh.
    const lifecycle = new Set(['name', '__iModelSetup', '__meshSetup', 'setDirty', 'dispose', '__objectUpdater',
        'setMaterial', 'setGeometry', '__autoParentDispatchEvents', 'parentRoot', '__materialUpdater', '__textureUpdater'])
    const drawData = (mesh: Mesh) => Object.entries(mesh.userData).filter(([key]) => !lifecycle.has(key))
    const references = new WeakMap<object, number>()
    let nextReference = 0
    const identity = (value: any): any => {
        if ((typeof value !== 'object' || !value) && typeof value !== 'function') return value
        if (!references.has(value)) references.set(value, nextReference++)
        return {reference: references.get(value)}
    }
    viewer.scene.modelRoot.traverseVisible(object => {
        const mesh = object as Mesh, m = mesh.material as any, g = mesh.geometry
        if (!mesh.isMesh || (mesh as any).isInstancedMesh || (mesh as any).isSkinnedMesh || mesh.children.length ||
            Array.isArray(m) || m.transparent || m.transmission || mesh.morphTargetInfluences?.length ||
            mesh.customDepthMaterial || mesh.customDistanceMaterial || !mesh.parent ||
            g.drawRange.start !== 0 || g.drawRange.count < (g.index?.count ?? g.attributes.position.count)) return
        const attributes = Object.entries(g.attributes).map(([name, a]) =>
            [name, a.itemSize, a.normalized, a.array.constructor.name, (a as BufferAttribute).gpuType]).sort()
        const key = JSON.stringify([mesh.parent.uuid, mesh.matrix.elements, m.uuid, mesh.castShadow, mesh.receiveShadow,
            mesh.layers.mask, mesh.renderOrder, mesh.frustumCulled,
            drawData(mesh).map(([key, value]) => [key, identity(value)]).sort(), !!g.index, attributes])
        const row = groups.get(key)
        if (row) row.push(mesh)
        else groups.set(key, [mesh])
    })
    const batches: {sources: Mesh[]; mesh: Mesh; parent: Object3D}[] = []
    for (const sources of groups.values()) {
        if (sources.length < 2) continue
        const geometry = mergeGeometries(sources.map(m => m.geometry))
        if (!geometry) continue
        const first = sources[0], mesh = new Mesh(geometry, first.material)
        mesh.name = (first.material as Material).name + ' static draw group'
        mesh.matrix.copy(first.matrix); mesh.matrixAutoUpdate = false
        mesh.castShadow = first.castShadow; mesh.receiveShadow = first.receiveShadow
        mesh.layers.mask = first.layers.mask; mesh.renderOrder = first.renderOrder
        mesh.frustumCulled = first.frustumCulled; mesh.userData = Object.fromEntries(drawData(first))
        geometry.computeBoundingBox(); geometry.computeBoundingSphere()
        batches.push({sources, mesh, parent: first.parent!})
    }
    let enabled = false
    const setEnabled = (value: boolean) => {
        if (value === enabled) return
        enabled = value
        for (const batch of batches) {
            if (value) {
                for (const source of batch.sources) source.removeFromParent()
                batch.parent.add(batch.mesh)
            } else {
                batch.mesh.removeFromParent()
                for (const source of batch.sources) batch.parent.add(source)
            }
        }
        viewer.scene.modelRoot.updateWorldMatrix(true, true)
        viewer.setDirty()
    }
    setEnabled(true)
    return {setEnabled, get enabled() { return enabled },
        sourceDraws: batches.reduce((n, b) => n + b.sources.length, 0), mergedDraws: batches.length,
        dispose: () => { setEnabled(false); for (const b of batches) b.mesh.geometry.dispose() }}
}
