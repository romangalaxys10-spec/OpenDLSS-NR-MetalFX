/** Rebuild the original triangular tubes from compact, quantized ring records. */
export interface GroomRings {
    encoding: 'rings-v2'
    rings: number
    strands: number
    vertices: number
    indices: number
    centerStep: number
    radiusStep: number
    frameScale: number
}

export function decodeGroomRings(buffer: ArrayBuffer, part: GroomRings) {
    const {rings, strands, vertices, indices, centerStep, radiusStep, frameScale} = part
    if (![rings, strands, vertices, indices].every(n => Number.isSafeInteger(n) && n > 0) ||
        vertices !== rings * 3 || indices !== (rings - strands) * 18 ||
        ![centerStep, radiusStep, frameScale].every(n => Number.isFinite(n) && n > 0) ||
        buffer.byteLength !== strands * 2 + rings * 22) throw new Error('Invalid compact groom metadata')
    const bytes = new Uint8Array(buffer)
    const centerOffset = strands * 2
    const frameOffset = centerOffset + rings * 12
    const omittedOffset = frameOffset + rings * 6
    const radiusOffset = omittedOffset + rings
    const opacityOffset = radiusOffset + rings * 2
    const position = new Float32Array(vertices * 3)
    const normal = new Float32Array(vertices * 3)
    const color = new Uint8Array(vertices * 4)
    const index = new Uint32Array(indices)
    const read16 = (base: number, i: number, count: number) => bytes[base + i] | bytes[base + count + i] << 8
    const read32 = (base: number, i: number) => bytes[base + i] | bytes[base + rings + i] << 8 |
        bytes[base + rings * 2 + i] << 16 | bytes[base + rings * 3 + i] << 24
    let cx = 0, cy = 0, cz = 0
    const q = new Float64Array(4)
    for (let i = 0; i < rings; i++) {
        cx += read32(centerOffset, i)
        cy += read32(centerOffset + rings * 4, i)
        cz += read32(centerOffset + rings * 8, i)
        const omitted = bytes[omittedOffset + i]
        if (omitted > 3) throw new Error('Invalid groom rotation')
        let component = 0, square = 0
        for (let k = 0; k < 4; k++) {
            if (k === omitted) continue
            const packed = read16(frameOffset + component++ * rings * 2, i, rings)
            q[k] = (packed << 16 >> 16) / frameScale
            square += q[k] * q[k]
        }
        q[omitted] = Math.sqrt(Math.max(0, 1 - square))
        const [x, y, z, w] = q
        const ux = 1 - 2 * (y * y + z * z), uy = 2 * (x * y + z * w), uz = 2 * (x * z - y * w)
        const vx = 2 * (x * y - z * w), vy = 1 - 2 * (x * x + z * z), vz = 2 * (y * z + x * w)
        const radius = read16(radiusOffset, i, rings) * radiusStep
        const opacity = bytes[opacityOffset + i]
        for (let side = 0; side < 3; side++) {
            const u = side === 0 ? 1 : -.5, v = side === 0 ? 0 : side === 1 ? .8660254 : -.8660254
            const nx = u * ux + v * vx, ny = u * uy + v * vy, nz = u * uz + v * vz
            const vertex = i * 3 + side, j = vertex * 3
            position[j] = cx * centerStep + nx * radius
            position[j + 1] = cy * centerStep + ny * radius
            position[j + 2] = cz * centerStep + nz * radius
            normal[j] = nx; normal[j + 1] = ny; normal[j + 2] = nz
            color[vertex * 4] = color[vertex * 4 + 1] = color[vertex * 4 + 2] = 255
            color[vertex * 4 + 3] = opacity
        }
    }
    let ring = 0, cursor = 0
    for (let strand = 0; strand < strands; strand++) {
        const length = read16(0, strand, strands)
        if (length < 2 || ring + length > rings) throw new Error('Invalid groom strand length')
        for (let j = 0; j < length - 1; j++) {
            const base = (ring + j) * 3
            for (let side = 0; side < 3; side++) {
                const a = base + side, b = base + (side + 1) % 3
                index[cursor++] = a; index[cursor++] = b; index[cursor++] = a + 3
                index[cursor++] = b; index[cursor++] = b + 3; index[cursor++] = a + 3
            }
        }
        ring += length
    }
    if (ring !== rings || cursor !== indices) throw new Error('Incomplete compact groom')
    return {position, normal, color, index}
}
