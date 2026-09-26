import {BufferGeometry, Float32BufferAttribute, Mesh, OrthographicCamera} from 'three';

export class Pass {
  readonly isPass = true;
  enabled = true;
  needsSwap = true;
  clear = false;
  renderToScreen = false;
  setSize(_width?: number, _height?: number) {}
  render(..._args: any[]) { throw new Error('Pass.render must be implemented'); }
  dispose() {}
}

const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
const geometry = new BufferGeometry();
geometry.setAttribute('position', new Float32BufferAttribute([-1, 3, 0, -1, -1, 0, 3, -1, 0], 3));
geometry.setAttribute('uv', new Float32BufferAttribute([0, 2, 0, 0, 2, 0], 2));

export class FullScreenQuad {
  private readonly mesh: any;
  constructor(material: any) { this.mesh = new Mesh(geometry, material); }
  dispose() { this.mesh.geometry.dispose(); }
  render(renderer: any) { renderer.render(this.mesh, camera); }
  get material() { return this.mesh.material; }
  set material(value) { this.mesh.material = value; }
}
