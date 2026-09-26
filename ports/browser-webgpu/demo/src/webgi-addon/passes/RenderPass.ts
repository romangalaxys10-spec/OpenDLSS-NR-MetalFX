import {Color} from 'three';
import {Pass} from './Pass';

export class RenderPass extends Pass {
  scene: any;
  camera: any;
  overrideMaterial: any;
  clearColor: any;
  clearAlpha: any;
  clearDepth = false;
  private readonly oldClearColor = new Color();

  constructor(scene?: any, camera?: any, overrideMaterial: any = null,
              clearColor: any = null, clearAlpha: any = null) {
    super();
    this.scene = scene;
    this.camera = camera;
    this.overrideMaterial = overrideMaterial;
    this.clearColor = clearColor;
    this.clearAlpha = clearAlpha;
    this.clear = true;
    this.needsSwap = false;
  }

  render(renderer: any, _writeBuffer: any, readBuffer: any, _deltaTime?: number,
         _maskActive?: boolean, depthRenderBuffer?: any) {
    if (!this.scene || !this.camera) return;
    const oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;
    let oldClearAlpha: number | undefined;
    let oldOverrideMaterial: any;
    if (this.overrideMaterial !== null) {
      oldOverrideMaterial = this.scene.overrideMaterial;
      this.scene.overrideMaterial = this.overrideMaterial;
    }
    if (this.clearColor !== null) {
      renderer.getClearColor(this.oldClearColor);
      renderer.setClearColor(this.clearColor, renderer.getClearAlpha());
    }
    if (this.clearAlpha !== null) {
      oldClearAlpha = renderer.getClearAlpha();
      renderer.setClearAlpha(this.clearAlpha);
    }
    if (this.clearDepth) renderer.clearDepth();
    renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
    if (depthRenderBuffer) {
      const gl = renderer.getContext();
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRenderBuffer);
    }
    if (this.clear) renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    renderer.render(this.scene, this.camera);
    if (depthRenderBuffer) {
      const gl = renderer.getContext();
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, null);
    }
    if (this.clearColor !== null) renderer.setClearColor(this.oldClearColor);
    if (this.clearAlpha !== null) renderer.setClearAlpha(oldClearAlpha);
    if (this.overrideMaterial !== null) this.scene.overrideMaterial = oldOverrideMaterial;
    renderer.autoClear = oldAutoClear;
  }
}
