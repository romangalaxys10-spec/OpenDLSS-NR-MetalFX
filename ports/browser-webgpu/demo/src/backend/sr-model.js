// Super resolution is not part of this port. The runtime imports these names but never reaches them unless
// it is started in a super-resolution mode, which this demo does not offer.
export class DlssSrWebGpuModel {
  static async create() { throw new Error('super resolution is not part of this port'); }
}
