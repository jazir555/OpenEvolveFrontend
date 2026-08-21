/**
 * Ambient module declarations for optional, dynamically-loaded plugins.
 *
 * These plugins (@bubblelabs-ragbits-plugin, @datapizza-bubblelab-plugin) are
 * loaded at runtime via dynamic `import()` in plugin-integration.ts and are not
 * hard dependencies of this package. They resolve at runtime when the host
 * application provides them; here we declare minimal shapes so the integration
 * library type-checks without coupling the build to those packages.
 */

declare module '@bubblelabs-ragbits-plugin' {
  export function createPlugin(config?: any): any;
  const _default: any;
  export default _default;
}

declare module '@datapizza-bubblelab-plugin' {
  export function createPlugin(config?: any): any;
  const _default: any;
  export default _default;
}
