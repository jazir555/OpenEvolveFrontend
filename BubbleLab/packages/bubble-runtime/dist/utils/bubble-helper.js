/**
 * Build a lookup map from className to bubble metadata
 */
export function buildClassNameLookup(factory) {
    const lookup = new Map();
    const all = factory.getAll();
    for (const ctor of all) {
        const className = ctor.name;
        const bubbleName = ctor.bubbleName ?? className;
        const nodeType = ctor.type ?? 'unknown';
        lookup.set(className, {
            bubbleName: bubbleName,
            className,
            nodeType,
        });
    }
    return lookup;
}
//# sourceMappingURL=bubble-helper.js.map