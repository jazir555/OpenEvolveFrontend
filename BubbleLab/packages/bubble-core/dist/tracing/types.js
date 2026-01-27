/**
 * Type definitions for OpenTelemetry distributed tracing
 */
/**
 * Exporter types
 */
export var ExporterType;
(function (ExporterType) {
    /** OpenTelemetry Collector (production) */
    ExporterType["COLLECTOR"] = "collector";
    /** Console output (debugging) */
    ExporterType["CONSOLE"] = "console";
    /** OTLP protocol */
    ExporterType["OTLP"] = "otlp";
})(ExporterType || (ExporterType = {}));
//# sourceMappingURL=types.js.map