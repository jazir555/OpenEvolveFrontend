declare module 'puppeteer-core' {
  export type Browser = any;
  export type Page = any;
  const _default: any;
  export default _default;
}

declare module 'winston' {
  export type Logger = any;
  export type transport = any;
  export const createLogger: any;
  export const format: any;
  export const transports: any;
  export const addColors: any;
  export const config: any;
  const _default: any;
  export default _default;
}

declare module 'winston-elasticsearch' {
  export type ElasticsearchTransport = any;
  export const ElasticsearchTransport: any;
}

declare module 'express' {
  export type Request = any;
  export type Response = any;
  export type NextFunction = any;
  export type Application = any;
  export type Router = any;
  const _default: any;
  export default _default;
}

declare module 'sharp' {
  export type Sharp = any;
  export const _default: any;
  export default _default;
}

declare module 'pdfkit' {
  export type PDFDocument = any;
  export const PDFDocument: any;
  export default PDFDocument;
}

declare module 'xml2js' {
  export type Parser = any;
  export const Parser: any;
  export const parseString: any;
  const _default: any;
  export default _default;
}

declare module 'prom-client' {
  export type Registry = any;
  export const Registry: any;
  export type Counter = any;
  export const Counter: any;
  export type Histogram = any;
  export const Histogram: any;
  export type Gauge = any;
  export const Gauge: any;
  export const collectDefaultMetrics: any;
  const _default: any;
  export default _default;
}

declare module '@opentelemetry/exporter-trace-otlp-grpc' {
  export type OTLPTraceExporter = any;
  export const OTLPTraceExporter: any;
}

declare module '@opentelemetry/exporter-trace-otlp-http' {
  export type OTLPTraceExporter = any;
  export const OTLPTraceExporter: any;
}

declare module '@opentelemetry/sdk-trace-node' {
  export type NodeTracerProvider = any;
  export const NodeTracerProvider: any;
}

declare module '@opentelemetry/context-async-hooks' {
  export type AsyncHookContextManager = any;
  export const AsyncHookContextManager: any;
}
