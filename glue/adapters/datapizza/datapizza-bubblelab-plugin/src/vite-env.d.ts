/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DATAPIZZA_SERVER_URL?: string;
  readonly VITE_DATAPIZZA_USE_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
