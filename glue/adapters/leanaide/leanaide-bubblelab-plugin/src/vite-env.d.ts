/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_LEANAIDE_SERVER_URL?: string;
  readonly VITE_RAGBITS_SERVER_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
