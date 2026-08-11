# OpenTelemetry Package Installation

To complete the distributed tracing implementation, add the following packages to `BubbleLab/packages/bubble-core/package.json`.

## Required Dependencies

Add to the `dependencies` section:

```json
{
  "dependencies": {
    "@opentelemetry/api": "^1.8.0",
    "@opentelemetry/sdk-trace-node": "^1.22.0",
    "@opentelemetry/exporter-trace-jaeger": "^1.22.0",
    "@opentelemetry/exporter-trace-otlp-grpc": "^1.22.0",
    "@opentelemetry/exporter-trace-otlp-http": "^1.22.0",
    "@opentelemetry/context-async-hooks": "^1.22.0",
    "@opentelemetry/resources": "^1.22.0",
    "@opentelemetry/semantic-conventions": "^1.22.0"
  }
}
```

## Optional Dev Dependencies

For automatic instrumentation (optional but recommended):

```json
{
  "devDependencies": {
    "@opentelemetry/auto-instrumentations": "^0.45.1"
  }
}
```

## Installation Commands

Run these commands from the `BubbleLab/packages/bubble-core` directory:

```bash
# Core OpenTelemetry packages
pnpm add @opentelemetry/api@^1.8.0
pnpm add @opentelemetry/sdk-trace-node@^1.22.0
pnpm add @opentelemetry/exporter-trace-jaeger@^1.22.0
pnpm add @opentelemetry/exporter-trace-otlp-grpc@^1.22.0
pnpm add @opentelemetry/exporter-trace-otlp-http@^1.22.0
pnpm add @opentelemetry/context-async-hooks@^1.22.0
pnpm add @opentelemetry/resources@^1.22.0
pnpm add @opentelemetry/semantic-conventions@^1.22.0

# Optional: Auto-instrumentations
pnpm add -D @opentelemetry/auto-instrumentations@^0.45.1
```

## Complete Updated package.json

Here's how the dependencies section should look after updates:

```json
{
  "name": "@bubblelab/bubble-core",
  "version": "0.1.11",
  "type": "module",
  "license": "Apache-2.0",
  "main": "./dist/index.js",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "files": [
    "dist",
    "LICENSE.txt"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/bubblelabai/BubbleLab.git",
    "directory": "packages/bubble-core"
  },
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsc && tsx scripts/bubble-bundler.ts && tsx scripts/bubble-metadata-bundler.ts",
    "build:types": "tsc",
    "build:bundle": "tsx scripts/bubble-bundler.ts",
    "build:metadata": "tsx scripts/bubble-metadata-bundler.ts",
    "dev": "tsc --watch",
    "typecheck": "tsc --noEmit",
    "test": "vitest run --exclude='**/*.integration.{test,spec}.ts'",
    "test:coverage": "vitest run --coverage --exclude='**/*.integration.{test,spec}.ts'",
    "test:integration": "vitest run --run 'src/**/*.integration.{test,spec}.ts'",
    "test:all": "vitest run",
    "test:all:coverage": "vitest run --coverage",
    "test:watch": "vitest",
    "lint": "eslint . --ext .ts,.tsx",
    "prepublishOnly": "pnpm run build"
  },
  "dependencies": {
    "@aws-sdk/client-s3": "^3.873.0",
    "@aws-sdk/s3-request-presigner": "^3.873.0",
    "@bubblelab/shared-schemas": "workspace:*",
    "@google/generative-ai": "^0.24.1",
    "@langchain/anthropic": "^0.3.32",
    "@langchain/community": "^0.3.53",
    "@langchain/core": "^0.3.66",
    "@langchain/google-genai": "^1.0.3",
    "@langchain/langgraph": "^0.3.10",
    "@langchain/openai": "^0.6.2",
    "@mendable/firecrawl-js": "^4.5.0",
    "@opentelemetry/api": "^1.8.0",
    "@opentelemetry/context-async-hooks": "^1.22.0",
    "@opentelemetry/exporter-trace-jaeger": "^1.22.0",
    "@opentelemetry/exporter-trace-otlp-grpc": "^1.22.0",
    "@opentelemetry/exporter-trace-otlp-http": "^1.22.0",
    "@opentelemetry/resources": "^1.22.0",
    "@opentelemetry/sdk-trace-node": "^1.22.0",
    "@opentelemetry/semantic-conventions": "^1.22.0",
    "@types/pg": "^8.15.4",
    "@typescript-eslint/typescript-estree": "8.46.0",
    "chart.js": "^4.5.0",
    "chartjs-node-canvas": "^5.0.0",
    "mathjs": "^14.0.0",
    "pg": "^8.16.3",
    "prom-client": "^15.1.3",
    "resend": "^4.8.0",
    "winston": "^3.17.0",
    "winston-elasticsearch": "^0.17.5",
    "zod": "^3.24.1",
    "zod-to-json-schema": "^3.24.6"
  },
  "devDependencies": {
    "@opentelemetry/auto-instrumentations": "^0.45.1",
    "@types/node": "^20.12.12",
    "@vitest/ui": "^3.2.4",
    "tsx": "^4.20.3",
    "typescript": "^5.8.3",
    "vitest": "^3.2.4"
  }
}
```

## Verification

After installation, verify the packages are installed:

```bash
# Check installed packages
pnpm list | grep @opentelemetry

# Should show:
# @opentelemetry/api x.x.x
# @opentelemetry/sdk-trace-node x.x.x
# @opentelemetry/exporter-trace-jaeger x.x.x
# @opentelemetry/exporter-trace-otlp-grpc x.x.x
# @opentelemetry/exporter-trace-otlp-http x.x.x
# @opentelemetry/context-async-hooks x.x.x
# @opentelemetry/resources x.x.x
# @opentelemetry/semantic-conventions x.x.x
```

## TypeScript Configuration

Ensure your `tsconfig.json` includes the following:

```json
{
  "compilerOptions": {
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "moduleResolution": "node"
  }
}
```

## Build Tracing Module

After installing dependencies, build the tracing module:

```bash
# From BubbleLab/packages/bubble-core
pnpm build

# Or build only the tracing module
npx tsc src/tracing/*.ts --outDir dist/tracing --module esnext --moduleResolution node --esModuleInterop
```

## Quick Test

Create a simple test to verify the installation:

```typescript
// test-tracing.ts
import { TracingManager } from '@bubblelab/bubble-core/tracing';

async function test() {
  const manager = TracingManager.getInstance();

  console.log('Tracing module loaded successfully!');
  console.log('Stats:', manager.getStats());

  await manager.initialize({
    serviceName: 'test-service',
    enabled: true,
    sampleRate: 1.0,
    exporter: {
      type: 'console',
      options: {},
    },
  });

  console.log('Tracing initialized successfully!');

  await manager.shutdown();
}

test().catch(console.error);
```

Run the test:
```bash
npx tsx test-tracing.ts
```

## Next Steps

After installing the packages:

1. ✅ Packages installed via `pnpm add`
2. ✅ Tracing module built with `pnpm build`
3. ✅ Verify installation with test script
4. ✅ Initialize tracing in your application
5. ✅ Start Jaeger with Docker Compose
6. ✅ Begin tracing your operations

## Troubleshooting

### Issue: Module not found

**Solution**: Make sure you've built the package after adding dependencies:
```bash
pnpm build
```

### Issue: Type errors

**Solution**: Ensure TypeScript configuration includes:
```json
{
  "compilerOptions": {
    "esModuleInterop": true,
    "moduleResolution": "node"
  }
}
```

### Issue: Version conflicts

**Solution**: Update all OpenTelemetry packages to the same version:
```bash
pnpm update @opentelemetry/* @^1.22.0
```

## Summary

This completes the package installation for distributed tracing. The OpenTelemetry packages are now available for use in the BubbleLab application.

**Files Modified:**
- `BubbleLab/packages/bubble-core/package.json`

**Next Action:**
Run `pnpm install` to install the new dependencies.
