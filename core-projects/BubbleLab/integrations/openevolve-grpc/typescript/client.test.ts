/**
 * Unit tests for the OpenEvolve gRPC TypeScript client.
 *
 * These tests deliberately do NOT require a running gRPC server. They cover
 * the behaviour that is verifiable offline:
 *   - proto discovery / loading (regression test for the dist/ __dirname bug)
 *   - service + health stub extraction
 *   - config defaulting
 *
 * Anything that requires an actual server (connect(), executeNode(), streaming,
 * health RPCs) is intentionally NOT asserted here and still requires live e2e.
 */

import * as path from 'path';
import { OpenEvolveGRPCClient, createGRPCClient } from './client';

describe('OpenEvolveGRPCClient - proto loading', () => {
  it('constructs successfully and loads proto definitions', () => {
    const client = createGRPCClient();
    expect(client).toBeInstanceOf(OpenEvolveGRPCClient);
  });

  it('extracts the NodeRegistry service stub from nodes.proto', () => {
    const client = createGRPCClient() as unknown as { nodeRegistry: unknown };
    expect(typeof client.nodeRegistry).toBe('function');
  });

  it('extracts the grpc.health.v1.Health stub from health.proto', () => {
    const client = createGRPCClient() as unknown as { healthClient: unknown };
    expect(typeof client.healthClient).toBe('function');
  });

  it('honours an explicit protoDir override', () => {
    const protoDir = path.join(__dirname, '..', 'proto');
    const client = createGRPCClient({ protoDir });
    expect(client).toBeInstanceOf(OpenEvolveGRPCClient);
  });

  it('throws a descriptive error when protoDir does not contain the protos', () => {
    expect(() => createGRPCClient({ protoDir: path.join(__dirname, 'does-not-exist') })).toThrow(
      /Unable to locate the OpenEvolve proto directory/
    );
  });
});

describe('OpenEvolveGRPCClient - configuration', () => {
  it('applies default host/port when not supplied', () => {
    const client = createGRPCClient() as unknown as {
      config: { host: string; port: number; poolSize: number };
    };
    expect(client.config.host).toBe('localhost');
    expect(client.config.port).toBe(50051);
    expect(client.config.poolSize).toBe(5);
  });

  it('allows overriding defaults', () => {
    const client = createGRPCClient({ host: 'example.internal', port: 6000 }) as unknown as {
      config: { host: string; port: number };
    };
    expect(client.config.host).toBe('example.internal');
    expect(client.config.port).toBe(6000);
  });

  it('reports not-connected before connect() is called', () => {
    const client = createGRPCClient() as unknown as { isConnected: boolean };
    expect(client.isConnected).toBe(false);
  });

  it('has no health check result before any check runs', () => {
    const client = createGRPCClient();
    expect(client.getLastHealthCheck()).toBeUndefined();
  });

  it('fails with a clear error when an RPC is issued before connect()', async () => {
    const client = createGRPCClient();
    await expect(client.listNodes()).rejects.toThrow(/call connect\(\) before issuing requests/);
  });
});
