#!/usr/bin/env node

/**
 * Verification script for metrics and monitoring system
 *
 * Run this to verify the monitoring system is properly configured
 */

import { initializeMonitoring, getMetricsCollector, HealthChecker, getTracer, getAlertManager } from './index';

async function verify() {
  console.log('🔍 Verifying OpenEvolve Metrics and Monitoring System...\n');

  // 1. Test initialization
  console.log('✅ Testing initialization...');
  try {
    await initializeMonitoring({
      serviceName: 'test-service',
      prometheus: {
        prefix: 'test_',
      },
      health: {
        enabled: true,
      },
      alerts: {
        enabled: true,
      },
    });
    console.log('   ✓ Monitoring system initialized\n');
  } catch (error) {
    console.error('   ✗ Failed to initialize:', error);
    process.exit(1);
  }

  // 2. Test metrics collector
  console.log('✅ Testing metrics collector...');
  try {
    const metrics = getMetricsCollector();

    // Record some test metrics
    metrics.recordHttpRequestDuration(
      { service: 'test', operation: 'test', status: '2xx' },
      0.5
    );
    metrics.incrementHttpRequests({
      service: 'test',
      operation: 'test',
      status: '2xx',
    });

    // Get metrics
    const metricsText = await metrics.getMetrics();
    if (!metricsText.includes('test_http_request_duration_seconds')) {
      throw new Error('Metrics not found in output');
    }

    console.log('   ✓ Metrics collector working\n');
  } catch (error) {
    console.error('   ✗ Metrics collector failed:', error);
    process.exit(1);
  }

  // 3. Test health checker
  console.log('✅ Testing health checker...');
  try {
    const health = new HealthChecker('test-service');

    // Register a simple health check
    health.register('test', async () => ({
      name: 'test',
      status: 'healthy',
      message: 'Test health check',
      timestamp: new Date().toISOString(),
    }));

    // Check health
    const result = await health.checkHealth();
    if (result.status !== 'healthy') {
      throw new Error('Health check failed');
    }

    console.log('   ✓ Health checker working\n');
  } catch (error) {
    console.error('   ✗ Health checker failed:', error);
    process.exit(1);
  }

  // 4. Test tracer
  console.log('✅ Testing tracer...');
  try {
    const tracer = getTracer('test-service');

    // Trace a simple operation
    const result = await tracer.traceAsync(
      {
        name: 'test-operation',
        correlationId: 'test-123',
      },
      async (span) => {
        span.setAttributes({ test: 'value' });
        return 'success';
      }
    );

    if (result !== 'success') {
      throw new Error('Tracer failed');
    }

    console.log('   ✓ Tracer working\n');
  } catch (error) {
    console.error('   ✗ Tracer failed:', error);
    process.exit(1);
  }

  // 5. Test alert manager
  console.log('✅ Testing alert manager...');
  try {
    const alerts = getAlertManager('test-service');

    // Register a test rule
    alerts.registerRule({
      id: 'test-rule',
      name: 'Test Rule',
      description: 'Test alert rule',
      severity: 'info',
      condition: {
        type: 'custom',
        eval: (data) => data.trigger === true,
      },
      notifications: [{ type: 'log', config: {} }],
      enabled: true,
    });

    // Evaluate rules (should not trigger)
    const triggered = await alerts.evaluateRules({ trigger: false });
    if (triggered.length !== 0) {
      throw new Error('Alert should not have triggered');
    }

    // Evaluate rules (should trigger)
    const triggeredAlerts = await alerts.evaluateRules({ trigger: true });
    if (triggeredAlerts.length !== 1) {
      throw new Error('Alert should have triggered');
    }

    console.log('   ✓ Alert manager working\n');
  } catch (error) {
    console.error('   ✗ Alert manager failed:', error);
    process.exit(1);
  }

  // Summary
  console.log('✅ All verification tests passed!\n');
  console.log('📊 Metrics and monitoring system is ready to use.\n');
  console.log('Environment variables:');
  console.log(`   PROMETHEUS_PORT: ${process.env.PROMETHEUS_PORT || '9090 (default)'}`);
  console.log(`   OTEL_EXPORTER_OTLP_ENDPOINT: ${process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4317 (default)'}`);
  console.log(`   SERVICE_NAME: ${process.env.SERVICE_NAME || 'not set'}`);
  console.log(`   METRICS_PREFIX: ${process.env.METRICS_PREFIX || 'openevolve_ (default)'}`);
  console.log();
}

// Run verification
verify().catch((error) => {
  console.error('\n❌ Verification failed:', error);
  process.exit(1);
});
