// Export monitoring functionality

export * from './monitoring_service';

// Export main MonitoringService class
import { MonitoringService } from './monitoring_service';
export { MonitoringService };

// Export monitoring utilities
export function createMonitoringService(config?: any): MonitoringService {
  return new MonitoringService(config);
}