/**
 * API module exports
 * Re-exports from services/api for backward compatibility
 */

export { apiClient } from '@/services/api/client';
export * from '@/services/api/endpoints';
export { createEvolutionWebSocket } from '@/services/api/websocket';
