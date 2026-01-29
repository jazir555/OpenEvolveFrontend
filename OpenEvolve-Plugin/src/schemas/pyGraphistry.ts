/**
 * PyGraphistry Configuration Schema
 * 
 * Defines the configuration schema for the PyGraphistry node
 */

import { z } from 'zod';

export const pyGraphistryConfigSchema = z.object({
  layout: z.enum(['force_directed', 'circular', 'hierarchical']).optional().default('force_directed'),
  clustering: z.boolean().optional().default(false),
  clusteringMethod: z.enum(['dbscan', 'kmeans']).optional().default('dbscan'),
  enableGPUAcceleration: z.boolean().optional().default(true),
  apiKey: z.string().optional().default(''),
  serverUrl: z.string().optional().default('http://localhost:8000'),
  enableBackendExecution: z.boolean().optional().default(true),
  backendUrl: z.string().optional().default('http://localhost:8000'),
});

export type PyGraphistryConfig = z.infer<typeof pyGraphistryConfigSchema>;

export default pyGraphistryConfigSchema;