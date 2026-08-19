/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * Local type stub mirroring the Graphiti canonical module
 * (`../../../schemas/graphiti-canonical`). The import in graphiti-memory.ts is
 * type-only and is erased from the emitted JavaScript, so this file has no
 * runtime footprint and keeps the adapter self-contained.
 */

export type EpisodeType = string;

export interface AddEpisodeOperation {
  name: string;
  content: string;
  source_description: string;
  episode_type: EpisodeType;
  valid_at: string;
  group_id: string;
  uuid: string;
  update_communities: boolean;
}

export interface AddEpisodeResult {
  success: boolean;
  episode_id?: string;
  entities_extracted: number;
  relationships_extracted: number;
  processing_time_ms: number;
}

export interface CanonicalEntityEdge {
  fact: string;
  created_at: string;
  metadata?: any;
  attributes?: any;
  [key: string]: any;
}

export interface CanonicalEpisode {
  [key: string]: any;
}

export interface CanonicalEntity {
  [key: string]: any;
}

export interface CanonicalSearchResult {
  edges: CanonicalEntityEdge[];
  [key: string]: any;
}
