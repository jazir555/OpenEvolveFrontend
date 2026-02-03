/**
 * OpenEvolve parameter schemas generated from parameter_definitions.py
 *
 * Provides both nested (category-based) and flattened parameter schemas
 * for use in BubbleLab OpenEvolve bubbles.
 */

import { z } from 'zod';
import definitions from './openevolve-parameter-definitions.json';

export type OpenEvolveParameterDefinition = {
  type: string;
  default?: unknown;
  description?: string;
  min_value?: number;
  max_value?: number;
  options?: string[];
  required?: boolean;
};

export type OpenEvolveParameterDefinitions = Record<
  string,
  Record<string, OpenEvolveParameterDefinition>
>;

export const OPENEVOLVE_PARAMETER_DEFINITIONS =
  definitions as OpenEvolveParameterDefinitions;

const buildNumericSchema = (
  definition: OpenEvolveParameterDefinition
): z.ZodNumber => {
  let schema = z.number();
  if (definition.type === 'integer') {
    schema = schema.int();
  }
  if (typeof definition.min_value === 'number') {
    schema = schema.min(definition.min_value);
  }
  if (typeof definition.max_value === 'number') {
    schema = schema.max(definition.max_value);
  }
  return schema;
};

const buildParameterSchema = (
  definition: OpenEvolveParameterDefinition
): z.ZodTypeAny => {
  let schema: z.ZodTypeAny;

  switch (definition.type) {
    case 'integer':
    case 'float':
      schema = buildNumericSchema(definition);
      break;
    case 'boolean':
      schema = z.boolean();
      break;
    case 'select':
      if (Array.isArray(definition.options) && definition.options.length > 0) {
        schema = z.enum(definition.options as [string, ...string[]]);
      } else {
        schema = z.string();
      }
      break;
    case 'list':
      schema = z.array(z.unknown());
      break;
    case 'dict':
      schema = z.record(z.unknown());
      break;
    case 'string':
    default:
      schema = z.string();
      break;
  }

  if (definition.default === null) {
    schema = schema.nullable();
  }

  if (!definition.required) {
    schema = schema.optional();
  }

  if (definition.default !== undefined && definition.default !== null) {
    schema = schema.default(definition.default);
  }

  return schema;
};

const buildCategorySchema = (
  category: Record<string, OpenEvolveParameterDefinition>
): z.ZodObject<z.ZodRawShape> => {
  const shape: Record<string, z.ZodTypeAny> = {};
  for (const [name, definition] of Object.entries(category)) {
    shape[name] = buildParameterSchema(definition);
  }
  return z.object(shape).partial().passthrough();
};

const buildNestedSchema = (): z.ZodObject<z.ZodRawShape> => {
  const shape: Record<string, z.ZodTypeAny> = {};
  for (const [category, params] of Object.entries(
    OPENEVOLVE_PARAMETER_DEFINITIONS
  )) {
    shape[category] = buildCategorySchema(params).optional();
  }
  return z.object(shape).partial().passthrough();
};

const buildFlatSchema = (): z.ZodObject<z.ZodRawShape> => {
  const shape: Record<string, z.ZodTypeAny> = {};
  for (const [category, params] of Object.entries(
    OPENEVOLVE_PARAMETER_DEFINITIONS
  )) {
    for (const [name, definition] of Object.entries(params)) {
      shape[`${category}.${name}`] = buildParameterSchema(definition);
    }
  }
  return z.object(shape).partial().passthrough();
};

export const OpenEvolveParametersSchema = buildNestedSchema();
export const OpenEvolveFlatParametersSchema = buildFlatSchema();

export type OpenEvolveParameters = z.infer<typeof OpenEvolveParametersSchema>;
export type OpenEvolveFlatParameters = z.infer<
  typeof OpenEvolveFlatParametersSchema
>;
