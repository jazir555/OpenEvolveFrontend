// Export configuration functionality

export * from './config_mapper';
export * from './config_generator';

// Export main classes
import { ConfigMapper } from './config_mapper';
import { ConfigGenerator } from './config_generator';
export { ConfigMapper, ConfigGenerator };

// Export configuration utilities
export function createConfigMapper(): ConfigMapper {
  return new ConfigMapper();
}

export function createConfigGenerator(): ConfigGenerator {
  return new ConfigGenerator();
}