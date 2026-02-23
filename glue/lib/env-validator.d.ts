export type EnvType = 'string' | 'number' | 'url' | 'port' | 'boolean';
export interface EnvVar {
    name: string;
    type: EnvType;
    required: boolean;
    default?: string | number | boolean;
}
export interface ValidationResult {
    valid: boolean;
    errors: string[];
}
export declare function validateEnv(required: string[]): void;
export declare function validateEnvWithTypes(vars: EnvVar[]): Record<string, any>;
export declare function getEnv(name: string, type?: EnvType): any;
//# sourceMappingURL=env-validator.d.ts.map