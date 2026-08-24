import { describe, it, expect } from 'vitest';
import {
  LLM_PROVIDER_BASE_URLS,
  LLM_PROVIDER_CREDENTIALS,
} from './llm-providers.js';
import { CredentialType, SYSTEM_CREDENTIALS } from '@bubblelab/shared-schemas';

describe('llm-providers (NVIDIA NIM support)', () => {
  it('maps the nvidia provider prefix to the NVIDIA NIM OpenAI-compatible endpoint', () => {
    expect(LLM_PROVIDER_BASE_URLS.nvidia).toBe(
      'https://integrate.api.nvidia.com/v1'
    );
  });

  it('maps the nvidia provider prefix to the NVIDIA_NIM_CRED credential', () => {
    expect(LLM_PROVIDER_CREDENTIALS.nvidia).toBe(
      CredentialType.NVIDIA_NIM_CRED
    );
  });

  it('preserves existing provider mappings (regression)', () => {
    expect(LLM_PROVIDER_BASE_URLS.openrouter).toBe(
      'https://openrouter.ai/api/v1'
    );
    expect(LLM_PROVIDER_BASE_URLS.fireworks).toBe(
      'https://api.fireworks.ai/inference/v1'
    );
    expect(LLM_PROVIDER_CREDENTIALS.openrouter).toBe(
      CredentialType.OPENROUTER_CRED
    );
    expect(LLM_PROVIDER_CREDENTIALS.openai).toBe(CredentialType.OPENAI_CRED);
    expect(LLM_PROVIDER_CREDENTIALS.google).toBe(
      CredentialType.GOOGLE_GEMINI_CRED
    );
    expect(LLM_PROVIDER_CREDENTIALS.anthropic).toBe(
      CredentialType.ANTHROPIC_CRED
    );
    expect(LLM_PROVIDER_CREDENTIALS.fireworks).toBe(
      CredentialType.FIREWORKS_CRED
    );
  });

  it('exposes NVIDIA_NIM_CRED in the CredentialType enum and system credentials', () => {
    expect(CredentialType.NVIDIA_NIM_CRED).toBe('NVIDIA_NIM_CRED');
    expect(SYSTEM_CREDENTIALS.has(CredentialType.NVIDIA_NIM_CRED)).toBe(true);
  });
});
