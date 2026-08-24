import { describe, it, expect } from 'vitest';
import { CredentialType } from '@bubblelab/shared-schemas';
import { CREDENTIAL_TO_MODEL } from './generation-provider.js';

describe('generation-provider (NVIDIA NIM support)', () => {
  it('maps NVIDIA_NIM_CRED to a NVIDIA NIM model id (unprefixed catalog id)', () => {
    expect(CREDENTIAL_TO_MODEL[CredentialType.NVIDIA_NIM_CRED]).toBe(
      'deepseek-ai/deepseek-v4-flash-0731'
    );
  });

  it('preserves existing generation model mappings (regression)', () => {
    expect(CREDENTIAL_TO_MODEL[CredentialType.OPENROUTER_CRED]).toBe(
      'openrouter/anthropic/claude-sonnet-4.5'
    );
    expect(CREDENTIAL_TO_MODEL[CredentialType.OPENAI_CRED]).toBe('openai/gpt-4o');
    expect(CREDENTIAL_TO_MODEL[CredentialType.GOOGLE_GEMINI_CRED]).toBe(
      'google/gemini-3-flash-preview'
    );
  });
});
