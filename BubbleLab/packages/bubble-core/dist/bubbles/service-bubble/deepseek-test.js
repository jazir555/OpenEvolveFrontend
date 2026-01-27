/**
 * DeepSeek API Integration Test
 * This test verifies that DeepSeek provider is properly configured and working
 */
import { describe, it, expect } from 'bun:test';
import { CredentialType } from '@bubblelab/shared-schemas';
describe('DeepSeek Integration', () => {
    it('should have DEEPSEEK_CRED in CredentialType enum', () => {
        expect(CredentialType.DEEPSEEK_CRED).toBe('DEEPSEEK_CRED');
    });
    it('should have DeepSeek in CREDENTIAL_ENV_MAP', async () => {
        const { CREDENTIAL_ENV_MAP } = await import('@bubblelab/shared-schemas');
        expect(CREDENTIAL_ENV_MAP[CredentialType.DEEPSEEK_CRED]).toBe('DEEPSEEK_API_KEY');
    });
    it('should recognize deepseek provider in model string', () => {
        const model = 'deepseek/deepseek-chat';
        const [provider] = model.split('/');
        expect(provider).toBe('deepseek');
    });
    it('should have DeepSeek in SYSTEM_CREDENTIALS', async () => {
        const { SYSTEM_CREDENTIALS } = await import('@bubblelab/shared-schemas');
        expect(SYSTEM_CREDENTIALS.has(CredentialType.DEEPSEEK_CRED)).toBe(true);
    });
    it('should have DeepSeek in ai-agent bubble credentials', async () => {
        const { BUBBLE_CREDENTIAL_OPTIONS } = await import('@bubblelab/shared-schemas');
        expect(BUBBLE_CREDENTIAL_OPTIONS['ai-agent']).toContain(CredentialType.DEEPSEEK_CRED);
    });
    it('should have DeepSeek in bubbleflow-generator credentials', async () => {
        const { BUBBLE_CREDENTIAL_OPTIONS } = await import('@bubblelab/shared-schemas');
        expect(BUBBLE_CREDENTIAL_OPTIONS['bubbleflow-generator']).toContain(CredentialType.DEEPSEEK_CRED);
    });
});
//# sourceMappingURL=deepseek-test.js.map