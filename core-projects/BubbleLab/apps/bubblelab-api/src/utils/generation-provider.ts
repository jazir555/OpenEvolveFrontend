import { db } from '../db/index.js';
import { userCredentials } from '../db/schema.js';
import { eq, and } from 'drizzle-orm';
import { CredentialType, AvailableModel } from '@bubblelab/shared-schemas';

/**
 * Mapping from credential types to their model identifiers
 * for generation providers (LLMs used for workflow generation)
 */
const CREDENTIAL_TO_MODEL: Record<string, AvailableModel> = {
  [CredentialType.GOOGLE_GEMINI_CRED]: 'google/gemini-3-flash-preview',
  [CredentialType.DEEPSEEK_CRED]: 'deepseek/deepseek-chat',
  [CredentialType.OPENAI_CRED]: 'openai/gpt-4o',
  [CredentialType.ANTHROPIC_CRED]: 'anthropic/claude-sonnet-4-5',
  [CredentialType.OPENROUTER_CRED]: 'openrouter/anthropic/claude-sonnet-4.5',
};

/**
 * Get the default generation model based on user's default credential
 * Falls back to Google Gemini if no default is set
 *
 * @param userId - The user ID to get the default credential for
 * @returns The model identifier string (e.g., 'google/gemini-3-flash-preview', 'deepseek/deepseek-chat')
 */
export async function getDefaultGenerationModel(
  userId: string
): Promise<AvailableModel> {
  try {
    // List of credential types that can be used for generation, in priority order
    const generationCredentialTypes: CredentialType[] = [
      CredentialType.GOOGLE_GEMINI_CRED,
      CredentialType.DEEPSEEK_CRED,
      CredentialType.OPENAI_CRED,
      CredentialType.ANTHROPIC_CRED,
      CredentialType.OPENROUTER_CRED,
    ];

    // Check each credential type to see if user has a default one
    for (const credentialType of generationCredentialTypes) {
      const defaultCredential = await db.query.userCredentials.findFirst({
        where: and(
          eq(userCredentials.userId, userId),
          eq(userCredentials.credentialType, credentialType),
          eq(userCredentials.isDefault, true)
        ),
        columns: {
          id: true,
          credentialType: true,
        },
      });

      if (defaultCredential) {
        const model = CREDENTIAL_TO_MODEL[credentialType];
        if (model) {
          console.log(
            `[GenerationProvider] Using default credential: ${credentialType} -> ${model}`
          );
          return model;
        }
      }
    }

    // No default credential found, fall back to Google Gemini
    console.log(
      '[GenerationProvider] No default credential found, using fallback: google/gemini-3-flash-preview'
    );
    return 'google/gemini-3-flash-preview';
  } catch (error) {
    console.error('[GenerationProvider] Error getting default generation model:', error);
    // On error, fall back to Google Gemini
    return 'google/gemini-3-flash-preview';
  }
}

/**
 * Get the default generation credential for a user
 *
 * @param userId - The user ID
 * @returns The credential type if a default is set, or null
 */
export async function getDefaultGenerationCredentialType(
  userId: string
): Promise<CredentialType | null> {
  try {
    const generationCredentialTypes: CredentialType[] = [
      CredentialType.GOOGLE_GEMINI_CRED,
      CredentialType.DEEPSEEK_CRED,
      CredentialType.OPENAI_CRED,
      CredentialType.ANTHROPIC_CRED,
      CredentialType.OPENROUTER_CRED,
    ];

    for (const credentialType of generationCredentialTypes) {
      const defaultCredential = await db.query.userCredentials.findFirst({
        where: and(
          eq(userCredentials.userId, userId),
          eq(userCredentials.credentialType, credentialType),
          eq(userCredentials.isDefault, true)
        ),
        columns: {
          id: true,
          credentialType: true,
        },
      });

      if (defaultCredential) {
        return credentialType;
      }
    }

    return null;
  } catch (error) {
    console.error('[GenerationProvider] Error getting default credential type:', error);
    return null;
  }
}
