import { CredentialType } from '@bubblelab/shared-schemas';
import { AccessibilityAgent } from './agents/AccessibilityAgent.js';
import { BrandAgent } from './agents/BrandAgent.js';
import { ConversionAgent } from './agents/ConversionAgent.js';
import { LayoutAgent } from './agents/LayoutAgent.js';
import { VisualLLMJudge } from './visual-llm-judge.js';
import type { JudgeInput } from './types.js';

const credentials = {
  [CredentialType.OPENAI_CRED]: 'test-openai',
  [CredentialType.ANTHROPIC_CRED]: 'test-anthropic',
  [CredentialType.GOOGLE_GEMINI_CRED]: 'test-google',
  [CredentialType.OPENROUTER_CRED]: 'test-openrouter',
};

describe('BaseVisualJudge parsing', () => {
  test('parses valid JSON response', async () => {
    const agent = new LayoutAgent();
    jest.spyOn(agent, 'action').mockResolvedValue({
      success: true,
      data: {
        response: JSON.stringify({
          score: 0.8,
          reasoning: 'Solid layout hierarchy',
          highlights: ['Clear CTA'],
          issues: [],
          recommendations: ['Tighten spacing'],
        }),
      },
    } as any);

    const result = await agent.evaluate(
      { image: { type: 'base64', data: 'abc' } },
      credentials
    );

    expect(result.score).toBe(0.8);
    expect(result.reasoning).toContain('layout');
  });

  test('returns fallback on invalid JSON', async () => {
    const agent = new LayoutAgent();
    jest.spyOn(agent, 'action').mockResolvedValue({
      success: true,
      data: { response: 'not-json' },
    } as any);

    const result = await agent.evaluate(
      { image: { type: 'base64', data: 'abc' } },
      credentials
    );

    expect(result.score).toBe(0);
    expect(result.issues.length).toBeGreaterThan(0);
  });

  test('parses accessibility response', async () => {
    const agent = new AccessibilityAgent();
    jest.spyOn(agent, 'action').mockResolvedValue({
      success: true,
      data: {
        response: JSON.stringify({
          score: 0.6,
          reasoning: 'Contrast needs improvement',
          highlights: ['Readable headings'],
          issues: ['Low contrast body text'],
          recommendations: ['Increase contrast ratio'],
        }),
      },
    } as any);

    const result = await agent.evaluate(
      { image: { type: 'base64', data: 'abc' } },
      credentials
    );

    expect(result.score).toBe(0.6);
    expect(result.issues).toContain('Low contrast body text');
  });

  test('parses brand response', async () => {
    const agent = new BrandAgent();
    jest.spyOn(agent, 'action').mockResolvedValue({
      success: true,
      data: {
        response: JSON.stringify({
          score: 0.7,
          reasoning: 'Brand palette feels cohesive',
          highlights: ['Consistent palette'],
          issues: [],
          recommendations: ['Clarify tone in hero copy'],
        }),
      },
    } as any);

    const result = await agent.evaluate(
      { image: { type: 'base64', data: 'abc' } },
      credentials
    );

    expect(result.score).toBe(0.7);
    expect(result.highlights).toContain('Consistent palette');
  });

  test('parses conversion response', async () => {
    const agent = new ConversionAgent();
    jest.spyOn(agent, 'action').mockResolvedValue({
      success: true,
      data: {
        response: JSON.stringify({
          score: 0.65,
          reasoning: 'CTA stands out but lacks urgency',
          highlights: ['Clear CTA'],
          issues: ['Weak urgency'],
          recommendations: ['Add urgency copy near CTA'],
        }),
      },
    } as any);

    const result = await agent.evaluate(
      { image: { type: 'base64', data: 'abc' } },
      credentials
    );

    expect(result.score).toBe(0.65);
    expect(result.recommendations).toContain('Add urgency copy near CTA');
  });
});

describe('VisualLLMJudge aggregation', () => {
  test('aggregates weighted score', async () => {
    const judge = new VisualLLMJudge();
    const agents = (judge as any).agents as Array<{ evaluate: jest.Mock }>;

    agents.forEach((agent, index) => {
      jest.spyOn(agent, 'evaluate').mockResolvedValue({
        agent: agent.agent,
        provider: agent.providerName,
        score: 0.5 + index * 0.1,
        reasoning: 'ok',
        highlights: [],
        issues: [],
        recommendations: [],
        rawResponse: '{}',
        costUsd: 0,
      });
    });

    const input: JudgeInput = { image: { type: 'base64', data: 'abc' } };
    const result = await judge.evaluate(input, credentials);

    expect(result.score).toBeGreaterThan(0);
    expect(result.agents).toHaveLength(4);
  });
});
