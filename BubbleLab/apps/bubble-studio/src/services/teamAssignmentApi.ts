/**
 * Team LLM Assignment API Client
 *
 * Handles flexible LLM assignment to teams with unified credential management
 */

import { ApiClient, ApiClientConfig } from '@/lib/api';
import { OPENEVOLVE_API_BASE_URL } from '@/env';
import { logger } from '@/utils/logger';
import type {
  Team,
  TeamMemberLLM,
  TeamRole,
  TeamAssignmentRequest,
  LLMModel,
  LLMProvider,
  LLMCapability,
  LLMSearchFilters,
  LLMSearchResponse,
  CredentialsListResponse,
  CredentialVerificationRequest,
  CredentialVerificationResponse,
  CredentialFormData,
  TeamCreateResponse,
  TeamTemplate,
  LLMGroup,
} from '@/types/team-assignment';

const teamAssignmentClientConfig: ApiClientConfig = {
  baseURL: OPENEVOLVE_API_BASE_URL,
  timeout: 30000,
  enableRetry: true,
  maxRetries: 3,
  retryDelay: 1000,
};

const teamAssignmentClient = new ApiClient(teamAssignmentClientConfig);

/**
 * Team LLM Assignment API Client
 */
export const teamAssignmentApi = {
  // ==================== LLM Catalog ====================

  /**
   * Get catalog of available LLMs
   */
  getLLMCatalog: async (filters?: LLMSearchFilters): Promise<LLMSearchResponse> => {
    logger.debug({
      msg: 'Fetching LLM catalog',
      component: 'teamAssignmentApi',
      filters,
    });

    const params = new URLSearchParams();
    if (filters?.provider) params.append('provider', filters.provider);
    if (filters?.capability) params.append('capability', filters.capability);
    if (filters?.vision_only) params.append('vision_only', 'true');

    const queryString = params.toString();
    const url = `/api/teams/llms/catalog${queryString ? `?${queryString}` : ''}`;

    return teamAssignmentClient.get<LLMSearchResponse>(url);
  },

  /**
   * Get supported LLM providers
   */
  getProviders: async (): Promise<{
    providers: Array<{
      id: string;
      name: string;
      vision_support: boolean;
    }>;
  }> => {
    return teamAssignmentClient.get('/api/teams/llms/providers');
  },

  /**
   * Get only vision LLMs (vLLMs)
   */
  getVisionLLMs: async (): Promise<LLMSearchResponse> => {
    return teamAssignmentApi.getLLMCatalog({ vision_only: true });
  },

  // ==================== Credential Management ====================

  /**
   * List all available credentials
   */
  listCredentials: async (): Promise<CredentialsListResponse> => {
    logger.debug({
      msg: 'Listing credentials',
      component: 'teamAssignmentApi',
    });

    return teamAssignmentClient.get<CredentialsListResponse>('/api/teams/credentials');
  },

  /**
   * Verify a credential by making a test API call
   */
  verifyCredential: async (
    request: CredentialVerificationRequest,
  ): Promise<CredentialVerificationResponse> => {
    logger.info({
      msg: 'Verifying credential',
      component: 'teamAssignmentApi',
      provider: request.provider,
    });

    return teamAssignmentClient.post<CredentialVerificationResponse>(
      '/api/teams/credentials/verify',
      request,
    );
  },

  /**
   * Add new credential (from BubbleLab credentials tab)
   */
  addCredential: async (
    credentialData: CredentialFormData,
  ): Promise<{ credential_id: string; message: string }> => {
    logger.info({
      msg: 'Adding new credential',
      component: 'teamAssignmentApi',
      provider: credentialData.provider,
    });

    // This would call BubbleLab credentials API
    // For now, verify and save through OpenEvolve
    const verification = await teamAssignmentApi.verifyCredential({
      provider: credentialData.provider,
      api_key: credentialData.api_key,
      api_base: credentialData.api_base,
      model_to_test: 'gpt-3.5-turbo', // Default test model
    });

    if (!verification.verified) {
      throw new Error(verification.message);
    }

    return {
      credential_id: verification.credential_id!,
      message: 'Credential verified and saved',
    };
  },

  // ==================== Team Management ====================

  /**
   * Create a new team with LLM members
   */
  createTeam: async (
    team: Omit<Team, 'team_id'>,
  ): Promise<TeamCreateResponse> => {
    logger.info({
      msg: 'Creating team',
      component: 'teamAssignmentApi',
      name: team.name,
      member_count: team.members.length,
    });

    return teamAssignmentClient.post<TeamCreateResponse>(
      '/api/teams/teams',
      team,
    );
  },

  /**
   * List all teams
   */
  listTeams: async (): Promise<{
    teams: Team[];
    total: number;
  }> => {
    return teamAssignmentClient.get('/api/teams/teams');
  },

  /**
   * Get specific team
   */
  getTeam: async (teamId: string): Promise<Team> => {
    return teamAssignmentClient.get<Team>(`/api/teams/teams/${teamId}`);
  },

  /**
   * Update team
   */
  updateTeam: async (
    teamId: string,
    team: Partial<Team>,
  ): Promise<Team> => {
    return teamAssignmentClient.put<Team>(
      `/api/teams/teams/${teamId}`,
      team,
    );
  },

  /**
   * Delete team
   */
  deleteTeam: async (teamId: string): Promise<{ message: string }> => {
    return teamAssignmentClient.delete<{ message: string }>(
      `/api/teams/teams/${teamId}`,
    );
  },

  /**
   * Add LLM member to team
   */
  addTeamMember: async (
    teamId: string,
    member: Omit<TeamMemberLLM, 'member_id'>,
  ): Promise<{
    member_id: string;
    team_id: string;
    member: TeamMemberLLM;
    added_at: string;
  }> => {
    logger.info({
      msg: 'Adding team member',
      component: 'teamAssignmentApi',
      team_id: teamId,
      llm_provider: member.llm.provider,
      llm_model: member.llm.model_id,
      role: member.role,
    });

    return teamAssignmentClient.post(
      `/api/teams/teams/${teamId}/members`,
      member,
    );
  },

  /**
   * Remove LLM member from team
   */
  removeTeamMember: async (
    teamId: string,
    memberId: string,
  ): Promise<{ message: string }> => {
    return teamAssignmentClient.delete<{ message: string }>(
      `/api/teams/teams/${teamId}/members/${memberId}`,
    );
  },

  // ==================== Quick Assignment ====================

  /**
   * Quick assign LLM to team
   */
  assignLLMToTeam: async (
    request: TeamAssignmentRequest,
  ): Promise<{
    member_id: string;
    team_id: string;
    llm: LLMModel;
    role: TeamRole;
  }> => {
    logger.info({
      msg: 'Assigning LLM to team',
      component: 'teamAssignmentApi',
      team_id: request.team_id,
      llm: request.llm_model_id,
      role: request.role,
    });

    return teamAssignmentClient.post('/api/teams/teams/assign', request);
  },

  // ==================== Team Templates ====================

  /**
   * Get predefined team templates
   */
  getTeamTemplates: async (): Promise<{ templates: TeamTemplate[] }> => {
    return teamAssignmentClient.get('/api/teams/teams/templates');
  },

  /**
   * Create team from template
   */
  createTeamFromTemplate: async (
    templateId: string,
    teamName: string,
  ): Promise<TeamCreateResponse> => {
    logger.info({
      msg: 'Creating team from template',
      component: 'teamAssignmentApi',
      template_id: templateId,
      team_name: teamName,
    });

    // Get template
    const { templates } = await teamAssignmentApi.getTeamTemplates();
    const template = templates.find(t => t.id === templateId);

    if (!template) {
      throw new Error(`Template ${templateId} not found`);
    }

    // Resolve LLMs from template
    const members: TeamMemberLLM[] = [];

    for (const comp of template.composition) {
      // Get LLM details
      const catalog = await teamAssignmentApi.getLLMCatalog();
      const llm = catalog.llms.find(l => l.model_id === comp.llm);

      if (!llm) {
        throw new Error(`LLM ${comp.llm} not found in catalog`);
      }

      // Add multiple members if count > 1
      for (let i = 0; i < comp.count; i++) {
        members.push({
          member_id: `member_${Date.now()}_${i}`,
          llm,
          role: comp.role,
          temperature: 0.7,
          max_tokens: 4096,
          total_requests: 0,
          successful_requests: 0,
        });
      }
    }

    // Create team
    return teamAssignmentApi.createTeam({
      name: teamName,
      description: template.description,
      members,
      ...DEFAULT_TEAM_COMPOSITION,
    });
  },
};

// ==================== Default Values ====================

const DEFAULT_TEAM_COMPOSITION = {
  voting_strategy: 'consensus' as const,
  quorum_threshold: 0.7,
  require_vision_for_design: true,
  require_diverse_providers: false,
};

export { DEFAULT_TEAM_COMPOSITION };
