-- OpenEvolve Database Initialization Script
-- This script is automatically run when PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create schemas
CREATE SCHEMA IF NOT EXISTS openevolve;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO openevolve, public;

-- ============================================
-- USERS AND AUTHENTICATION
-- ============================================

-- Users table
CREATE TABLE IF NOT EXISTS openevolve.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes
CREATE INDEX idx_users_email ON openevolve.users(email);
CREATE INDEX idx_users_username ON openevolve.users(username);
CREATE INDEX idx_users_is_active ON openevolve.users(is_active);

-- Sessions table
CREATE TABLE IF NOT EXISTS openevolve.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES openevolve.users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

CREATE INDEX idx_sessions_user_id ON openevolve.sessions(user_id);
CREATE INDEX idx_sessions_token ON openevolve.sessions(session_token);
CREATE INDEX idx_sessions_expires_at ON openevolve.sessions(expires_at);

-- ============================================
-- WORKSPACES AND PROJECTS
-- ============================================

-- Workspaces table
CREATE TABLE IF NOT EXISTS openevolve.workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL REFERENCES openevolve.users(id) ON DELETE CASCADE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_workspaces_owner_id ON openevolve.workspaces(owner_id);

-- Workspace members
CREATE TABLE IF NOT EXISTS openevolve.workspace_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES openevolve.workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES openevolve.users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member', -- 'owner', 'admin', 'member', 'viewer'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workspace_id, user_id)
);

CREATE INDEX idx_workspace_members_workspace_id ON openevolve.workspace_members(workspace_id);
CREATE INDEX idx_workspace_members_user_id ON openevolve.workspace_members(user_id);

-- Projects table
CREATE TABLE IF NOT EXISTS openevolve.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES openevolve.workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100), -- 'evolutionary', 'adversarial', 'maker', 'decomposition'
    config JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'paused', 'completed', 'archived'
    created_by UUID NOT NULL REFERENCES openevolve.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_projects_workspace_id ON openevolve.projects(workspace_id);
CREATE INDEX idx_projects_type ON openevolve.projects(type);
CREATE INDEX idx_projects_status ON openevolve.projects(status);
CREATE INDEX idx_projects_created_by ON openevolve.projects(created_by);

-- ============================================
-- EXPERIMENTS AND RUNS
-- ============================================

-- Experiments table
CREATE TABLE IF NOT EXISTS openevolve.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID NOT NULL REFERENCES openevolve.projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'created', -- 'created', 'running', 'completed', 'failed', 'cancelled'
    metrics JSONB DEFAULT '{}',
    artifacts JSONB DEFAULT '[]',
    created_by UUID REFERENCES openevolve.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_experiments_project_id ON openevolve.experiments(project_id);
CREATE INDEX idx_experiments_status ON openevolve.experiments(status);
CREATE INDEX idx_experiments_created_at ON openevolve.experiments(created_at DESC);

-- Experiment runs (for multiple trials)
CREATE TABLE IF NOT EXISTS openevolve.experiment_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES openevolve.experiments(id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    config JSONB DEFAULT '{}',
    results JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(experiment_id, run_number)
);

CREATE INDEX idx_experiment_runs_experiment_id ON openevolve.experiment_runs(experiment_id);
CREATE INDEX idx_experiment_runs_status ON openevolve.experiment_runs(status);

-- ============================================
-- KNOWLEDGE ENGINE
-- ============================================

-- Knowledge entities
CREATE TABLE IF NOT EXISTS openevolve.knowledge_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES openevolve.projects(id) ON DELETE CASCADE,
    type VARCHAR(100) NOT NULL, -- 'concept', 'relation', 'rule', 'pattern'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    properties JSONB DEFAULT '{}',
    embeddings VECTOR(1536), -- Requires pgvector extension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_knowledge_entities_project_id ON openevolve.knowledge_entities(project_id);
CREATE INDEX idx_knowledge_entities_type ON openevolve.knowledge_entities(type);
CREATE INDEX idx_knowledge_entities_name_gin ON openevolve.knowledge_entities USING gin(name gin_trgm_ops);

-- Knowledge relations
CREATE TABLE IF NOT EXISTS openevolve.knowledge_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES openevolve.knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES openevolve.knowledge_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    properties JSONB DEFAULT '{}',
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);

CREATE INDEX idx_knowledge_relations_source ON openevolve.knowledge_relations(source_entity_id);
CREATE INDEX idx_knowledge_relations_target ON openvolve.knowledge_relations(target_entity_id);

-- ============================================
-- FILES AND ARTIFACTS
-- ============================================

-- Files table
CREATE TABLE IF NOT EXISTS openevolve.files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES openevolve.projects(id) ON DELETE CASCADE,
    experiment_id UUID REFERENCES openevolve.experiments(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    checksum VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    uploaded_by UUID REFERENCES openevolve.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_files_project_id ON openevolve.files(project_id);
CREATE INDEX idx_files_experiment_id ON openevolve.files(experiment_id);

-- ============================================
-- AUDIT LOGGING
-- ============================================

-- Audit log table
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES openevolve.users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_user_id ON audit.audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit.audit_logs(action);
CREATE INDEX idx_audit_logs_resource ON audit.audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_created_at ON audit.audit_logs(created_at DESC);

-- ============================================
-- ANALYTICS AND METRICS
-- ============================================

-- Usage metrics
CREATE TABLE IF NOT EXISTS analytics.usage_metrics (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES openevolve.users(id) ON DELETE SET NULL,
    project_id UUID REFERENCES openevolve.projects(id) ON DELETE SET NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dimensions JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_metrics_user_id ON analytics.usage_metrics(user_id);
CREATE INDEX idx_usage_metrics_project_id ON analytics.usage_metrics(project_id);
CREATE INDEX idx_usage_metrics_timestamp ON analytics.usage_metrics(timestamp DESC);

-- Performance metrics
CREATE TABLE IF NOT EXISTS analytics.performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_metrics_endpoint ON analytics.performance_metrics(endpoint);
CREATE INDEX idx_performance_metrics_timestamp ON analytics.performance_metrics(timestamp DESC);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION openevolve.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON openevolve.users
    FOR EACH ROW EXECUTE FUNCTION openevolve.update_updated_at_column();

CREATE TRIGGER update_workspaces_updated_at BEFORE UPDATE ON openevolve.workspaces
    FOR EACH ROW EXECUTE FUNCTION openevolve.update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON openevolve.projects
    FOR EACH ROW EXECUTE FUNCTION openevolve.update_updated_at_column();

CREATE TRIGGER update_knowledge_entities_updated_at BEFORE UPDATE ON openevolve.knowledge_entities
    FOR EACH ROW EXECUTE FUNCTION openevolve.update_updated_at_column();

-- ============================================
-- INITIAL DATA
-- ============================================

-- Create default admin user (password: admin123 - CHANGE THIS IMMEDIATELY!)
INSERT INTO openevolve.users (email, username, hashed_password, full_name, is_superuser, is_verified)
VALUES (
    'admin@openevolve.ai',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5aqBkYLmfQhCq', -- admin123
    'System Administrator',
    true,
    true
) ON CONFLICT (email) DO NOTHING;

-- Create default workspace
INSERT INTO openevolve.workspaces (name, description, owner_id)
VALUES (
    'Default Workspace',
    'Default workspace for new projects',
    (SELECT id FROM openevolve.users WHERE email = 'admin@openevolve.ai' LIMIT 1)
) ON CONFLICT DO NOTHING;

-- ============================================
-- GRANTS
-- ============================================

-- Grant necessary permissions (adjust based on your security requirements)
GRANT USAGE ON SCHEMA openevolve, audit, analytics TO openvolve;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA openevolve, audit, analytics TO openvolve;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA openevolve, audit, analytics TO openvolve;
ALTER DEFAULT PRIVILEGES IN SCHEMA openevolve, audit, analytics GRANT ALL ON TABLES TO openvolve;
ALTER DEFAULT PRIVILEGES IN SCHEMA openevolve, audit, analytics GRANT USAGE, SELECT ON SEQUENCES TO openvolve;

-- ============================================
-- VACUUM ANALYZE
-- ============================================

VACUUM ANALYZE;
