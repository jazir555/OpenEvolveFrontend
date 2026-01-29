# Changelog

All notable changes to BubbleLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- OpenTelemetry integration for distributed tracing
- Horizontal Pod Autoscaler (HPA) configurations
- Comprehensive runbooks for operations
- Security incident response procedures
- Performance monitoring dashboards
- Automated backup and recovery scripts

### Changed
- Improved error handling in workflow execution
- Enhanced logging with structured output
- Updated dependencies to latest stable versions
- Optimized database queries for better performance

### Fixed
- Memory leak in long-running workflow executions
- Race condition in parallel bubble execution
- Token counting for AI responses
- Webhook timeout handling

### Security
- Added encryption for stored credentials
- Implemented rate limiting on API endpoints
- Enhanced input validation across all endpoints
- Fixed CORS configuration issues

## [1.0.0] - 2026-01-15

### Added
- Initial stable release of BubbleLab
- Visual workflow builder (Bubble Studio)
- Core workflow execution engine
- AI-powered workflow generation (Pearl AI)
- 20+ pre-built bubbles including:
  - HTTP requests
  - AI agents (OpenAI, Anthropic, Google)
  - PostgreSQL database operations
  - Slack integration
  - Email sending (Resend)
  - Code execution
  - Data transformation
  - Webhooks
- Workflow templates
- Export workflows as TypeScript
- Execution history and tracing
- User authentication (Clerk)
- Credential management
- API documentation
- Docker deployment support
- Kubernetes deployment manifests

### Features
- **Prompt to Workflow**: Describe what you want in natural language
- **Full Observability**: Built-in execution tracing and logging
- **Export as TypeScript**: Own your workflows completely
- **Import from n8n**: Migrate existing workflows
- **Type Safety**: Full TypeScript support throughout

### Documentation
- Comprehensive README
- Contributing guidelines
- Deployment guide
- API documentation
- Bubble reference guide

## [0.9.0] - 2025-12-20

### Added
- Beta release for public testing
- Core bubble system implementation
- Basic workflow execution engine
- Frontend UI skeleton
- API server with Hono framework
- Database schema with Drizzle ORM

### Changed
- Migrated from Express to Hono for better performance
- Switched from npm to pnpm for faster installs
- Updated to Bun runtime for backend

### Fixed
- Initial stability issues
- Memory management problems
- Database connection pooling

## [0.1.0] - 2025-11-01

### Added
- Initial proof of concept
- Basic workflow builder UI
- Simple execution engine
- First bubble implementations

---

## Versioning Strategy

BubbleLab follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Cadence

- **Major releases**: Quarterly (significant new features)
- **Minor releases**: Monthly (new features, improvements)
- **Patch releases**: As needed (bug fixes, security updates)

### Pre-release Versions

- **Alpha**: Early development, not feature-complete
- **Beta**: Feature-complete, testing needed
- **RC**: Release candidate, testing final version

---

## Release Process

### 1. Development
```bash
git checkout -b feature/new-feature
# ... development ...
git checkout develop
git merge feature/new-feature
```

### 2. Pre-release
```bash
git checkout -b release/1.0.0
# ... update version numbers ...
git checkout main
git merge release/1.0.0
git tag -a v1.0.0 -m "Release v1.0.0"
```

### 3. Post-release
```bash
git checkout develop
git merge main
```

---

## Types of Changes

### Added
- New features
- New bubbles
- New integrations
- New documentation

### Changed
- Changes in existing functionality
- Improvements to existing features
- Refactoring code

### Deprecated
- Soon-to-be removed features
- Features being replaced

### Removed
- Removed features
- Removed dependencies

### Fixed
- Bug fixes
- Security fixes
- Performance improvements

### Security
- Security vulnerability fixes
- Security enhancements
- Dependency updates for security

---

## Upgrade Guides

### From 0.9.x to 1.0.0

**Breaking Changes:**
- Database schema has changed - run migrations
- API endpoints have been reorganized
- Some bubble parameters have changed

**Migration Steps:**

1. Backup your data
2. Update code to latest version
3. Run database migrations
4. Update environment variables
5. Test workflows in development
6. Deploy to production

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.

---

## Future Releases

### Upcoming in 1.1.0
- [ ] Real-time collaboration on workflows
- [ ] Version control for workflows
- [ ] Workflow scheduling
- [ ] Enhanced error recovery
- [ ] Additional AI provider integrations
- [ ] Mobile app for monitoring

### Roadmap
- [ ] Workflow marketplace
- [ ] Custom bubble development kit
- [ ] Advanced debugging tools
- [ ] Performance profiling
- [ ] Multi-region deployment support
- [ ] Enterprise SSO integration
- [ ] Advanced analytics dashboard

---

## Contributors

Thanks to everyone who has contributed to BubbleLab!

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for the full list.

---

## Support

- **Documentation**: https://docs.bubblelab.ai/
- **Discord**: https://discord.gg/PkJvcU2myV
- **GitHub Issues**: https://github.com/bubblelabai/BubbleLab/issues
- **Email**: support@bubblelab.ai

---

*This changeline follows the [Keep a Changelog](https://keepachangelog.com/) format.*
