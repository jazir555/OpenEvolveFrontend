# Production Cutover Plan

**Document Version:** 1.0
**Last Updated:** 2026-01-06
**Owner:** Agent 5 (Deployment & Operations Engineer)

---

## Executive Summary

This document outlines the production cutover plan for migrating from BubbleLab UI to BubbleLab React UI. The cutover will be executed with zero-downtime using a blue-green deployment strategy.

**Timeline:** Approximately 2 hours
**Risk Level:** Medium (mitigated by comprehensive testing and rollback plan)
**Downtime:** Zero (planned)

---

## Pre-Cutover Checklist

### Environment Setup
- [ ] Staging environment fully operational
- [ ] Production environment configured and tested
- [ ] DNS records configured (staging.openevolve.ai, openevolve.ai)
- [ ] SSL certificates obtained and installed
- [ ] Database backups automated and tested
- [ ] Monitoring tools configured (Prometheus, Grafana, Alertmanager)
- [ ] Log aggregation configured (Loki)

### Code Readiness
- [ ] All 126+ tests passing
- [ ] Code coverage ≥ 87%
- [ ] Security scan clean (no critical vulnerabilities)
- [ ] Performance benchmarks met (p95 < 500ms)
- [ ] API documentation updated
- [ ] Runbooks created and reviewed

### Infrastructure
- [ ] Docker images built and pushed to registry
- [ ] Container registry configured and accessible
- [ ] Load balancer configured (Nginx)
- [ ] CDN configured (Cloudflare, if applicable)
- [ ] Backup strategy tested and verified
- [ ] Rollback procedure tested

### Team Readiness
- [ ] On-call engineer assigned
- [ ] Rollback plan documented and communicated
- [ ] Communication plan prepared
- [ ] User notification drafted
- [ ] All stakeholders informed of cutover window

---

## Cutover Timeline

### Phase 1: Pre-Cutover Preparation (T-60 minutes)

**Time: T-60 to T-30 minutes**

**Objectives:**
- Verify staging environment stability
- Create final backup of production BubbleLab UI
- Prepare rollback plan
- Notify team of impending cutover

**Tasks:**
1. Verify staging environment health
   ```bash
   bash deploy/scripts/health-check.sh staging http://staging.openevolve.ai
   bash deploy/scripts/smoke-tests.sh staging http://staging.openevolve.ai
   ```

2. Create production backup
   ```bash
   bash deploy/scripts/backup-production.sh
   ```

3. Verify monitoring dashboards
   - Grafana dashboards loaded
   - Alertmanager rules active
   - Slack alerts configured

4. Team communication
   - Send notification to #openevolve-team
   - Confirm on-call engineer availability
   - Verify rollback procedure understanding

**Success Criteria:**
- All health checks passing
- Backup created successfully
- All team members acknowledged

---

### Phase 2: Pre-Cutover Deployment (T-30 minutes)

**Time: T-30 to T-10 minutes**

**Objectives:**
- Deploy API Gateway to production (behind feature flag)
- Deploy BubbleLab UI to production (behind feature flag)
- Run smoke tests on new deployment
- Monitor metrics

**Tasks:**
1. Deploy API Gateway
   ```bash
   cd deploy/production
   docker-compose -f docker-compose.production.yml pull api-gateway
   docker-compose -f docker-compose.production.yml up -d api-gateway
   ```

2. Deploy BubbleLab UI
   ```bash
   docker-compose -f docker-compose.production.yml pull bubblelab-frontend
   docker-compose -f docker-compose.production.yml up -d bubblelab-frontend
   ```

3. Update Nginx to route internal traffic to new services
   ```bash
   # Update nginx.conf with new upstream configuration
   docker-compose -f docker-compose.production.yml exec nginx nginx -s reload
   ```

4. Run smoke tests
   ```bash
   bash deploy/scripts/smoke-tests.sh production https://openevolve.ai
   ```

5. Monitor metrics (Grafana dashboards)
   - API response times
   - Error rates
   - Database connections
   - Redis performance

**Success Criteria:**
- All services healthy
- Smoke tests passing (100%)
- Error rate < 1%
- Response time p95 < 500ms

---

### Phase 3: Final Checks (T-10 minutes)

**Time: T-10 to T-0 minutes**

**Objectives:**
- Verify all systems green
- Confirm team readiness
- Start monitoring dashboards
- Final verification

**Tasks:**
1. Final health check
   ```bash
   bash deploy/scripts/health-check.sh production https://openevolve.ai
   ```

2. Verify metrics
   - API Gateway: healthy
   - Frontend: healthy
   - Database: healthy
   - Redis: healthy
   - WebSocket: healthy

3. Team readiness confirmation
   - On-call engineer: ready
   - Database team: notified
   - Support team: notified

4. Start monitoring dashboards
   - Open Grafana dashboard
   - Enable Slack alerts
   - Start recording metrics

**Success Criteria:**
- All health checks passing
- All team members ready
- Monitoring dashboards active

---

### Phase 4: Cutover Execution (T-0)

**Time: T-0 to T+30 minutes**

**Objectives:**
- Update DNS to point to new UI
- Enable feature flags for new system
- Monitor health metrics
- Verify user traffic flowing to new UI

**Tasks:**
1. **T-0: DNS Update**
   - Update DNS A record for openevolve.ai to point to new Nginx
   - Set TTL to 300 seconds (5 minutes)
   - Verify DNS propagation

   ```bash
   # Via Cloudflare API or DNS provider
   # Update A record: openevolve.ai -> <new IP>
   ```

2. **T+5: Feature Flag Enablement**
   - Enable BubbleLab UI feature flag
   - Disable BubbleLab UI UI feature flag
   - Verify routing

3. **T+10: Traffic Verification**
   - Check Google Analytics / Cloudflare Analytics
   - Verify traffic flowing to new UI
   - Monitor error rates

4. **T+20: WebSocket Verification**
   - Verify WebSocket connections stable
   - Check real-time features working
   - Monitor connection counts

5. **T+30: Full Verification**
   - Run comprehensive health checks
   - Verify all features working
   - Check user feedback channels

**Success Criteria:**
- DNS propagated
- User traffic flowing to new UI
- Error rate < 1%
- Response time p95 < 500ms
- WebSocket connections stable

---

### Phase 5: Verification (T+30 to T+60 minutes)

**Time: T+30 to T+60 minutes**

**Objectives:**
- Verify all systems operational
- Check error rates and response times
- Verify WebSocket connections stable
- Keep BubbleLab UI available as fallback

**Tasks:**
1. Run comprehensive smoke tests
   ```bash
   bash deploy/scripts/smoke-tests.sh production https://openevolve.ai
   ```

2. Check metrics
   - Error rate: should be < 1%
   - Response time p95: should be < 500ms
   - WebSocket connections: stable
   - Database performance: normal
   - Redis performance: normal

3. Verify user experience
   - Check support tickets
   - Monitor user feedback
   - Check social media mentions

4. Monitor for issues
   - Check Grafana dashboards
   - Review error logs
   - Monitor Slack alerts

**Success Criteria:**
- All smoke tests passing
- Error rate < 1%
- Response time p95 < 500ms
- No critical user issues

---

### Phase 6: Stabilization (T+60 to T+120 minutes)

**Time: T+60 to T+120 minutes**

**Objectives:**
- Monitor for issues
- Address any problems
- Keep BubbleLab UI available as fallback
- Prepare for BubbleLab UI decommissioning

**Tasks:**
1. Continuous monitoring
   - Watch Grafana dashboards
   - Monitor error logs
   - Track user feedback

2. Address issues
   - Fix any bugs discovered
   - Address user concerns
   - Update documentation as needed

3. Performance tuning
   - Optimize database queries if needed
   - Adjust cache settings
   - Tune WebSocket configuration

4. Prepare for decommissioning
   - Final verification of BubbleLab UI replacement
   - Plan BubbleLab UI shutdown
   - Archive BubbleLab UI code

**Success Criteria:**
- System stable for 60+ minutes
- No critical issues
- All user concerns addressed

---

### Phase 7: Completion (T+120 minutes)

**Time: T+120 minutes**

**Objectives:**
- Decommission BubbleLab UI
- Archive old code
- Update documentation
- Send user notification

**Tasks:**
1. Decommission BubbleLab UI
   ```bash
   bash deploy/scripts/decommission-BubbleLab UI.sh
   ```

2. Archive old code
   - Move BubbleLab UI files to archive
   - Update repository structure
   - Commit changes

3. Update documentation
   - Update README.md
   - Update API documentation
   - Update deployment guides

4. Send user notification
   - Announce UI migration
   - Highlight new features
   - Provide feedback channel

5. Retrospective
   - Document lessons learned
   - Identify improvements
   - Update runbooks

**Success Criteria:**
- BubbleLab UI decommissioned
- All documentation updated
- Users notified
- Retrospective completed

---

## Rollback Plan

If critical issues detected at any point:

### Immediate Rollback (T+0 to T+30)

**Trigger:**
- Error rate > 5%
- Critical functionality broken
- Security vulnerability discovered
- Data loss detected

**Actions:**
1. Revert DNS changes
   ```bash
   # Revert DNS A record to point to BubbleLab UI
   ```

2. Re-enable BubbleLab UI
   ```bash
   cd deploy/production
   docker-compose -f docker-compose.production.yml up -d BubbleLab UI
   ```

3. Stop BubbleLab services
   ```bash
   docker-compose -f docker-compose.production.yml stop api-gateway bubblelab-frontend
   ```

4. Verify BubbleLab UI operational
   ```bash
   bash deploy/scripts/health-check.sh production https://openevolve.ai
   ```

5. Investigate and fix issues
   - Review logs
   - Analyze metrics
   - Fix bugs
   - Test fixes in staging

6. Schedule retry cutover
   - Minimum 24 hours later
   - During low-traffic period
   - With additional testing

### Full Rollback (T+30 to T+120)

**Trigger:**
- Persistent performance degradation
- User complaints > 10% of users
- Data inconsistency discovered

**Actions:**
1. Execute rollback script
   ```bash
   bash deploy/scripts/rollback.sh
   ```

2. Verify system restored
   ```bash
   bash deploy/scripts/health-check.sh production https://openevolve.ai
   ```

3. Notify team and users
   - Send rollback notification
   - Explain issue
   - Provide timeline for retry

---

## Post-Cutover Tasks

### Immediate (Day 1)
- [ ] Monitor system continuously
- [ ] Address user issues promptly
- [ ] Track performance metrics
- [ ] Collect user feedback

### Short-term (Week 1)
- [ ] Optimize performance
- [ ] Fix discovered bugs
- [ ] Update documentation
- [ ] Train users on new UI

### Long-term (Month 1)
- [ ] Conduct retrospective
- [ ] Implement improvements
- [ ] Plan next phase of features
- [ ] Archive old infrastructure

---

## Communication Plan

### Pre-Cutover
- **Team:** 24 hours notice
- **Stakeholders:** 24 hours notice
- **Users:** 1 hour notice (banner on site)

### During Cutover
- **Team:** Real-time updates in Slack
- **Stakeholders:** Email at T-60, T-30, T-0, T+30, T+120

### Post-Cutover
- **Users:** Announcement email
- **Team:** Retrospective meeting

---

## Success Metrics

- ✅ Zero downtime
- ✅ Error rate < 1%
- ✅ Response time p95 < 500ms
- ✅ WebSocket connections stable
- ✅ Zero data loss
- ✅ BubbleLab UI decommissioned
- ✅ User satisfaction > 90%

---

## Appendix

### A. Emergency Contacts
- On-call Engineer: [Name, Phone]
- Engineering Lead: [Name, Phone]
- Product Manager: [Name, Phone]

### B. Related Documents
- Deployment Runbook
- Troubleshooting Guide
- Monitoring Dashboard Guide
- User Migration Guide

### C. Change History
- 2026-01-06: Initial document creation (Agent 5)

