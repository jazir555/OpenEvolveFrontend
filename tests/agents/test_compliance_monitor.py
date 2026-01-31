"""
Compliance Monitor Tests

Comprehensive test suite for the continuous compliance monitoring system.

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add openevolve to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'openevolve'))

from openevolve.agents.compliance_monitor import (
    ComplianceMonitor,
    CompliancePhase,
    AlertSeverity,
    ComplianceState
)
from openevolve.agents.compliance.regulatory_ingestor import (
    RegulatoryIngestor,
    RegulatoryChange,
    SourceType
)
from openevolve.agents.compliance.rule_evolver import (
    RuleEvolver,
    RuleStatus,
    EvolutionResult
)
from openevolve.agents.compliance.edge_discovery import (
    EdgeCaseDiscovery,
    EdgeCase,
    EdgeCaseType,
    CoverageReport
)
from openevolve.agents.compliance.verifier import (
    ComplianceVerifier,
    VerificationMethod,
    ProofType,
    VerificationResult
)
from openevolve.agents.compliance.alerter import (
    ComplianceAlerter,
    Alert,
    AlertSeverity,
    AlertStatus,
    EscalationLevel
)


class TestComplianceMonitor:
    """Test the main compliance monitor"""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a compliance monitor for testing"""
        return ComplianceMonitor(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            scan_interval_seconds=60,  # Fast for testing
            alert_threshold=AlertSeverity.MEDIUM,
            use_formal_verification=False,  # Faster for testing
            log_level="WARNING"
        )

    @pytest.mark.asyncio
    async def test_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor.state is not None
        assert monitor.state.regulatory_version == "0.0.0"
        assert monitor.state.current_rules == {}
        assert monitor._current_phase == CompliancePhase.MONITOR

    @pytest.mark.asyncio
    async def test_monitor_cycle(self, monitor):
        """Test one monitoring cycle"""
        await monitor._monitoring_cycle()

        # Should have completed all phases
        assert monitor.state.last_scan_time is not None
        assert monitor._current_phase == CompliancePhase.ALERT

    @pytest.mark.asyncio
    async def test_force_update(self, monitor):
        """Test forced update"""
        regulatory_changes = [
            {
                'title': 'Test Regulation',
                'description': 'Test change',
                'url': 'https://test.com',
                'change_type': 'new_rule'
            }
        ]

        await monitor.force_update(regulatory_changes)

        # Should have updated
        assert monitor.state.last_update_time is not None

    @pytest.mark.asyncio
    async def test_compliance_report(self, monitor):
        """Test compliance report generation"""
        report = await monitor.get_compliance_report()

        assert 'timestamp' in report
        assert 'regulatory_version' in report
        assert 'metrics' in report
        assert 'active_alerts' in report
        assert 'recommendations' in report

    def test_get_status(self, monitor):
        """Test status retrieval"""
        status = monitor.get_status()

        assert 'running' in status
        assert 'current_phase' in status
        assert 'regulatory_version' in status
        assert 'metrics' in status


class TestRegulatoryIngestor:
    """Test regulatory ingestor"""

    @pytest.fixture
    def ingestor(self, tmp_path):
        """Create ingestor for testing"""
        return RegulatoryIngestor(
            sources=['https://example.com/rss'],
            cache_dir=str(tmp_path / "regulatory"),
            logger=None
        )

    @pytest.mark.asyncio
    async def test_initialization(self, ingestor):
        """Test ingestor initialization"""
        assert ingestor.sources is not None
        assert len(ingestor.sources) > 0
        assert ingestor.seen_hashes == set()

    @pytest.mark.asyncio
    async def test_source_type_detection(self, ingestor):
        """Test source type detection"""
        rss_type = ingestor._detect_source_type('https://example.com/feed.rss')
        assert rss_type == SourceType.RSS_FEED

        web_type = ingestor._detect_source_type('https://example.com/page')
        assert web_type == SourceType.WEB_PAGE

    @pytest.mark.asyncio
    async def test_change_type_parsing(self, ingestor):
        """Test change type parsing"""
        repeal_type = ingestor._parse_change_type('Rule repealed by commission')
        assert repeal_type == 'repeal'

        amend_type = ingestor._parse_change_type('Amendment to Rule 10b-5')
        assert amend_type == 'amendment'

    @pytest.mark.asyncio
    async def test_affected_areas_extraction(self, ingestor):
        """Test affected areas extraction"""
        areas = ingestor._extract_affected_areas(
            'New rules for crypto trading and reporting requirements'
        )

        assert 'trading' in areas
        assert 'reporting' in areas
        assert 'crypto' in areas

    @pytest.mark.asyncio
    async def test_regulatory_change_creation(self, ingestor):
        """Test regulatory change object creation"""
        change = RegulatoryChange(
            source='test',
            title='Test Change',
            description='Test description',
            url='https://test.com',
            published_date=datetime.utcnow(),
            change_type='new_rule',
            raw_content='Test content'
        )

        assert change.content_hash != ''
        assert change.change_type == 'new_rule'

    @pytest.mark.asyncio
    async def test_ingest_changes(self, ingestor):
        """Test manual change ingestion"""
        changes = [
            {
                'source': 'test',
                'title': 'Test',
                'description': 'Test',
                'url': 'https://test.com',
                'published_date': datetime.utcnow().isoformat(),
                'change_type': 'new_rule',
                'raw_content': 'Test'
            }
        ]

        await ingestor.ingest_changes(changes)

        assert len(ingestor.changes) > 0


class TestRuleEvolver:
    """Test rule evolver"""

    @pytest.fixture
    def evolver(self, tmp_path):
        """Create evolver for testing"""
        return RuleEvolver(
            cache_dir=str(tmp_path / "evolution"),
            max_generations=2,  # Fast for testing
            population_size=3,
            logger=None
        )

    @pytest.fixture
    def sample_rules(self):
        """Sample compliance rules"""
        return {
            'rule_001': {
                'name': 'Test Rule',
                'description': 'Test rule for trading',
                'logic': 'if volume > 10000 then flag'
            }
        }

    @pytest.fixture
    def sample_changes(self):
        """Sample regulatory changes"""
        return [
            {
                'title': 'New volume threshold',
                'description': 'Volume threshold changed to 5000',
                'change_type': 'amendment'
            }
        ]

    def test_initialization(self, evolver):
        """Test evolver initialization"""
        assert evolver.max_generations == 2
        assert evolver.population_size == 3
        assert evolver.rule_history == []

    def test_generate_rule_id(self, evolver):
        """Test rule ID generation"""
        rule_id = evolver._generate_rule_id()
        assert rule_id.startswith('rule_')
        assert len(rule_id) > 10

    def test_get_next_version(self, evolver):
        """Test version numbering"""
        version = evolver._get_next_version()
        assert version == "1.0.0"

    @pytest.mark.asyncio
    async def test_evolve_rules(self, evolver, sample_rules, sample_changes):
        """Test rule evolution"""
        # This is a basic test - real evolution would take longer
        evolved = await evolver.evolve_rules(
            current_rules=sample_rules,
            regulatory_changes=sample_changes
        )

        # Should return rules (either evolved or original)
        assert isinstance(evolved, dict)


class TestEdgeCaseDiscovery:
    """Test edge case discovery"""

    @pytest.fixture
    def discovery(self, tmp_path):
        """Create edge case discovery for testing"""
        return EdgeCaseDiscovery(
            cache_dir=str(tmp_path / "edge_cases"),
            max_adversarial_iterations=5,  # Fast for testing
            max_fuzz_iterations=10,
            logger=None
        )

    @pytest.fixture
    def sample_rules(self):
        """Sample rules for testing"""
        return {
            'rule_001': {
                'name': 'Volume Threshold',
                'description': 'Flag trades over $10,000',
                'logic': 'if amount > 10000 then flag'
            }
        }

    def test_initialization(self, discovery):
        """Test discovery initialization"""
        assert discovery.edge_cases == []
        assert discovery.max_adversarial_iterations == 5

    @pytest.mark.asyncio
    async def test_boundary_testing(self, discovery, sample_rules):
        """Test boundary testing"""
        cases = await discovery._boundary_testing(sample_rules)

        # Should find boundary cases
        assert len(cases) > 0
        assert any(c.case_type == EdgeCaseType.BOUNDARY for c in cases)

    def test_extract_thresholds(self, discovery, sample_rules):
        """Test threshold extraction"""
        thresholds = discovery._extract_thresholds(sample_rules)

        # Should find numeric threshold
        assert len(thresholds) > 0
        assert any('10000' in str(t.get('value', '')) for t in thresholds)

    @pytest.mark.asyncio
    async def test_analyze_coverage(self, discovery, sample_rules):
        """Test coverage analysis"""
        # Add some edge cases
        case = EdgeCase(
            case_id='test_001',
            case_type=EdgeCaseType.BOUNDARY,
            description='Test case',
            scenario={},
            expected_behavior='Test',
            severity='low',
            affected_rules=['rule_001']
        )
        discovery.edge_cases.append(case)

        report = await discovery.analyze_coverage(sample_rules)

        assert report.total_rules == 1
        assert report.total_scenarios >= 1
        assert report.coverage_percentage >= 0

    def test_mark_case_addressed(self, discovery):
        """Test marking cases as addressed"""
        case = EdgeCase(
            case_id='test_001',
            case_type=EdgeCaseType.BOUNDARY,
            description='Test',
            scenario={},
            expected_behavior='Test',
            severity='low'
        )

        discovery.edge_cases.append(case)
        discovery.mark_case_addressed('test_001', 'Fixed by adding check')

        assert case.addressed == True
        assert case.mitigation == 'Fixed by adding check'


class TestComplianceVerifier:
    """Test compliance verifier"""

    @pytest.fixture
    def verifier(self):
        """Create verifier for testing"""
        return ComplianceVerifier(
            use_formal_methods=False,  # Faster for testing
            timeout_seconds=10,
            logger=None
        )

    @pytest.fixture
    def sample_rules(self):
        """Sample rules"""
        return {
            'rule_001': {
                'name': 'Test Rule',
                'description': 'Test compliance rule',
                'logic': 'if condition then action'
            }
        }

    def test_initialization(self, verifier):
        """Test verifier initialization"""
        assert verifier.timeout == 10
        assert verifier.logger is not None

    @pytest.mark.asyncio
    async def test_logical_verify(self, verifier, sample_rules):
        """Test logical verification"""
        result = await verifier._logical_verify(
            sample_rules,
            ProofType.CONSISTENCY
        )

        assert result.method == VerificationMethod.LOGICAL
        assert isinstance(result.success, bool)
        assert isinstance(result.confidence, float)

    @pytest.mark.asyncio
    async def test_verify_rules(self, verifier, sample_rules):
        """Test full verification workflow"""
        results = await verifier.verify_rules(
            rules=sample_rules,
            proof_types=[ProofType.CONSISTENCY]
        )

        assert len(results) > 0
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_extract_constraints(self, verifier, sample_rules):
        """Test constraint extraction"""
        constraints = verifier._extract_constraints(sample_rules)

        assert len(constraints) > 0
        assert all(hasattr(c, 'constraint_id') for c in constraints)

    def test_rules_contradict(self, verifier):
        """Test contradiction detection"""
        rule1 = {
            'description': 'Must require approval'
        }
        rule2 = {
            'description': 'Must not require approval'
        }

        contradicts = verifier._rules_contradict(rule1, rule2)
        # Should detect contradiction
        assert contradicts == True


class TestComplianceAlerter:
    """Test compliance alerter"""

    @pytest.fixture
    def alerter(self, tmp_path):
        """Create alerter for testing"""
        return ComplianceAlerter(
            threshold=AlertSeverity.MEDIUM,
            enable_fatigue_prevention=True,
            logger=None
        )

    @pytest.fixture
    def sample_violation(self):
        """Sample violation"""
        return {
            'title': 'Test Violation',
            'message': 'Test violation detected',
            'type': 'trading_violation',
            'risk_score': 75,
            'source': 'test_system'
        }

    def test_initialization(self, alerter):
        """Test alerter initialization"""
        assert alerter.threshold == AlertSeverity.MEDIUM
        assert alerter.enable_fatigue_prevention == True
        assert len(alerter.escalation_rules) > 0

    @pytest.mark.asyncio
    async def test_generate_alert(self, alerter, sample_violation):
        """Test alert generation"""
        alert = await alerter.generate_alert(sample_violation)

        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == 'Test Violation'
        assert alert.status == AlertStatus.OPEN

    @pytest.mark.asyncio
    async def test_alert_below_threshold(self, alerter):
        """Test alert below threshold"""
        violation = {
            'title': 'Minor Issue',
            'message': 'Low risk issue',
            'type': 'minor',
            'risk_score': 10  # Low
        }

        alert = await alerter.generate_alert(violation)

        # Should be filtered out
        assert alert is None

    def test_determine_severity(self, alerter):
        """Test severity determination"""
        violation1 = {'risk_score': 95}
        severity1 = alerter._determine_severity(violation1)
        assert severity1 == AlertSeverity.CRITICAL

        violation2 = {'risk_score': 50}
        severity2 = alerter._determine_severity(violation2)
        assert severity2 == AlertSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alerter):
        """Test alert acknowledgment"""
        violation = {
            'title': 'Test',
            'message': 'Test',
            'type': 'test',
            'risk_score': 70
        }

        alert = await alerter.generate_alert(violation)
        if alert:
            success = await alerter.acknowledge_alert(alert.alert_id, 'test_user')

            assert success == True
            assert alert.status == AlertStatus.ACKNOWLEDGED
            assert alert.acknowledged_by == 'test_user'

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alerter):
        """Test alert resolution"""
        violation = {
            'title': 'Test',
            'message': 'Test',
            'type': 'test',
            'risk_score': 70
        }

        alert = await alerter.generate_alert(violation)
        if alert:
            success = await alerter.resolve_alert(
                alert.alert_id,
                'test_user',
                is_false_positive=False
            )

            assert success == True
            assert alert.status == AlertStatus.RESOLVED

    def test_violation_signature(self, alerter):
        """Test violation signature generation"""
        violation = {
            'type': 'trading',
            'source': 'system_a',
            'rule_id': 'rule_001'
        }

        signature = alerter._get_violation_signature(violation)

        assert 'trading' in signature
        assert 'system_a' in signature
        assert 'rule_001' in signature

    def test_false_positive_score(self, alerter):
        """Test false positive scoring"""
        violation1 = {'risk_score': 90}
        score1 = alerter._calculate_false_positive_score(violation1)
        # High risk = low false positive score
        assert score1 < 0.5

        violation2 = {'risk_score': 10}
        score2 = alerter._calculate_false_positive_score(violation2)
        # Low risk = higher false positive score
        assert score2 > score1

    def test_get_alert_statistics(self, alerter):
        """Test statistics generation"""
        # Add some alerts
        stats = alerter.get_alert_statistics()

        assert 'total' in stats
        assert 'by_severity' in stats
        assert 'by_status' in stats
        assert 'false_positive_rate' in stats


class TestIntegration:
    """Integration tests for compliance monitoring system"""

    @pytest.fixture
    def system(self, tmp_path):
        """Create full system for testing"""
        monitor = ComplianceMonitor(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            scan_interval_seconds=60,
            alert_threshold=AlertSeverity.MEDIUM,
            use_formal_verification=False,
            log_level="WARNING"
        )
        return monitor

    @pytest.mark.asyncio
    async def test_full_workflow(self, system):
        """Test complete monitoring workflow"""
        # Start monitoring
        system._running = True

        # Run one cycle
        await system._monitoring_cycle()

        # Check state
        assert system.state.last_scan_time is not None
        assert system._current_phase == CompliancePhase.ALERT

        # Get report
        report = await system.get_compliance_report()
        assert 'metrics' in report

    @pytest.mark.asyncio
    async def test_regulatory_update_workflow(self, system):
        """Test workflow with regulatory updates"""
        # Simulate regulatory changes
        changes = [
            {
                'title': 'New SEC Rule',
                'description': 'Updated requirements',
                'url': 'https://sec.gov/new-rule',
                'change_type': 'new_rule'
            }
        ]

        # Force update
        await system.force_update(changes)

        # Verify update occurred
        assert system.state.last_update_time is not None

    @pytest.mark.asyncio
    async def test_alert_workflow(self, system):
        """Test alert generation and workflow"""
        # Create test violation
        violation = {
            'title': 'Critical Violation',
            'message': 'Major compliance breach',
            'type': 'critical',
            'risk_score': 95,
            'source': 'test'
        }

        # Generate alert
        alert = await system.alerter.generate_alert(violation)

        if alert:
            assert alert.severity == AlertSeverity.CRITICAL

            # Check escalation
            escalated = await system.alerter.escalate_alerts([alert])

            # Critical should escalate immediately
            assert len(escalated) > 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
