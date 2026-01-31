"""
Continuous Compliance Monitoring Agent
A never-sleeping compliance officer that monitors regulatory changes 24/7,
adapts rule sets, tests edge cases, and provides mathematical proofs of compliance.

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Import compliance modules
from .compliance.regulatory_ingestor import RegulatoryIngestor
from .compliance.rule_evolver import RuleEvolver
from .compliance.edge_discovery import EdgeCaseDiscovery
from .compliance.verifier import ComplianceVerifier
from .compliance.alerter import ComplianceAlerter

# Import long-horizon framework components (when available)
try:
    from ..long_horizon.state_manager import StateManager
    from ..long_horizon.workflow_orchestrator import WorkflowOrchestrator
    LONG_HORIZON_AVAILABLE = True
except ImportError:
    LONG_HORIZON_AVAILABLE = False
    StateManager = None
    WorkflowOrchestrator = None

# Import unified evolution API
from ..unified.unified_evolution_api import evolve


class CompliancePhase(Enum):
    """Compliance monitoring phases"""
    MONITOR = "monitor"
    UPDATE = "update"
    EVOLVE = "evolve"
    DEPLOY = "deploy"
    ALERT = "alert"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceState:
    """Persistent state for compliance monitoring"""
    current_rules: Dict[str, Any] = field(default_factory=dict)
    regulatory_version: str = "0.0.0"
    last_scan_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    edge_cases_found: List[Dict[str, Any]] = field(default_factory=list)
    rule_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            'current_rules': self.current_rules,
            'regulatory_version': self.regulatory_version,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'active_alerts': self.active_alerts,
            'edge_cases_found': self.edge_cases_found,
            'rule_history': self.rule_history,
            'metrics': self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceState':
        """Create from dictionary"""
        state = cls(
            current_rules=data.get('current_rules', {}),
            regulatory_version=data.get('regulatory_version', '0.0.0'),
            active_alerts=data.get('active_alerts', []),
            edge_cases_found=data.get('edge_cases_found', []),
            rule_history=data.get('rule_history', []),
            metrics=data.get('metrics', {})
        )
        if data.get('last_scan_time'):
            state.last_scan_time = datetime.fromisoformat(data['last_scan_time'])
        if data.get('last_update_time'):
            state.last_update_time = datetime.fromisoformat(data['last_update_time'])
        return state


class ComplianceMonitor:
    """
    Continuous Compliance Monitoring Agent

    Orchestrates the continuous compliance monitoring workflow:
    1. Monitor Phase: Scan regulatory sources and internal systems
    2. Update Phase: Analyze changes and plan updates
    3. Evolve Phase: Evolve rules using LoongFlow PES
    4. Deploy Phase: Deploy and monitor new rules
    5. Alert Phase: Generate and manage alerts

    Example:
        >>> monitor = ComplianceMonitor(checkpoint_dir="/checkpoints")
        >>> await monitor.start()
        >>> # Runs continuously until stopped
        >>> await monitor.stop()
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        scan_interval_seconds: int = 3600,  # 1 hour
        regulatory_sources: Optional[List[str]] = None,
        alert_threshold: AlertSeverity = AlertSeverity.MEDIUM,
        use_formal_verification: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize compliance monitor

        Args:
            checkpoint_dir: Directory for checkpointing state
            scan_interval_seconds: Seconds between regulatory scans
            regulatory_sources: List of regulatory source URLs
            alert_threshold: Minimum severity for alerts
            use_formal_verification: Enable formal verification
            log_level: Logging level
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints/compliance")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scan_interval = timedelta(seconds=scan_interval_seconds)
        self.regulatory_sources = regulatory_sources or self._get_default_sources()
        self.alert_threshold = alert_threshold
        self.use_formal_verification = use_formal_verification

        # Setup logging
        self.logger = self._setup_logging(log_level)

        # Initialize state management
        if LONG_HORIZON_AVAILABLE and StateManager:
            self.state_manager = StateManager(
                checkpoint_dir=str(self.checkpoint_dir),
                max_checkpoints=100
            )
        else:
            self.state_manager = None
            self.logger.warning("Long-horizon StateManager not available, using basic persistence")

        # Initialize compliance modules
        self.ingestor = RegulatoryIngestor(
            sources=self.regulatory_sources,
            logger=self.logger
        )
        self.evolver = RuleEvolver(
            logger=self.logger
        )
        self.edge_discovery = EdgeCaseDiscovery(
            logger=self.logger
        )
        self.verifier = ComplianceVerifier(
            use_formal_methods=self.use_formal_verification,
            logger=self.logger
        )
        self.alerter = ComplianceAlerter(
            threshold=self.alert_threshold,
            logger=self.logger
        )

        # Current state
        self.state = ComplianceState()
        self._running = False
        self._current_phase = CompliancePhase.MONITOR

        # Load previous state if available
        self._load_state()

    def _setup_logging(self, level: str) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("ComplianceMonitor")
        logger.setLevel(getattr(logging, level.upper()))

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _get_default_sources(self) -> List[str]:
        """Get default regulatory sources"""
        return [
            "https://www.sec.gov/news/pressreleases",
            "https://www.finra.org/rules-guidance/rulebooks",
            "https://www.esma.europa.eu/press-releases",
            # Add more sources as needed
        ]

    def _load_state(self):
        """Load state from checkpoint"""
        checkpoint_file = self.checkpoint_dir / "compliance_state.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                self.state = ComplianceState.from_dict(data)
                self.logger.info(f"Loaded state from checkpoint (version {self.state.regulatory_version})")
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")

    def _save_state(self):
        """Save state to checkpoint"""
        checkpoint_file = self.checkpoint_dir / "compliance_state.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            self.logger.debug("State checkpointed")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    async def start(self):
        """
        Start continuous compliance monitoring

        Runs the monitoring loop until stop() is called
        """
        if self._running:
            self.logger.warning("Compliance monitor already running")
            return

        self._running = True
        self.logger.info("Starting continuous compliance monitoring")

        try:
            while self._running:
                await self._monitoring_cycle()

                # Wait for next scan interval
                if self._running:
                    await asyncio.sleep(self.scan_interval.total_seconds())

        except Exception as e:
            self.logger.error(f"Fatal error in monitoring loop: {e}", exc_info=True)
            raise
        finally:
            self._running = False
            self.logger.info("Compliance monitoring stopped")

    async def stop(self):
        """Stop continuous monitoring"""
        self.logger.info("Stopping compliance monitoring")
        self._running = False
        self._save_state()

    async def _monitoring_cycle(self):
        """
        Execute one complete monitoring cycle

        1. Monitor: Scan for changes
        2. Update: Plan updates if changes detected
        3. Evolve: Evolve rules using LoongFlow
        4. Deploy: Deploy and verify
        5. Alert: Handle violations
        """
        self.logger.info("Starting monitoring cycle")

        # Phase 1: Monitor
        await self._monitor_phase()

        # Phase 2: Update (if changes detected)
        if await self._should_update():
            await self._update_phase()

            # Phase 3: Evolve
            await self._evolve_phase()

            # Phase 4: Deploy
            await self._deploy_phase()

        # Phase 5: Alert (always run)
        await self._alert_phase()

        # Update metrics
        await self._update_metrics()

        # Checkpoint state
        self._save_state()

        self.logger.info("Monitoring cycle complete")

    async def _monitor_phase(self):
        """Monitor regulatory sources and internal systems"""
        self._current_phase = CompliancePhase.MONITOR
        self.logger.info("Phase 1: Monitoring regulatory sources")

        # Scan regulatory sources
        regulatory_changes = await self.ingestor.scan_sources()

        # Scan internal systems for violations
        internal_violations = await self._scan_internal_systems()

        # Update state
        self.state.last_scan_time = datetime.utcnow()

        if regulatory_changes:
            self.logger.info(f"Detected {len(regulatory_changes)} regulatory changes")
            self.state.metrics['regulatory_changes_detected'] = len(regulatory_changes)

        if internal_violations:
            self.logger.warning(f"Found {len(internal_violations)} internal violations")
            self.state.metrics['internal_violations'] = len(internal_violations)

    async def _should_update(self) -> bool:
        """Check if update is needed"""
        # Update if:
        # 1. Regulatory changes detected
        # 2. Edge cases found that need addressing
        # 3. Rule performance degraded

        changes_detected = await self.ingestor.has_changes()
        edge_cases_need_attention = any(
            not case.get('addressed', False)
            for case in self.state.edge_cases_found
        )

        return changes_detected or edge_cases_need_attention

    async def _update_phase(self):
        """Plan rule updates using LoongFlow"""
        self._current_phase = CompliancePhase.UPDATE
        self.logger.info("Phase 2: Planning rule updates")

        # Get regulatory changes
        changes = await self.ingestor.get_changes()

        # Analyze impact and plan updates
        update_plan = await self._plan_updates(changes)

        self.logger.info(f"Update plan: {update_plan}")

    async def _plan_updates(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Plan rule updates using LoongFlow

        Args:
            changes: List of regulatory changes

        Returns:
            Update plan with affected systems and test scenarios
        """
        problem_statement = f"""
        Analyze the following regulatory changes and create an update plan:

        Changes:
        {json.dumps(changes, indent=2)}

        Current Rules:
        {json.dumps(self.state.current_rules, indent=2)}

        Create a plan that includes:
        1. Affected systems and rules
        2. Required rule updates
        3. Test scenarios to verify compliance
        4. Risk assessment
        """

        try:
            result = await evolve(
                problem_statement=problem_statement,
                mode="standard",
                max_generations=5
            )

            if result.get('success'):
                return json.loads(result.get('best_solution', '{}'))
            else:
                self.logger.error("Failed to generate update plan")
                return {}

        except Exception as e:
            self.logger.error(f"Error planning updates: {e}")
            return {}

    async def _evolve_phase(self):
        """Evolve rule sets using LoongFlow PES"""
        self._current_phase = CompliancePhase.EVOLVE
        self.logger.info("Phase 3: Evolving rule sets")

        # Get current rules and required changes
        changes = await self.ingestor.get_changes()

        # Evolve rules using PES
        evolved_rules = await self.evolver.evolve_rules(
            current_rules=self.state.current_rules,
            regulatory_changes=changes
        )

        # Discover edge cases
        edge_cases = await self.edge_discovery.discover_cases(
            rules=evolved_rules
        )

        self.state.edge_cases_found.extend(edge_cases)

        # Verify rules
        if self.use_formal_verification:
            verification_result = await self.verifier.verify_rules(evolved_rules)
            self.logger.info(f"Verification result: {verification_result}")

        # Update state
        self.state.current_rules = evolved_rules

    async def _deploy_phase(self):
        """Deploy and monitor updated rules"""
        self._current_phase = CompliancePhase.DEPLOY
        self.logger.info("Phase 4: Deploying updated rules")

        # Record rule history
        self.state.rule_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'version': self.state.regulatory_version,
            'rules': self.state.current_rules,
            'changes': await self.ingestor.get_changes()
        })

        # A/B test if possible
        ab_test_result = await self.evolver.ab_test_rules(
            old_rules=self.state.rule_history[-2]['rules'] if len(self.state.rule_history) >= 2 else {},
            new_rules=self.state.current_rules
        )

        self.logger.info(f"A/B test result: {ab_test_result}")

        # Update version
        self.state.last_update_time = datetime.utcnow()

    async def _alert_phase(self):
        """Generate and manage alerts"""
        self._current_phase = CompliancePhase.ALERT
        self.logger.info("Phase 5: Checking for compliance violations")

        # Scan for violations
        violations = await self._scan_internal_systems()

        # Generate alerts
        for violation in violations:
            alert = await self.alerter.generate_alert(violation)
            if alert:
                self.state.active_alerts.append(alert)
                self.logger.warning(f"Alert generated: {alert['severity']} - {alert['message']}")

        # Escalate if needed
        await self.alerter.escalate_alerts(self.state.active_alerts)

        # Clean resolved alerts
        self.state.active_alerts = [
            alert for alert in self.state.active_alerts
            if not alert.get('resolved', False)
        ]

    async def _scan_internal_systems(self) -> List[Dict[str, Any]]:
        """
        Scan internal systems for compliance violations

        Returns:
            List of violations detected
        """
        # This is a placeholder - implementation depends on your systems
        # In production, this would:
        # 1. Query transaction logs
        # 2. Check against current rules
        # 3. Identify violations

        violations = []

        # Example: Check recent transactions
        # violations = await self._check_transactions_against_rules()

        return violations

    async def _update_metrics(self):
        """Update compliance metrics"""
        # Calculate metrics
        self.state.metrics.update({
            'total_rules': len(self.state.current_rules),
            'active_alerts': len(self.state.active_alerts),
            'edge_cases_found': len(self.state.edge_cases_found),
            'rule_versions': len(self.state.rule_history),
            'last_phase': self._current_phase.value,
            'uptime_hours': (datetime.utcnow() - self.state.last_scan_time).total_seconds() / 3600
            if self.state.last_scan_time else 0
        })

    async def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report

        Returns:
            Compliance report with metrics, alerts, and recommendations
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'regulatory_version': self.state.regulatory_version,
            'last_scan': self.state.last_scan_time.isoformat() if self.state.last_scan_time else None,
            'last_update': self.state.last_update_time.isoformat() if self.state.last_update_time else None,
            'metrics': self.state.metrics,
            'active_alerts': self.state.active_alerts,
            'edge_cases': self.state.edge_cases_found,
            'current_rules': self.state.current_rules,
            'recommendations': await self._generate_recommendations()
        }

    async def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        # Analyze alerts
        high_severity_alerts = [
            a for a in self.state.active_alerts
            if a.get('severity') in ['high', 'critical']
        ]
        if high_severity_alerts:
            recommendations.append(
                f"Address {len(high_severity_alerts)} high-severity alerts immediately"
            )

        # Analyze edge cases
        unaddressed_cases = [
            c for c in self.state.edge_cases_found
            if not c.get('addressed', False)
        ]
        if unaddressed_cases:
            recommendations.append(
                f"Review and address {len(unaddressed_cases)} unaddressed edge cases"
            )

        # Check rule age
        if self.state.last_update_time:
            days_since_update = (datetime.utcnow() - self.state.last_update_time).days
            if days_since_update > 30:
                recommendations.append(
                    f"Rules haven't been updated in {days_since_update} days - consider review"
                )

        return recommendations

    async def force_update(self, regulatory_changes: Optional[List[Dict[str, Any]]] = None):
        """
        Force an update cycle (for testing or manual trigger)

        Args:
            regulatory_changes: Optional specific changes to apply
        """
        self.logger.info("Forcing update cycle")

        if regulatory_changes:
            await self.ingestor.ingest_changes(regulatory_changes)

        await self._update_phase()
        await self._evolve_phase()
        await self._deploy_phase()
        await self._alert_phase()
        self._save_state()

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        return {
            'running': self._running,
            'current_phase': self._current_phase.value,
            'regulatory_version': self.state.regulatory_version,
            'last_scan': self.state.last_scan_time.isoformat() if self.state.last_scan_time else None,
            'last_update': self.state.last_update_time.isoformat() if self.state.last_update_time else None,
            'active_alerts': len(self.state.active_alerts),
            'metrics': self.state.metrics
        }


# Convenience function for quick start
async def run_compliance_monitor(
    checkpoint_dir: str = "./checkpoints/compliance",
    scan_interval_seconds: int = 3600,
    **kwargs
):
    """
    Run compliance monitor with default settings

    Args:
        checkpoint_dir: Checkpoint directory
        scan_interval_seconds: Seconds between scans
        **kwargs: Additional arguments for ComplianceMonitor

    Example:
        >>> await run_compliance_monitor(scan_interval_seconds=1800)
    """
    monitor = ComplianceMonitor(
        checkpoint_dir=checkpoint_dir,
        scan_interval_seconds=scan_interval_seconds,
        **kwargs
    )

    try:
        await monitor.start()
    except KeyboardInterrupt:
        await monitor.stop()
        return monitor.get_status()
