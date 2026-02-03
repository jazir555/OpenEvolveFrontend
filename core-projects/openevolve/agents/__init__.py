"""
OpenEvolve Agents Module

A collection of autonomous agents for complex decision-making tasks.

Available Agents:
- ComplianceMonitor: Continuous compliance monitoring
- InvestmentCommittee: Investment decision support

Modules:
- compliance: Compliance monitoring components
"""

__version__ = "0.1.0"

from .compliance_monitor import ComplianceMonitor
from .investment_committee import InvestmentCommitteeAgent

__all__ = [
    'ComplianceMonitor',
    'InvestmentCommitteeAgent'
]
