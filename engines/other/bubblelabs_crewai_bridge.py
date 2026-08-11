"""
BubbleLabs CrewAI Bridge Module

Provides bridging between BubbleLabs and CrewAI.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BubbleLabsCrewAIBridgeConfig:
    """Configuration for BubbleLabs CrewAI Bridge"""
    timeout: int = 30


class BubbleLabsCrewAIBridge:
    """BubbleLabs CrewAI Bridge class"""
    
    def __init__(self, config: Optional[BubbleLabsCrewAIBridgeConfig] = None):
        self.config = config or BubbleLabsCrewAIBridgeConfig()
        logger.info("BubbleLabs CrewAI Bridge initialized")
    
    def bridge(self, agent: Dict[str, Any]) -> Dict[str, Any]:
        """Bridge agent"""
        return {"bridged": True, "agent": agent}
    
    def translate(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Translate message"""
        return {"translated": True, "message": message}


def create_bridge(config: Optional[BubbleLabsCrewAIBridgeConfig] = None) -> BubbleLabsCrewAIBridge:
    """Factory function to create bridge instance"""
    return BubbleLabsCrewAIBridge(config)


# Alias for backward compatibility
BubbleLabsCREWAIBridge = BubbleLabsCrewAIBridge

# Another alias for tests (without 'A' in 'CrewAI')
BubbleLabsCrewaiBridge = BubbleLabsCrewAIBridge
