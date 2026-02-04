"""
Global Context Manager - Powered by Matryoshka

This module provides a global singleton to manage context for any LLM use case,
solving the "context rot" issue by recursively distilling information.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from glue.adapters.matryoshka_adapter import StatefulMatryoshkaClient

logger = logging.getLogger(__name__)

class GlobalContextManager:
    """
    Singleton manager for global context distillation and rotation.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalContextManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        self.client = StatefulMatryoshkaClient()
        self.context_threshold_chars = 20000 # ~5k tokens
        self.enabled = os.getenv("MATRYOSHKA_ENABLED", "true").lower() == "true"
        self._initialized = True
        logger.info(f"GlobalContextManager initialized. Matryoshka enabled: {self.enabled}")

    def manage(self, 
               session_id: str, 
               messages: List[Dict[str, str]], 
               force_compress: bool = False) -> List[Dict[str, str]]:
        """
        Manage context for a given session.
        If context is too large, it uses Matryoshka to distill it.
        
        Args:
            session_id: Unique ID for the context session.
            messages: List of chat messages.
            force_compress: Whether to force distillation regardless of size.
            
        Returns:
            A optimized list of messages (potentially with distilled context).
        """
        if not self.enabled or not self.client.is_available():
            if not self.enabled:
                logger.debug("Matryoshka is disabled via MATRYOSHKA_ENABLED.")
            else:
                logger.warning("Matryoshka not available for global context management.")
            return messages

        total_chars = sum(len(m['content']) for m in messages)
        
        if total_chars > self.context_threshold_chars or force_compress:
            logger.info(f"Context rot detected for session {session_id} ({total_chars} chars). Distilling...")
            
            # Distill the middle part of the history, keeping the system prompt and the latest messages intact
            system_msg = messages[0] if messages and messages[0]['role'] == 'system' else None
            latest_msgs = messages[-2:] if len(messages) > 2 else []
            history_to_distill = messages[1:-2] if system_msg else messages[:-2]
            
            if not history_to_distill:
                return messages

            distilled_summary = self.client.distill_history(session_id, history_to_distill)
            
            new_messages = []
            if system_msg:
                new_messages.append(system_msg)
            
            new_messages.append({
                "role": "system",
                "content": f"PREVIOUS_CONTEXT_SUMMARY (Distilled via Matryoshka):\n{distilled_summary}"
            })
            
            new_messages.extend(latest_msgs)
            
            logger.info(f"Context distilled for session {session_id}. New size: {sum(len(m['content']) for m in new_messages)} chars.")
            return new_messages
            
        return messages

def get_global_context_manager() -> GlobalContextManager:
    return GlobalContextManager()
