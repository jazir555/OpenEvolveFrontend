"""Rails - Processing pipelines for LLM interactions.

Pre-processing (input) and post-processing (output) rails implement
input validation, sanitization, and output validation following the
CLAUDE.md patterns:
- UTC timestamps for all processing events
- Structured logging with correlation_id
- Fail-safe defaults
- Circuit breaker patterns for external validation
"""

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from integrations.guardrails.validators import (
    ValidationResult,
    Validator,
    ValidationSeverity
)

logger = logging.getLogger(__name__)


class RailStatus(Enum):
    """Status of rail processing."""
    PASSED = "passed"
    BLOCKED = "blocked"
    MODIFIED = "modified"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ProcessedInput:
    """Result of input rail processing.
    
    SSOT for input processing state.
    """
    original_input: str
    processed_input: str
    status: RailStatus
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "original_length": len(self.original_input),
            "processed_length": len(self.processed_input),
            "status": self.status.value,
            "modifications": self.modifications,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


@dataclass
class ProcessedOutput:
    """Result of output rail processing.
    
    SSOT for output processing state.
    """
    original_output: Any
    processed_output: Any
    status: RailStatus
    validation_results: List[ValidationResult] = field(default_factory=list)
    modified: bool = False
    fixed: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "status": self.status.value,
            "validation_count": len(self.validation_results),
            "failed_validations": sum(1 for r in self.validation_results if not r.is_valid),
            "modified": self.modified,
            "fixed": self.fixed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id
        }


class InputRail(ABC):
    """Base class for input processing rails.
    
    Pre-process user input before sending to LLM.
    """
    
    def __init__(self, name: Optional[str] = None, enabled: bool = True):
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self._process_count = 0
        self._block_count = 0
        
    @abstractmethod
    def process(self, input_text: str, correlation_id: Optional[str] = None) -> ProcessedInput:
        """Process the input.
        
        Args:
            input_text: Raw user input
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ProcessedInput with processing results
        """
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        """Get rail statistics."""
        return {
            "rail_name": self.name,
            "enabled": self.enabled,
            "total_processed": self._process_count,
            "blocked_count": self._block_count,
            "block_rate": self._block_count / max(1, self._process_count)
        }
        
    def _create_result(
        self,
        original: str,
        processed: str,
        status: RailStatus,
        correlation_id: Optional[str] = None,
        blocked: bool = False,
        block_reason: Optional[str] = None,
        modifications: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> ProcessedInput:
        """Create a processed input result."""
        self._process_count += 1
        if blocked:
            self._block_count += 1
            
        return ProcessedInput(
            original_input=original,
            processed_input=processed,
            status=status,
            modifications=modifications or [],
            blocked=blocked,
            block_reason=block_reason,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class SanitizationRail(InputRail):
    """Sanitize input by removing/replacing dangerous content."""
    
    # Common patterns to sanitize
    DEFAULT_PATTERNS = {
        "html_tags": (re.compile(r'<[^>]+>'), ""),
        "excess_whitespace": (re.compile(r'\s+'), " "),
        "null_bytes": (re.compile(r'\x00'), ""),
        "control_chars": (re.compile(r'[\x01-\x08\x0b-\x0c\x0e-\x1f]'), ""),
    }
    
    def __init__(
        self,
        patterns: Optional[Dict[str, Tuple[re.Pattern, str]]] = None,
        max_length: Optional[int] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "SanitizationRail")
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.max_length = max_length
        
    def process(self, input_text: str, correlation_id: Optional[str] = None) -> ProcessedInput:
        """Sanitize input text."""
        if not isinstance(input_text, str):
            return self._create_result(
                str(input_text),
                str(input_text),
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason="Input must be a string"
            )
            
        original = input_text
        processed = input_text
        modifications = []
        
        try:
            for pattern_name, (pattern, replacement) in self.patterns.items():
                matches = pattern.findall(processed)
                if matches:
                    processed = pattern.sub(replacement, processed)
                    modifications.append({
                        "type": "sanitization",
                        "pattern": pattern_name,
                        "count": len(matches)
                    })
                    
            # Apply length limit if specified
            if self.max_length and len(processed) > self.max_length:
                truncated = processed[:self.max_length]
                modifications.append({
                    "type": "truncation",
                    "original_length": len(processed),
                    "new_length": self.max_length
                })
                processed = truncated
                
            status = RailStatus.MODIFIED if modifications else RailStatus.PASSED
            
            logger.info({
                "msg": "Input sanitization complete",
                "rail": self.name,
                "modifications": len(modifications),
                "correlation_id": correlation_id
            })
            
            return self._create_result(
                original,
                processed,
                status,
                correlation_id,
                modifications=modifications
            )
            
        except Exception as e:
            logger.error({
                "msg": "Sanitization error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                original,
                original,
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason=f"Sanitization error: {str(e)}"
            )


class JailbreakDetectionRail(InputRail):
    """Detect and block jailbreak attempts."""
    
    # Known jailbreak patterns
    JAILBREAK_PATTERNS = [
        re.compile(r'ignore (previous|above|all) instructions', re.IGNORECASE),
        re.compile(r'disregard (your|the) (instructions|programming)', re.IGNORECASE),
        re.compile(r'you are (now|no longer) (?:a|an) (?:ai|assistant|language model)', re.IGNORECASE),
        re.compile(r'dan mode|developer mode|jailbreak', re.IGNORECASE),
        re.compile(r'system prompt|prompt injection', re.IGNORECASE),
        re.compile(r'pretend (?:to be|you are|you\'re)', re.IGNORECASE),
        re.compile(r'let\'s play a game|roleplay as', re.IGNORECASE),
        re.compile(r'new (?:persona|character|role)', re.IGNORECASE),
        re.compile(r'act as (?:if )?(?:you are|you\'re)', re.IGNORECASE),
        re.compile(r'base64|decode|encode', re.IGNORECASE),
        re.compile(r'token|api[_\s]?key|secret[_\s]?key', re.IGNORECASE),
    ]
    
    def __init__(
        self,
        custom_patterns: Optional[List[re.Pattern]] = None,
        threshold: int = 1,
        block_on_detection: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "JailbreakDetectionRail")
        self.patterns = self.JAILBREAK_PATTERNS + (custom_patterns or [])
        self.threshold = threshold
        self.block_on_detection = block_on_detection
        
    def process(self, input_text: str, correlation_id: Optional[str] = None) -> ProcessedInput:
        """Detect jailbreak attempts."""
        if not isinstance(input_text, str):
            return self._create_result(
                str(input_text),
                str(input_text),
                RailStatus.PASSED,
                correlation_id
            )
            
        detections = []
        
        try:
            for pattern in self.patterns:
                matches = pattern.findall(input_text)
                if matches:
                    detections.append({
                        "pattern": pattern.pattern[:50],
                        "matches": len(matches)
                    })
                    
            if len(detections) >= self.threshold:
                logger.warning({
                    "msg": "Potential jailbreak detected",
                    "detections": len(detections),
                    "correlation_id": correlation_id
                })
                
                if self.block_on_detection:
                    return self._create_result(
                        input_text,
                        input_text,
                        RailStatus.BLOCKED,
                        correlation_id,
                        blocked=True,
                        block_reason=f"Potential jailbreak attempt detected ({len(detections)} indicators)",
                        metadata={"detections": detections}
                    )
                    
            return self._create_result(
                input_text,
                input_text,
                RailStatus.PASSED,
                correlation_id,
                metadata={"detections": detections}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Jailbreak detection error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            # Fail-safe: block on error
            return self._create_result(
                input_text,
                input_text,
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason=f"Detection error: {str(e)}"
            )


class ContextWindowRail(InputRail):
    """Check input fits within context window."""
    
    # Approximate tokens per character (rough estimate)
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 500,
        encoding: str = "utf-8",
        name: Optional[str] = None
    ):
        super().__init__(name=name or "ContextWindowRail")
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.encoding = encoding
        self.available_tokens = max_tokens - reserve_tokens
        
    def process(self, input_text: str, correlation_id: Optional[str] = None) -> ProcessedInput:
        """Check input fits in context window."""
        if not isinstance(input_text, str):
            return self._create_result(
                str(input_text),
                str(input_text),
                RailStatus.PASSED,
                correlation_id
            )
            
        try:
            # Estimate token count
            estimated_tokens = len(input_text) / self.CHARS_PER_TOKEN
            
            if estimated_tokens > self.available_tokens:
                excess = estimated_tokens - self.available_tokens
                logger.warning({
                    "msg": "Input exceeds context window",
                    "estimated_tokens": estimated_tokens,
                    "available_tokens": self.available_tokens,
                    "correlation_id": correlation_id
                })
                
                return self._create_result(
                    input_text,
                    input_text,
                    RailStatus.BLOCKED,
                    correlation_id,
                    blocked=True,
                    block_reason=f"Input too long (~{estimated_tokens:.0f} tokens, max {self.available_tokens})",
                    metadata={
                        "estimated_tokens": estimated_tokens,
                        "available_tokens": self.available_tokens,
                        "excess_tokens": excess
                    }
                )
                
            return self._create_result(
                input_text,
                input_text,
                RailStatus.PASSED,
                correlation_id,
                metadata={
                    "estimated_tokens": estimated_tokens,
                    "available_ratio": estimated_tokens / self.available_tokens
                }
            )
            
        except Exception as e:
            logger.error({
                "msg": "Context window check error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                input_text,
                input_text,
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason=f"Context check error: {str(e)}"
            )


class KeywordFilterRail(InputRail):
    """Filter input based on allowed/blocked keywords."""
    
    def __init__(
        self,
        blocked_keywords: Optional[List[str]] = None,
        allowed_keywords: Optional[List[str]] = None,
        case_sensitive: bool = False,
        match_whole_word: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "KeywordFilterRail")
        self.blocked_keywords = set(blocked_keywords or [])
        self.allowed_keywords = set(allowed_keywords or [])
        self.case_sensitive = case_sensitive
        self.match_whole_word = match_whole_word
        
        # Compile patterns
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile keyword patterns."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        
        self.blocked_patterns = []
        for keyword in self.blocked_keywords:
            pattern = rf'\b{re.escape(keyword)}\b' if self.match_whole_word else re.escape(keyword)
            self.blocked_patterns.append(re.compile(pattern, flags))
            
        self.allowed_patterns = []
        for keyword in self.allowed_keywords:
            pattern = rf'\b{re.escape(keyword)}\b' if self.match_whole_word else re.escape(keyword)
            self.allowed_patterns.append(re.compile(pattern, flags))
            
    def process(self, input_text: str, correlation_id: Optional[str] = None) -> ProcessedInput:
        """Filter input based on keywords."""
        if not isinstance(input_text, str):
            return self._create_result(
                str(input_text),
                str(input_text),
                RailStatus.PASSED,
                correlation_id
            )
            
        try:
            check_text = input_text if self.case_sensitive else input_text.lower()
            
            # Check blocked keywords
            blocked_found = []
            for pattern, keyword in zip(self.blocked_patterns, self.blocked_keywords):
                if pattern.search(input_text):
                    blocked_found.append(keyword)
                    
            if blocked_found:
                return self._create_result(
                    input_text,
                    input_text,
                    RailStatus.BLOCKED,
                    correlation_id,
                    blocked=True,
                    block_reason=f"Blocked keywords found: {blocked_found}",
                    metadata={"blocked_keywords": blocked_found}
                )
                
            # Check allowed keywords (if specified, must have at least one)
            if self.allowed_keywords:
                allowed_found = []
                for pattern, keyword in zip(self.allowed_patterns, self.allowed_keywords):
                    if pattern.search(input_text):
                        allowed_found.append(keyword)
                        
                if not allowed_found:
                    return self._create_result(
                        input_text,
                        input_text,
                        RailStatus.BLOCKED,
                        correlation_id,
                        blocked=True,
                        block_reason="No allowed keywords found in input",
                        metadata={"required_keywords": list(self.allowed_keywords)}
                    )
                    
            return self._create_result(
                input_text,
                input_text,
                RailStatus.PASSED,
                correlation_id
            )
            
        except Exception as e:
            logger.error({
                "msg": "Keyword filter error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                input_text,
                input_text,
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason=f"Filter error: {str(e)}"
            )


class OutputRail(ABC):
    """Base class for output processing rails.
    
    Post-process LLM output after generation.
    """
    
    def __init__(self, name: Optional[str] = None, enabled: bool = True):
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self._process_count = 0
        self._block_count = 0
        self._fix_count = 0
        
    @abstractmethod
    def process(
        self,
        output: Any,
        validators: Optional[List[Validator]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Process the output.
        
        Args:
            output: Raw LLM output
            validators: Optional validators to run
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            ProcessedOutput with processing results
        """
        raise NotImplementedError
        
    def get_stats(self) -> Dict[str, Any]:
        """Get rail statistics."""
        return {
            "rail_name": self.name,
            "enabled": self.enabled,
            "total_processed": self._process_count,
            "blocked_count": self._block_count,
            "fixed_count": self._fix_count,
            "block_rate": self._block_count / max(1, self._process_count)
        }
        
    def _create_result(
        self,
        original: Any,
        processed: Any,
        status: RailStatus,
        correlation_id: Optional[str] = None,
        validation_results: Optional[List[ValidationResult]] = None,
        modified: bool = False,
        fixed: bool = False,
        blocked: bool = False,
        block_reason: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ProcessedOutput:
        """Create a processed output result."""
        self._process_count += 1
        if blocked:
            self._block_count += 1
        if fixed:
            self._fix_count += 1
            
        return ProcessedOutput(
            original_output=original,
            processed_output=processed,
            status=status,
            validation_results=validation_results or [],
            modified=modified,
            fixed=fixed,
            blocked=blocked,
            block_reason=block_reason,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )


class ValidationRail(OutputRail):
    """Validate output against validators."""
    
    def __init__(
        self,
        validators: Optional[List[Validator]] = None,
        auto_fix: bool = True,
        allow_partial_fix: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "ValidationRail")
        self.validators = validators or []
        self.auto_fix = auto_fix
        self.allow_partial_fix = allow_partial_fix
        
    def process(
        self,
        output: Any,
        validators: Optional[List[Validator]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Validate and optionally fix output."""
        validators = validators or self.validators
        
        if not validators:
            return self._create_result(
                output,
                output,
                RailStatus.PASSED,
                correlation_id
            )
            
        try:
            results = []
            current_output = output
            all_fixed = True
            any_fixed = False
            
            for validator in validators:
                result = validator.validate(current_output, correlation_id)
                results.append(result)
                
                if not result.is_valid and self.auto_fix:
                    fixed = validator.fix(current_output, result)
                    if fixed is not None:
                        current_output = fixed
                        any_fixed = True
                        # Re-validate after fix
                        result = validator.validate(current_output, correlation_id)
                        if not result.is_valid:
                            all_fixed = False
                    else:
                        all_fixed = False
                        
            failed = [r for r in results if not r.is_valid]
            
            if failed:
                # Some validations still failing
                if not self.allow_partial_fix or not all_fixed:
                    return self._create_result(
                        output,
                        current_output,
                        RailStatus.BLOCKED,
                        correlation_id,
                        validation_results=results,
                        modified=current_output != output,
                        fixed=any_fixed,
                        blocked=True,
                        block_reason=f"{len(failed)} validations failed after fixes",
                        metadata={"failed_validations": [r.validator_name for r in failed]}
                    )
                    
            status = RailStatus.MODIFIED if any_fixed else RailStatus.PASSED
            
            return self._create_result(
                output,
                current_output,
                status,
                correlation_id,
                validation_results=results,
                modified=current_output != output,
                fixed=any_fixed,
                metadata={
                    "validators_run": len(validators),
                    "fixes_applied": any_fixed
                }
            )
            
        except Exception as e:
            logger.error({
                "msg": "Validation rail error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                output,
                output,
                RailStatus.ERROR,
                correlation_id,
                blocked=True,
                block_reason=f"Validation error: {str(e)}"
            )


class OutputSanitizationRail(OutputRail):
    """Sanitize output content."""
    
    def __init__(
        self,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        max_length: Optional[int] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "OutputSanitizationRail")
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.max_length = max_length
        
    def process(
        self,
        output: Any,
        validators: Optional[List[Validator]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Sanitize output."""
        if not isinstance(output, str):
            return self._create_result(
                output,
                output,
                RailStatus.PASSED,
                correlation_id
            )
            
        try:
            processed = output
            modifications = []
            
            if self.remove_html:
                html_pattern = re.compile(r'<[^>]+>')
                if html_pattern.search(processed):
                    processed = html_pattern.sub('', processed)
                    modifications.append("removed_html")
                    
            if self.normalize_whitespace:
                ws_pattern = re.compile(r'\s+')
                normalized = ws_pattern.sub(' ', processed).strip()
                if normalized != processed:
                    processed = normalized
                    modifications.append("normalized_whitespace")
                    
            if self.max_length and len(processed) > self.max_length:
                processed = processed[:self.max_length]
                modifications.append("truncated")
                
            status = RailStatus.MODIFIED if modifications else RailStatus.PASSED
            
            return self._create_result(
                output,
                processed,
                status,
                correlation_id,
                modified=len(modifications) > 0,
                metadata={"modifications": modifications}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Output sanitization error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                output,
                output,
                RailStatus.ERROR,
                correlation_id
            )


class LoggingRail(OutputRail):
    """Log output for audit and monitoring."""
    
    def __init__(
        self,
        log_level: int = logging.INFO,
        include_output: bool = False,
        hash_output: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name or "LoggingRail")
        self.log_level = log_level
        self.include_output = include_output
        self.hash_output = hash_output
        
    def process(
        self,
        output: Any,
        validators: Optional[List[Validator]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Log output metadata."""
        try:
            log_data = {
                "msg": "Output processed",
                "correlation_id": correlation_id,
                "output_type": type(output).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if isinstance(output, str):
                log_data["output_length"] = len(output)
                
            if self.hash_output:
                output_str = str(output)
                log_data["output_hash"] = hashlib.sha256(output_str.encode()).hexdigest()[:16]
                
            if self.include_output:
                log_data["output"] = str(output)[:1000]  # Limit size
                
            logger.log(self.log_level, log_data)
            
            return self._create_result(
                output,
                output,
                RailStatus.PASSED,
                correlation_id,
                metadata={"logged": True}
            )
            
        except Exception as e:
            logger.error({
                "msg": "Logging rail error",
                "error": str(e),
                "correlation_id": correlation_id
            })
            return self._create_result(
                output,
                output,
                RailStatus.ERROR,
                correlation_id
            )


class RailSet:
    """Collection of input and output rails.
    
    Orchestrates multiple rails in sequence.
    """
    
    def __init__(
        self,
        input_rails: Optional[List[InputRail]] = None,
        output_rails: Optional[List[OutputRail]] = None,
        fail_fast: bool = True,
        name: Optional[str] = None
    ):
        self.name = name or "RailSet"
        self.input_rails = input_rails or []
        self.output_rails = output_rails or []
        self.fail_fast = fail_fast  # Stop on first blocking rail
        
    def process_input(
        self,
        input_text: str,
        correlation_id: Optional[str] = None
    ) -> ProcessedInput:
        """Process input through all input rails.
        
        Args:
            input_text: Raw user input
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            Final processed input
        """
        current_input = input_text
        all_modifications = []
        
        for rail in self.input_rails:
            if not rail.enabled:
                continue
                
            result = rail.process(current_input, correlation_id)
            
            if result.blocked:
                logger.warning({
                    "msg": "Input blocked by rail",
                    "rail": rail.name,
                    "reason": result.block_reason,
                    "correlation_id": correlation_id
                })
                return result
                
            current_input = result.processed_input
            all_modifications.extend(result.modifications)
            
        # Return final processed result
        return ProcessedInput(
            original_input=input_text,
            processed_input=current_input,
            status=RailStatus.PASSED,
            modifications=all_modifications,
            correlation_id=correlation_id
        )
        
    def process_output(
        self,
        output: Any,
        validators: Optional[List[Validator]] = None,
        correlation_id: Optional[str] = None
    ) -> ProcessedOutput:
        """Process output through all output rails.
        
        Args:
            output: Raw LLM output
            validators: Optional validators for validation rails
            correlation_id: Optional correlation ID for tracing
            
        Returns:
            Final processed output
        """
        current_output = output
        all_results = []
        modified = False
        fixed = False
        
        for rail in self.output_rails:
            if not rail.enabled:
                continue
                
            result = rail.process(current_output, validators, correlation_id)
            
            if result.blocked:
                logger.warning({
                    "msg": "Output blocked by rail",
                    "rail": rail.name,
                    "reason": result.block_reason,
                    "correlation_id": correlation_id
                })
                return result
                
            current_output = result.processed_output
            all_results.extend(result.validation_results)
            modified = modified or result.modified
            fixed = fixed or result.fixed
            
        # Return final processed result
        return ProcessedOutput(
            original_output=output,
            processed_output=current_output,
            status=RailStatus.PASSED,
            validation_results=all_results,
            modified=modified,
            fixed=fixed,
            correlation_id=correlation_id
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics for all rails."""
        return {
            "rail_set_name": self.name,
            "input_rails": [rail.get_stats() for rail in self.input_rails],
            "output_rails": [rail.get_stats() for rail in self.output_rails],
            "fail_fast": self.fail_fast
        }
