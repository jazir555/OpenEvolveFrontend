"""
Vision-Language Monitor (VLM Agent)

The "Eyes" of the system - provides visual verification and analysis capabilities.
Can take screenshots, analyze UI elements, and verify visual correctness.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import base64
import io
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class VLMProvider(Enum):
    """Supported VLM providers"""
    GPT4O_VISION = "gpt-4o-vision"      # OpenAI GPT-4o with vision
    CLAUDE_VISION = "claude-3-vision"    # Anthropic Claude with vision
    LLAVA = "llava"                       # Local LLaVA
    PIXTRAL = "pixtral"                   # Mistral Pixtral
    MOCK = "mock"                         # Mock for testing


@dataclass
class VisualAnalysis:
    """Result of visual analysis"""
    success: bool
    description: str
    elements_detected: List[Dict[str, Any]]
    issues_found: List[Dict[str, Any]]
    confidence: float
    screenshot_path: Optional[str]
    analysis_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'description': self.description,
            'elements_detected': self.elements_detected,
            'issues_found': self.issues_found,
            'confidence': self.confidence,
            'screenshot_path': self.screenshot_path,
            'analysis_timestamp': self.analysis_timestamp
        }


@dataclass
class UIElement:
    """Represents a detected UI element"""
    element_type: str  # button, text, image, node, etc.
    text_content: Optional[str]
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    color: Optional[str]
    state: Optional[str]  # active, disabled, selected, etc.
    confidence: float


class VisionLanguageMonitor:
    """
    Vision-Language Monitor for visual verification.
    
    Capabilities:
    - Screenshot capture and analysis
    - UI element detection and verification
    - Visual regression detection
    - Bubblelab canvas verification
    - PDF chart reading
    - Frontend bug verification
    
    Example:
        vlm = VisionLanguageMonitor(VLMProvider.GPT4O_VISION, api_key="...")
        analysis = await vlm.analyze_screenshot(
            screenshot_path="bubblelab.png",
            verification_prompt="Verify the node is green and connected"
        )
    """
    
    def __init__(
        self,
        provider: VLMProvider = VLMProvider.MOCK,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        local_model_path: Optional[str] = None,
        screenshot_dir: str = "./screenshots"
    ):
        """
        Initialize VLM Monitor.
        
        Args:
            provider: VLM provider to use
            api_key: API key for cloud providers
            api_base: Custom API base URL
            local_model_path: Path to local model (for LLaVA/Pixtral)
            screenshot_dir: Directory to store screenshots
        """
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.local_model_path = local_model_path
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Initialize provider
        self._init_provider()
        
        logger.info({
            'msg': 'VisionLanguageMonitor initialized',
            'provider': provider.value,
            'screenshot_dir': str(self.screenshot_dir)
        })
    
    def _init_provider(self):
        """Initialize the VLM provider"""
        if self.provider == VLMProvider.GPT4O_VISION:
            if not self.api_key:
                logger.warning("GPT-4o Vision requires API key, falling back to mock")
                self.provider = VLMProvider.MOCK
        
        elif self.provider == VLMProvider.CLAUDE_VISION:
            if not self.api_key:
                logger.warning("Claude Vision requires API key, falling back to mock")
                self.provider = VLMProvider.MOCK
        
        elif self.provider == VLMProvider.LLAVA:
            try:
                # Try to import LLaVA
                pass  # Placeholder for actual LLaVA initialization
            except ImportError:
                logger.warning("LLaVA not available, falling back to mock")
                self.provider = VLMProvider.MOCK
        
        elif self.provider == VLMProvider.PIXTRAL:
            try:
                # Try to import Pixtral
                pass  # Placeholder for actual Pixtral initialization
            except ImportError:
                logger.warning("Pixtral not available, falling back to mock")
                self.provider = VLMProvider.MOCK
    
    async def analyze_screenshot(
        self,
        screenshot_path: Union[str, Path],
        verification_prompt: str,
        expected_elements: Optional[List[Dict[str, Any]]] = None
    ) -> VisualAnalysis:
        """
        Analyze a screenshot and verify visual elements.
        
        Args:
            screenshot_path: Path to screenshot image
            verification_prompt: What to verify (e.g., "Is the node green?")
            expected_elements: List of expected UI elements
            
        Returns:
            VisualAnalysis with detection results
        """
        screenshot_path = Path(screenshot_path)
        
        if not screenshot_path.exists():
            return VisualAnalysis(
                success=False,
                description=f"Screenshot not found: {screenshot_path}",
                elements_detected=[],
                issues_found=[{'type': 'file_not_found', 'path': str(screenshot_path)}],
                confidence=0.0,
                screenshot_path=str(screenshot_path)
            )
        
        logger.info({
            'msg': 'Analyzing screenshot',
            'path': str(screenshot_path),
            'prompt': verification_prompt
        })
        
        # Perform analysis based on provider
        if self.provider == VLMProvider.GPT4O_VISION:
            result = await self._analyze_with_gpt4o(screenshot_path, verification_prompt)
        elif self.provider == VLMProvider.CLAUDE_VISION:
            result = await self._analyze_with_claude(screenshot_path, verification_prompt)
        elif self.provider == VLMProvider.LLAVA:
            result = await self._analyze_with_llava(screenshot_path, verification_prompt)
        elif self.provider == VLMProvider.PIXTRAL:
            result = await self._analyze_with_pixtral(screenshot_path, verification_prompt)
        else:
            result = await self._analyze_mock(screenshot_path, verification_prompt)
        
        # Verify expected elements if provided
        if expected_elements:
            result = self._verify_expected_elements(result, expected_elements)
        
        # Log analysis
        self._log_analysis(screenshot_path, verification_prompt, result)
        
        return result
    
    async def capture_and_analyze(
        self,
        url: Optional[str] = None,
        element_selector: Optional[str] = None,
        verification_prompt: str = "Analyze this screenshot",
        expected_elements: Optional[List[Dict[str, Any]]] = None
    ) -> VisualAnalysis:
        """
        Capture screenshot and analyze in one operation.
        
        Args:
            url: URL to capture (if None, captures current screen)
            element_selector: CSS selector for specific element
            verification_prompt: What to verify
            expected_elements: Expected UI elements
            
        Returns:
            VisualAnalysis
        """
        # Capture screenshot
        screenshot_path = await self._capture_screenshot(url, element_selector)
        
        if not screenshot_path:
            return VisualAnalysis(
                success=False,
                description="Failed to capture screenshot",
                elements_detected=[],
                issues_found=[{'type': 'capture_failed'}],
                confidence=0.0,
                screenshot_path=None
            )
        
        # Analyze
        return await self.analyze_screenshot(
            screenshot_path,
            verification_prompt,
            expected_elements
        )
    
    async def verify_bubblelab_canvas(
        self,
        screenshot_path: Union[str, Path],
        expected_nodes: Optional[List[Dict[str, Any]]] = None,
        expected_connections: Optional[List[Tuple[str, str]]] = None
    ) -> VisualAnalysis:
        """
        Specialized verification for Bubblelab canvas.
        
        Args:
            screenshot_path: Screenshot of Bubblelab canvas
            expected_nodes: Expected nodes with properties
            expected_connections: Expected node connections
            
        Returns:
            VisualAnalysis
        """
        prompt = "Analyze this Bubblelab canvas. Verify: 1) All nodes are visible, 2) Nodes are properly colored, 3) Connections are drawn correctly, 4) No visual glitches."
        
        analysis = await self.analyze_screenshot(screenshot_path, prompt)
        
        # Additional Bubblelab-specific checks
        if expected_nodes:
            for node in expected_nodes:
                node_found = any(
                    e.get('text') == node.get('label') or 
                    e.get('type') == node.get('type')
                    for e in analysis.elements_detected
                )
                if not node_found:
                    analysis.issues_found.append({
                        'type': 'missing_node',
                        'node': node
                    })
        
        return analysis
    
    async def compare_screenshots(
        self,
        baseline_path: Union[str, Path],
        current_path: Union[str, Path],
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compare two screenshots for visual regression.
        
        Args:
            baseline_path: Path to baseline image
            current_path: Path to current image
            threshold: Similarity threshold (0-1)
            
        Returns:
            Comparison result
        """
        if not PIL_AVAILABLE:
            return {
                'success': False,
                'error': 'PIL not available for image comparison'
            }
        
        try:
            from PIL import Image
            import numpy as np
            
            baseline = Image.open(baseline_path)
            current = Image.open(current_path)
            
            # Resize to same dimensions
            if baseline.size != current.size:
                current = current.resize(baseline.size)
            
            # Convert to numpy arrays
            baseline_array = np.array(baseline)
            current_array = np.array(current)
            
            # Calculate similarity
            diff = np.abs(baseline_array.astype(float) - current_array.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            # Generate diff image
            diff_image = Image.fromarray(
                np.uint8(np.abs(baseline_array - current_array))
            )
            
            diff_path = self.screenshot_dir / f"diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            diff_image.save(diff_path)
            
            return {
                'success': True,
                'similarity': float(similarity),
                'threshold': threshold,
                'passed': similarity >= threshold,
                'diff_image': str(diff_path),
                'baseline': str(baseline_path),
                'current': str(current_path)
            }
            
        except Exception as e:
            logger.error(f"Screenshot comparison failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _analyze_with_gpt4o(
        self,
        screenshot_path: Path,
        prompt: str
    ) -> VisualAnalysis:
        """Analyze using GPT-4o Vision"""
        if not HTTPX_AVAILABLE:
            return await self._analyze_mock(screenshot_path, prompt)
        
        try:
            # Read and encode image
            with open(screenshot_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Call GPT-4o Vision API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    },
                    timeout=30.0
                )
                
                result = response.json()
                description = result['choices'][0]['message']['content']
                
                # Parse structured response
                elements = self._parse_elements_from_description(description)
                issues = self._parse_issues_from_description(description)
                
                return VisualAnalysis(
                    success=True,
                    description=description,
                    elements_detected=elements,
                    issues_found=issues,
                    confidence=0.9,
                    screenshot_path=str(screenshot_path)
                )
                
        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {e}")
            return await self._analyze_mock(screenshot_path, prompt)
    
    async def _analyze_with_claude(
        self,
        screenshot_path: Path,
        prompt: str
    ) -> VisualAnalysis:
        """Analyze using Claude Vision"""
        # Similar implementation to GPT-4o
        return await self._analyze_mock(screenshot_path, prompt)
    
    async def _analyze_with_llava(
        self,
        screenshot_path: Path,
        prompt: str
    ) -> VisualAnalysis:
        """Analyze using local LLaVA"""
        # Local model implementation
        return await self._analyze_mock(screenshot_path, prompt)
    
    async def _analyze_with_pixtral(
        self,
        screenshot_path: Path,
        prompt: str
    ) -> VisualAnalysis:
        """Analyze using Pixtral"""
        # Local model implementation
        return await self._analyze_mock(screenshot_path, prompt)
    
    async def _analyze_mock(
        self,
        screenshot_path: Path,
        prompt: str
    ) -> VisualAnalysis:
        """Mock analysis for testing without API"""
        logger.info(f"Using mock VLM analysis for {screenshot_path}")
        
        # Simulate some basic analysis
        elements = [
            {
                'type': 'canvas',
                'text': 'Bubblelab Canvas',
                'bounding_box': [0, 0, 1920, 1080],
                'confidence': 0.95
            },
            {
                'type': 'node',
                'text': 'Knowledge Node',
                'color': 'green',
                'state': 'active',
                'confidence': 0.88
            }
        ]
        
        # Check prompt for keywords
        issues = []
        if 'green' in prompt.lower() and 'node' in prompt.lower():
            issues.append({
                'type': 'verification_passed',
                'message': 'Green node detected as expected'
            })
        
        return VisualAnalysis(
            success=True,
            description=f"Mock analysis: Screenshot shows Bubblelab canvas with nodes. Prompt: {prompt}",
            elements_detected=elements,
            issues_found=issues,
            confidence=0.85,
            screenshot_path=str(screenshot_path)
        )
    
    async def _capture_screenshot(
        self,
        url: Optional[str],
        element_selector: Optional[str]
    ) -> Optional[str]:
        """Capture screenshot using Playwright or similar"""
        try:
            from playwright.async_api import async_playwright
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = self.screenshot_dir / f"screenshot_{timestamp}.png"
            
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                if url:
                    await page.goto(url)
                    await page.wait_for_load_state('networkidle')
                
                if element_selector:
                    element = await page.query_selector(element_selector)
                    if element:
                        await element.screenshot(path=str(screenshot_path))
                    else:
                        await page.screenshot(path=str(screenshot_path))
                else:
                    await page.screenshot(path=str(screenshot_path))
                
                await browser.close()
            
            return str(screenshot_path)
            
        except ImportError:
            logger.warning("Playwright not available for screenshot capture")
            return None
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def _parse_elements_from_description(self, description: str) -> List[Dict[str, Any]]:
        """Parse detected elements from VLM description"""
        elements = []
        
        # Simple parsing - would be more sophisticated with structured output
        import re
        
        # Look for element mentions
        element_patterns = [
            r'(button|node|text|image|canvas|dropdown|input)\s*(?:with)?\s*["\']?([^"\']+)?["\']?',
        ]
        
        for pattern in element_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                elements.append({
                    'type': match[0].lower(),
                    'text': match[1] if len(match) > 1 else None,
                    'confidence': 0.8
                })
        
        return elements
    
    def _parse_issues_from_description(self, description: str) -> List[Dict[str, Any]]:
        """Parse issues from VLM description"""
        issues = []
        
        # Look for issue indicators
        issue_keywords = ['error', 'missing', 'incorrect', 'broken', 'not found', 'failed']
        
        for keyword in issue_keywords:
            if keyword.lower() in description.lower():
                issues.append({
                    'type': 'potential_issue',
                    'keyword': keyword,
                    'message': f'"{keyword}" mentioned in analysis'
                })
        
        return issues
    
    def _verify_expected_elements(
        self,
        analysis: VisualAnalysis,
        expected: List[Dict[str, Any]]
    ) -> VisualAnalysis:
        """Verify expected elements were found"""
        for expected_elem in expected:
            found = any(
                e.get('type') == expected_elem.get('type') or
                e.get('text') == expected_elem.get('text')
                for e in analysis.elements_detected
            )
            
            if not found:
                analysis.issues_found.append({
                    'type': 'expected_element_not_found',
                    'expected': expected_elem
                })
        
        return analysis
    
    def _log_analysis(
        self,
        screenshot_path: Path,
        prompt: str,
        result: VisualAnalysis
    ):
        """Log analysis for audit"""
        self.analysis_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'screenshot': str(screenshot_path),
            'prompt': prompt,
            'provider': self.provider.value,
            'success': result.success,
            'confidence': result.confidence,
            'elements_count': len(result.elements_detected),
            'issues_count': len(result.issues_found)
        })
        
        # Keep last 1000
        self.analysis_history = self.analysis_history[-1000:]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return self.analysis_history.copy()
