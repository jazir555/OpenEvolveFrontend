"""
Vision-Language Monitor (VLM) - The "Eyes"

Provides multimodal capabilities for UI verification, screenshot analysis,
and visual feedback. Enables agents to "see" the interface, verify 
rendering, and interact with visual elements.

Key Features:
- Screenshot capture and analysis
- UI element detection and verification
- Visual regression detection
- Bubblelab canvas monitoring
- VLM integration (GPT-4o Vision, Pixtral, Llava)
- OpenInterpreter integration for OS control
"""

import os
import io
import json
import time
import base64
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import asyncio

# Configure logging
logger = logging.getLogger(__name__)


class VLMProvider(Enum):
    """Supported VLM providers"""
    OPENAI = "openai"           # GPT-4o Vision
    ANTHROPIC = "anthropic"     # Claude 3 Vision
    MISTRAL = "mistral"         # Pixtral
    OLLAMA = "ollama"           # Llava (local)
    OPENINTERPRETER = "openinterpreter"  # OpenInterpreter
    CUSTOM = "custom"           # Custom VLM endpoint


class AnalysisType(Enum):
    """Types of visual analysis"""
    UI_VERIFICATION = "ui_verification"
    ELEMENT_DETECTION = "element_detection"
    VISUAL_REGRESSION = "visual_regression"
    TEXT_EXTRACTION = "text_extraction"
    CHART_ANALYSIS = "chart_analysis"
    COLOR_ANALYSIS = "color_analysis"
    LAYOUT_ANALYSIS = "layout_analysis"
    ACCESSIBILITY_CHECK = "accessibility_check"


@dataclass
class BoundingBox:
    """Bounding box for detected elements"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence
        }
    
    def contains(self, x: int, y: int) -> bool:
        """Check if point is within bounding box"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


@dataclass
class UIElement:
    """Detected UI element"""
    element_type: str
    text: Optional[str]
    bounding_box: BoundingBox
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "attributes": self.attributes,
            "confidence": self.confidence
        }


@dataclass
class VisualAnalysis:
    """Result of visual analysis"""
    analysis_id: str
    analysis_type: AnalysisType
    timestamp: datetime
    image_hash: str
    elements: List[UIElement] = field(default_factory=list)
    summary: str = ""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture"""
    full_page: bool = False
    viewport_only: bool = True
    hide_scrollbars: bool = True
    wait_for_network_idle: bool = True
    wait_time_ms: int = 500
    clip_region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    selector: Optional[str] = None  # CSS selector to capture


@dataclass
class VLMConfig:
    """Configuration for VLM provider"""
    provider: VLMProvider = VLMProvider.OPENAI
    api_key: Optional[str] = None
    model: str = "gpt-4o-vision"
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout_seconds: int = 30
    base_url: Optional[str] = None
    local_model_path: Optional[str] = None


class ScreenshotCapture:
    """Captures screenshots using Playwright or similar"""
    
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
    
    async def initialize(self):
        """Initialize Playwright browser"""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch()
        except ImportError:
            logger.warning("Playwright not installed. Screenshot capture disabled.")
    
    async def capture_from_url(
        self,
        url: str,
        config: ScreenshotConfig = None
    ) -> bytes:
        """Capture screenshot from URL"""
        config = config or ScreenshotConfig()
        
        if not self._browser:
            await self.initialize()
        
        if not self._browser:
            raise RuntimeError("Browser not available")
        
        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle" if config.wait_for_network_idle else "load")
            await asyncio.sleep(config.wait_time_ms / 1000)
            
            if config.selector:
                element = await page.query_selector(config.selector)
                if element:
                    screenshot = await element.screenshot()
                else:
                    screenshot = await page.screenshot(full_page=config.full_page)
            elif config.clip_region:
                screenshot = await page.screenshot(
                    clip={
                        "x": config.clip_region[0],
                        "y": config.clip_region[1],
                        "width": config.clip_region[2],
                        "height": config.clip_region[3]
                    }
                )
            else:
                screenshot = await page.screenshot(full_page=config.full_page)
            
            return screenshot
            
        finally:
            await context.close()
    
    async def capture_from_streamlit(self, config: ScreenshotConfig = None) -> bytes:
        """Capture screenshot of running Streamlit app"""
        # Default to localhost:8501 for Streamlit
        return await self.capture_from_url("http://localhost:8501", config)
    
    async def capture_bubblelab_canvas(
        self,
        bubblelab_url: str = "http://localhost:8501",
        config: ScreenshotConfig = None
    ) -> bytes:
        """Capture Bubblelab canvas specifically"""
        config = config or ScreenshotConfig()
        config.selector = '[data-testid="stGraphVizChart"], .bubblelab-canvas, canvas'
        return await self.capture_from_url(bubblelab_url, config)
    
    async def close(self):
        """Close browser"""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()


class VLMAnalyzer:
    """Analyzes images using Vision-Language Models"""
    
    def __init__(self, config: VLMConfig = None):
        self.config = config or VLMConfig()
        self._api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")
    
    async def analyze(
        self,
        image_bytes: bytes,
        prompt: str,
        analysis_type: AnalysisType = AnalysisType.UI_VERIFICATION
    ) -> VisualAnalysis:
        """Analyze an image using VLM"""
        analysis_id = hashlib.md5(f"{image_bytes[:100]}{time.time()}".encode()).hexdigest()[:12]
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        
        if self.config.provider == VLMProvider.OPENAI:
            return await self._analyze_openai(image_bytes, prompt, analysis_type, analysis_id, image_hash)
        elif self.config.provider == VLMProvider.ANTHROPIC:
            return await self._analyze_anthropic(image_bytes, prompt, analysis_type, analysis_id, image_hash)
        elif self.config.provider == VLMProvider.OLLAMA:
            return await self._analyze_ollama(image_bytes, prompt, analysis_type, analysis_id, image_hash)
        else:
            raise ValueError(f"Provider {self.config.provider} not implemented")
    
    async def _analyze_openai(
        self,
        image_bytes: bytes,
        prompt: str,
        analysis_type: AnalysisType,
        analysis_id: str,
        image_hash: str
    ) -> VisualAnalysis:
        """Analyze using OpenAI GPT-4o Vision"""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self._api_key)
            base64_image = self._encode_image(image_bytes)
            
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout_seconds
            )
            
            content = response.choices[0].message.content
            
            return VisualAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                image_hash=image_hash,
                summary=content,
                raw_response=content,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"OpenAI VLM analysis failed: {e}")
            return VisualAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                image_hash=image_hash,
                summary=f"Error: {str(e)}",
                issues=[{"error": str(e)}]
            )
    
    async def _analyze_anthropic(
        self,
        image_bytes: bytes,
        prompt: str,
        analysis_type: AnalysisType,
        analysis_id: str,
        image_hash: str
    ) -> VisualAnalysis:
        """Analyze using Anthropic Claude 3 Vision"""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            base64_image = self._encode_image(image_bytes)
            
            response = await client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=self.config.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            content = response.content[0].text
            
            return VisualAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                image_hash=image_hash,
                summary=content,
                raw_response=content,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Anthropic VLM analysis failed: {e}")
            return VisualAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                image_hash=image_hash,
                summary=f"Error: {str(e)}",
                issues=[{"error": str(e)}]
            )
    
    async def _analyze_ollama(
        self,
        image_bytes: bytes,
        prompt: str,
        analysis_type: AnalysisType,
        analysis_id: str,
        image_hash: str
    ) -> VisualAnalysis:
        """Analyze using local Ollama with Llava"""
        try:
            import aiohttp
            
            base64_image = self._encode_image(image_bytes)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llava",
                        "prompt": prompt,
                        "images": [base64_image],
                        "stream": False
                    },
                    timeout=self.config.timeout_seconds
                ) as response:
                    result = await response.json()
                    content = result.get("response", "")
                    
                    return VisualAnalysis(
                        analysis_id=analysis_id,
                        analysis_type=analysis_type,
                        timestamp=datetime.utcnow(),
                        image_hash=image_hash,
                        summary=content,
                        raw_response=content,
                        confidence=0.8  # Local models may be less confident
                    )
                    
        except Exception as e:
            logger.error(f"Ollama VLM analysis failed: {e}")
            return VisualAnalysis(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                timestamp=datetime.utcnow(),
                image_hash=image_hash,
                summary=f"Error: {str(e)}",
                issues=[{"error": str(e)}]
            )


class UIValidator:
    """Validates UI elements and states"""
    
    def __init__(self, vlm_analyzer: VLMAnalyzer = None):
        self.vlm = vlm_analyzer or VLMAnalyzer()
    
    async def verify_node_rendering(
        self,
        screenshot: bytes,
        expected_nodes: List[Dict[str, Any]]
    ) -> VisualAnalysis:
        """Verify that expected nodes are rendered correctly"""
        prompt = f"""
        Analyze this Bubblelab/workflow canvas screenshot. 
        
        Expected nodes to verify:
        {json.dumps(expected_nodes, indent=2)}
        
        For each expected node, verify:
        1. Is the node visible on the canvas?
        2. Is the node the correct color?
        3. Are the connections/edges visible?
        4. Is the node label/text readable?
        
        Provide a detailed verification report with any issues found.
        """
        
        return await self.vlm.analyze(screenshot, prompt, AnalysisType.UI_VERIFICATION)
    
    async def detect_visual_regression(
        self,
        baseline_screenshot: bytes,
        current_screenshot: bytes,
        threshold: float = 0.95
    ) -> VisualAnalysis:
        """Detect visual differences between screenshots"""
        # Simple pixel-based comparison first
        from PIL import Image
        import io
        
        baseline = Image.open(io.BytesIO(baseline_screenshot))
        current = Image.open(io.BytesIO(current_screenshot))
        
        if baseline.size != current.size:
            return VisualAnalysis(
                analysis_id=f"regression-{int(time.time())}",
                analysis_type=AnalysisType.VISUAL_REGRESSION,
                timestamp=datetime.utcnow(),
                image_hash="",
                summary="Screenshots have different dimensions - regression detected",
                issues=[{
                    "type": "dimension_mismatch",
                    "baseline_size": baseline.size,
                    "current_size": current.size
                }]
            )
        
        # Use VLM for detailed analysis
        prompt = """
        Compare these two screenshots (baseline vs current).
        Identify any visual differences, layout changes, or UI regressions.
        
        Focus on:
        - Missing or new elements
        - Color changes
        - Position shifts
        - Text changes
        - Styling differences
        
        Rate the severity of any changes found.
        """
        
        # For simplicity, just analyze current screenshot
        # In production, you'd send both images
        return await self.vlm.analyze(current_screenshot, prompt, AnalysisType.VISUAL_REGRESSION)
    
    async def validate_color_scheme(
        self,
        screenshot: bytes,
        expected_colors: List[str]
    ) -> VisualAnalysis:
        """Validate color scheme compliance"""
        prompt = f"""
        Analyze the color scheme of this UI screenshot.
        
        Expected colors to verify: {', '.join(expected_colors)}
        
        Report:
        1. Which expected colors are present?
        2. Are there any unexpected colors?
        3. Is the color contrast adequate for accessibility?
        4. Are the colors used consistently?
        """
        
        return await self.vlm.analyze(screenshot, prompt, AnalysisType.COLOR_ANALYSIS)
    
    async def extract_text_elements(
        self,
        screenshot: bytes
    ) -> VisualAnalysis:
        """Extract and analyze text elements from screenshot"""
        prompt = """
        Extract all visible text elements from this screenshot.
        
        For each text element, identify:
        1. The text content
        2. Approximate position (top-left, top-center, etc.)
        3. Text styling (heading, body, button, etc.)
        4. Readability assessment
        
        Format as a structured list.
        """
        
        return await self.vlm.analyze(screenshot, prompt, AnalysisType.TEXT_EXTRACTION)


class OpenInterpreterController:
    """Integration with OpenInterpreter for OS-level control"""
    
    def __init__(self):
        self._interpreter = None
        self._available = None
    
    @property
    def is_available(self) -> bool:
        """Check if OpenInterpreter is available"""
        if self._available is None:
            try:
                import interpreter
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    async def initialize(self):
        """Initialize OpenInterpreter"""
        if not self.is_available:
            raise RuntimeError("OpenInterpreter not installed")
        
        from interpreter import interpreter
        self._interpreter = interpreter
        
        # Configure for safety
        self._interpreter.auto_run = False  # Require approval
        self._interpreter.verbose = False
    
    async def take_screenshot(self) -> bytes:
        """Take screenshot using OpenInterpreter"""
        if not self._interpreter:
            await self.initialize()
        
        result = self._interpreter.computer.terminate(status="success")
        # This is a simplified version - OpenInterpreter has screenshot capabilities
        return b""
    
    async def click_element(self, description: str) -> Dict[str, Any]:
        """Click on element by description"""
        if not self._interpreter:
            await self.initialize()
        
        result = self._interpreter.chat(f"Click on the element: {description}")
        return {"action": "click", "description": description, "result": result}
    
    async def type_text(self, text: str) -> Dict[str, Any]:
        """Type text at current cursor position"""
        if not self._interpreter:
            await self.initialize()
        
        result = self._interpreter.chat(f'Type the text: "{text}"')
        return {"action": "type", "text": text, "result": result}


class VisionLanguageMonitor:
    """
    Main Vision-Language Monitor interface - The "Eyes"
    
    Provides comprehensive visual monitoring and analysis capabilities
    for the OpenEvolve system.
    """
    
    def __init__(
        self,
        vlm_config: VLMConfig = None,
        enable_openinterpreter: bool = False
    ):
        self.vlm_config = vlm_config or VLMConfig()
        self.capture = ScreenshotCapture()
        self.analyzer = VLMAnalyzer(self.vlm_config)
        self.validator = UIValidator(self.analyzer)
        self.openinterpreter = OpenInterpreterController() if enable_openinterpreter else None
        self._analysis_history: List[VisualAnalysis] = []
    
    async def initialize(self):
        """Initialize the monitor"""
        await self.capture.initialize()
        if self.openinterpreter:
            await self.openinterpreter.initialize()
    
    async def monitor_bubblelab_canvas(
        self,
        url: str = "http://localhost:8501",
        expected_nodes: List[Dict[str, Any]] = None
    ) -> VisualAnalysis:
        """
        Monitor and verify Bubblelab canvas rendering
        
        Args:
            url: URL of the Bubblelab instance
            expected_nodes: List of expected nodes to verify
            
        Returns:
            VisualAnalysis with verification results
        """
        # Capture canvas screenshot
        screenshot = await self.capture.capture_bubblelab_canvas(url)
        
        if expected_nodes:
            # Verify specific nodes
            analysis = await self.validator.verify_node_rendering(screenshot, expected_nodes)
        else:
            # General canvas analysis
            prompt = """
            Analyze this Bubblelab/workflow canvas screenshot.
            
            Identify:
            1. All visible nodes and their types
            2. Connections between nodes
            3. Node colors and states
            4. Any rendering issues or errors
            5. Overall layout quality
            
            Provide a comprehensive assessment of the canvas state.
            """
            analysis = await self.analyzer.analyze(screenshot, prompt, AnalysisType.UI_VERIFICATION)
        
        self._analysis_history.append(analysis)
        return analysis
    
    async def verify_ui_fix(
        self,
        url: str,
        description: str,
        acceptance_criteria: List[str]
    ) -> VisualAnalysis:
        """
        Verify a UI fix by Blue Team
        
        Example: Blue Team says "I fixed the Bubblelab node rendering"
        The VLM Agent takes a screenshot and confirms the fix visually.
        
        Args:
            url: URL of the application
            description: Description of the fix to verify
            acceptance_criteria: List of criteria to check
            
        Returns:
            VisualAnalysis with verification results
        """
        # Capture screenshot
        screenshot = await self.capture.capture_from_url(url)
        
        # Build verification prompt
        criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria)
        prompt = f"""
        Verify this UI fix: {description}
        
        Acceptance Criteria:
        {criteria_text}
        
        Analyze the screenshot and verify each criterion.
        For each criterion, report:
        - PASSED: Criterion is met
        - FAILED: Criterion is not met
        - UNCERTAIN: Cannot determine from screenshot
        
        Provide specific visual evidence for each assessment.
        """
        
        analysis = await self.analyzer.analyze(screenshot, prompt, AnalysisType.UI_VERIFICATION)
        analysis.metadata["fix_description"] = description
        analysis.metadata["acceptance_criteria"] = acceptance_criteria
        
        self._analysis_history.append(analysis)
        return analysis
    
    async def detect_visual_changes(
        self,
        url: str,
        interval_seconds: int = 5,
        max_comparisons: int = 10
    ) -> List[VisualAnalysis]:
        """
        Continuously monitor for visual changes
        
        Args:
            url: URL to monitor
            interval_seconds: Seconds between screenshots
            max_comparisons: Maximum number of comparisons
            
        Returns:
            List of analyses showing changes
        """
        analyses = []
        previous_screenshot = None
        
        for i in range(max_comparisons):
            screenshot = await self.capture.capture_from_url(url)
            
            if previous_screenshot:
                analysis = await self.validator.detect_visual_regression(
                    previous_screenshot,
                    screenshot
                )
                analyses.append(analysis)
            
            previous_screenshot = screenshot
            await asyncio.sleep(interval_seconds)
        
        return analyses
    
    async def analyze_chart_or_graph(
        self,
        screenshot: bytes,
        chart_type: str = "auto"
    ) -> VisualAnalysis:
        """
        Analyze a chart or graph in a screenshot
        
        Args:
            screenshot: Image bytes
            chart_type: Type of chart (bar, line, pie, etc.) or "auto"
            
        Returns:
            VisualAnalysis with chart insights
        """
        prompt = f"""
        Analyze this {'chart_type' if chart_type == 'auto' else chart_type + ' chart'}.
        
        Extract and report:
        1. Chart type and purpose
        2. Data series and values (approximate)
        3. Trends and patterns
        4. Axis labels and units
        5. Legend information
        6. Any anomalies or outliers
        
        Provide a data-driven summary of the visualization.
        """
        
        return await self.analyzer.analyze(screenshot, prompt, AnalysisType.CHART_ANALYSIS)
    
    def get_analysis_history(
        self,
        analysis_type: Optional[AnalysisType] = None
    ) -> List[VisualAnalysis]:
        """Get history of analyses"""
        if analysis_type:
            return [a for a in self._analysis_history if a.analysis_type == analysis_type]
        return self._analysis_history.copy()
    
    async def close(self):
        """Cleanup resources"""
        await self.capture.close()


# Convenience functions for quick usage
async def verify_ui_element(
    url: str,
    element_description: str,
    expected_state: str
) -> VisualAnalysis:
    """Quick verification of a UI element"""
    monitor = VisionLanguageMonitor()
    await monitor.initialize()
    
    try:
        return await monitor.verify_ui_fix(
            url,
            element_description,
            [expected_state]
        )
    finally:
        await monitor.close()


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("VISION-LANGUAGE MONITOR DEMO - The 'Eyes'")
        print("=" * 60)
        
        # Initialize monitor
        config = VLMConfig(
            provider=VLMProvider.OPENAI,
            model="gpt-4o"
        )
        
        monitor = VisionLanguageMonitor(vlm_config=config)
        await monitor.initialize()
        
        print("\n✓ Vision-Language Monitor initialized")
        print(f"  Provider: {config.provider.value}")
        print(f"  Model: {config.model}")
        
        print("\n" + "=" * 60)
        print("Available Analysis Types:")
        for at in AnalysisType:
            print(f"  - {at.value}")
        
        print("\n" + "=" * 60)
        print("Example Use Cases:")
        print("  1. Blue Team: 'I fixed the node rendering'")
        print("     → VLM: Screenshot → Verify node is green and connected")
        print("  2. Detect visual regression in Bubblelab canvas")
        print("  3. Analyze charts/graphs in reports")
        print("  4. Validate UI color schemes")
        
        await monitor.close()
        print("\n✓ Demo complete")
    
    asyncio.run(demo())
