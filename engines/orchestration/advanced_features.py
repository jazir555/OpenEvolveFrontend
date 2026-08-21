"""
Sovereign-Grade Problem Decomposition System - Advanced Features
Implements multi-modal support, collaboration features, and domain-specific templates.
"""
from __future__ import annotations


import asyncio
import threading
import json
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum
import uuid
import base64
from PIL import Image
import io
import requests
from io import BytesIO
import re
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class MultiModalType(Enum):
    """Types of multi-modal content"""
    IMAGE = "image"
    DIAGRAM = "diagram"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    STRUCTURED_DATA = "structured_data"


@dataclass
class MultiModalContent:
    """Represents multi-modal content for analysis"""
    id: str
    content_type: MultiModalType
    data: Union[str, bytes, Dict[str, Any]]
    metadata: Dict[str, Any]
    analysis_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content_type': self.content_type.value,
            'data': self.data if isinstance(self.data, (str, dict)) else base64.b64encode(self.data).decode(),
            'metadata': self.metadata,
            'analysis_result': self.analysis_result
        }


class DiagramAnalyzer:
    """Analyzes diagrams and images for problem decomposition"""
    
    def __init__(self, openevolve_client=None):
        self.openevolve_client = openevolve_client
    
    def analyze_image(self, image_data: Union[str, bytes], content_type: str = "image") -> Dict[str, Any]:
        """Analyze an image or diagram"""
        try:
            # If image_data is a URL, download it
            if isinstance(image_data, str) and image_data.startswith(('http://', 'https://')):
                response = requests.get(image_data)
                image_data = response.content
            
            # If image_data is bytes but needs to be processed
            if isinstance(image_data, bytes):
                # Convert to base64 for potential API use
                image_base64 = base64.b64encode(image_data).decode()
                
                # Try to analyze using OpenEvolve if available
                if self.openevolve_client:
                    analysis = self._analyze_with_openevolve_image(image_base64, content_type)
                else:
                    # Fallback: basic analysis
                    analysis = self._analyze_basic_image(image_data, content_type)
            else:
                analysis = {"error": "Unsupported image data format"}
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
    
    def _analyze_with_openevolve_image(self, image_base64: str, content_type: str) -> Dict[str, Any]:
        """Analyze image using OpenEvolve"""
        try:
            prompt = f"Analyze this {content_type} and extract key information relevant to problem decomposition. Provide structured output."
            
            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="image_base64",
                max_iterations=1,
                image_data=image_base64
            )
            
            if result.success and result.best_code:
                return json.loads(result.best_code) if isinstance(result.best_code, str) else result.best_code
            else:
                return {"error": "OpenEvolve analysis failed"}
        except Exception as e:
            logger.warning(f"Falling back to basic analysis due to OpenEvolve error: {e}")
            return self._analyze_basic_image(base64.b64decode(image_base64), content_type)
    
    def _analyze_basic_image(self, image_data: bytes, content_type: str) -> Dict[str, Any]:
        """Basic image analysis when OpenEvolve is not available"""
        try:
            # Try to get image info using PIL
            image = Image.open(BytesIO(image_data))
            width, height = image.size
            mode = image.mode
            
            # Basic analysis
            analysis = {
                "size": {"width": width, "height": height},
                "mode": mode,
                "format": image.format,
                "content_type": content_type,
                "contains_text": self._image_has_text(image)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error in basic image analysis: {e}")
            return {"error": f"Basic analysis failed: {str(e)}"}
    
    def _image_has_text(self, image: Image.Image) -> bool:
        """Detect if image likely contains text (simple heuristic)"""
        # This is a very basic implementation
        # In practice, you'd use OCR or more sophisticated image analysis
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Look for high contrast areas that might indicate text
            pixels = list(gray.getdata())
            unique_values = len(set(pixels))
            
            # If there's high variation in pixel values, it might contain text
            return unique_values > 50
        except Exception as e:
            # Log the specific error for debugging
            import logging
            logging.exception(f"Error in unique_values check: {e}")
            return False


class VisualDecompositionRenderer:
    """Creates visual representations of decompositions"""
    
    def __init__(self):
        self.supported_formats = ['mermaid', 'graphviz', 'plantuml', 'svg']
    
    def create_visual_decomposition(self, decomposition_plan: Dict[str, Any], 
                                  format_type: str = 'mermaid') -> str:
        """Create a visual representation of a decomposition plan"""
        try:
            if format_type == 'mermaid':
                return self._create_mermaid_diagram(decomposition_plan)
            elif format_type == 'graphviz':
                return self._create_graphviz_diagram(decomposition_plan)
            elif format_type == 'plantuml':
                return self._create_plantuml_diagram(decomposition_plan)
            elif format_type == 'svg':
                return self._create_svg_diagram(decomposition_plan)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            logger.error(f"Error creating visual decomposition: {e}")
            return f"Error creating visual decomposition: {e}"
    
    def _create_mermaid_diagram(self, decomposition_plan: Dict[str, Any]) -> str:
        """Create a Mermaid diagram for the decomposition plan"""
        try:
            # Get sub-problems
            sub_problems = decomposition_plan.get('sub_problems', [])
            
            mermaid_lines = [
                "graph TD;",
                "    PROBLEM[Original Problem];"
            ]
            
            for i, sub_problem in enumerate(sub_problems):
                sub_id = f"SP_{i+1}"
                sub_title = sub_problem.get('title', f'Sub-problem {i+1}')
                mermaid_lines.append(f"    PROBLEM --> {sub_id}[{sub_title}];")
                
                # Add dependencies if they exist
                dependencies = sub_problem.get('dependencies', [])
                for dep in dependencies:
                    # Find the dependency index
                    dep_idx = next((j for j, sp in enumerate(sub_problems) if sp.get('id') == dep), -1)
                    if dep_idx >= 0:
                        dep_id = f"SP_{dep_idx+1}"
                        mermaid_lines.append(f"    {dep_id} --> {sub_id};")
            
            return "\n".join(mermaid_lines)
        except Exception as e:
            logger.error(f"Error creating Mermaid diagram: {e}")
            return f"Error creating Mermaid diagram: {e}"
    
    def _create_graphviz_diagram(self, decomposition_plan: Dict[str, Any]) -> str:
        """Create a Graphviz diagram for the decomposition plan"""
        try:
            sub_problems = decomposition_plan.get('sub_problems', [])
            
            gv_lines = [
                "digraph Decomposition {",
                "    node [shape=box];",
                f'    PROBLEM [label="{decomposition_plan.get("title", "Problem")}", style=filled, color=lightblue];'
            ]
            
            # Add sub-problems
            for i, sub_problem in enumerate(sub_problems):
                sub_id = f"SP_{i+1}"
                sub_title = sub_problem.get('title', f'Sub-problem {i+1}')
                gv_lines.append(f'    {sub_id} [label="{sub_title}"];')
                
                # Add edge from main problem
                gv_lines.append(f"    PROBLEM -> {sub_id};")
                
                # Add dependency edges
                dependencies = sub_problem.get('dependencies', [])
                for dep in dependencies:
                    # Find the dependency index
                    dep_idx = next((j for j, sp in enumerate(sub_problems) if sp.get('id') == dep), -1)
                    if dep_idx >= 0:
                        dep_id = f"SP_{dep_idx+1}"
                        gv_lines.append(f"    {dep_id} -> {sub_id};")
            
            gv_lines.append("}")
            return "\n".join(gv_lines)
        except Exception as e:
            logger.error(f"Error creating Graphviz diagram: {e}")
            return f"Error creating Graphviz diagram: {e}"
    
    def _create_plantuml_diagram(self, decomposition_plan: Dict[str, Any]) -> str:
        """Create a PlantUML diagram for the decomposition plan"""
        try:
            sub_problems = decomposition_plan.get('sub_problems', [])
            
            plantuml_lines = [
                "@startuml",
                "title Problem Decomposition",
                "",
                "component \"Original Problem\" as PROBLEM",
            ]
            
            for i, sub_problem in enumerate(sub_problems):
                sub_title = sub_problem.get('title', f'Sub-problem {i+1}')
                plantuml_lines.append(f"component \"{sub_title}\" as SP{i+1}")
            
            # Add relationships
            plantuml_lines.append("")
            plantuml_lines.append("PROBLEM --> SP1 : contains")
            
            for i, sub_problem in enumerate(sub_problems):
                dependencies = sub_problem.get('dependencies', [])
                for dep in dependencies:
                    # Find the dependency index
                    dep_idx = next((j for j, sp in enumerate(sub_problems) if sp.get('id') == dep), -1)
                    if dep_idx >= 0:
                        plantuml_lines.append(f"SP{dep_idx+1} --> SP{i+1} : depends on")
            
            plantuml_lines.append("@enduml")
            return "\n".join(plantuml_lines)
        except Exception as e:
            logger.error(f"Error creating PlantUML diagram: {e}")
            return f"Error creating PlantUML diagram: {e}"
    
    def _create_svg_diagram(self, decomposition_plan: Dict[str, Any]) -> str:
        """Create an SVG diagram for the decomposition plan"""
        try:
            sub_problems = decomposition_plan.get('sub_problems', [])
            width = 800
            height = 400
            
            svg_lines = [
                f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
                '  <style>',
                '    .problem {{ fill: #e1f5fe; stroke: #0277bd; stroke-width: 2; }}',
                '    .subproblem {{ fill: #f3e5f5; stroke: #7b1fa2; stroke-width: 1; }}',
                '    .text {{ font-family: Arial, sans-serif; font-size: 12px; }}',
                '    .line {{ stroke: #9e9e9e; stroke-width: 1; marker-end: url(#arrowhead); }}',
                '  </style>',
                '  <defs>',
                '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">',
                '      <polygon points="0 0, 10 3.5, 0 7" fill="#9e9e9e" />',
                '    </marker>',
                '  </defs>',
                f'  <rect class="problem" x="10" y="10" width="200" height="50" rx="10" ry="10"/>',
                f'  <text class="text" x="110" y="40" text-anchor="middle">{decomposition_plan.get("title", "Problem")[:20]}...</text>'
            ]
            
            # Position sub-problems in a grid
            for i, sub_problem in enumerate(sub_problems):
                row = i // 3
                col = i % 3
                x = 300 + col * 150
                y = 20 + row * 80
                
                sub_title = sub_problem.get('title', f'Sub-problem {i+1}')[:15] + '...'
                
                svg_lines.append(f'  <rect class="subproblem" x="{x}" y="{y}" width="120" height="40" rx="5" ry="5"/>')
                svg_lines.append(f'  <text class="text" x="{x + 60}" y="{y + 25}" text-anchor="middle">{sub_title}</text>')
                svg_lines.append(f'  <line class="line" x1="210" y1="35" x2="{x}" y2="{y + 20}"/>')
            
            svg_lines.append('</svg>')
            return "\n".join(svg_lines)
        except Exception as e:
            logger.error(f"Error creating SVG diagram: {e}")
            return f"Error creating SVG diagram: {e}"


class InteractiveEditor:
    """Interactive editor for decomposition plans"""
    
    def __init__(self):
        self.edit_sessions = {}  # session_id -> decomposition_plan
        self.changes_history = {}  # session_id -> list of changes
    
    def start_edit_session(self, decomposition_plan: Dict[str, Any]) -> str:
        """Start a new interactive editing session"""
        session_id = str(uuid.uuid4())
        self.edit_sessions[session_id] = decomposition_plan.copy()
        self.changes_history[session_id] = []
        
        logger.info(f"Started edit session: {session_id}")
        return session_id
    
    def get_current_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of the plan in an edit session"""
        return self.edit_sessions.get(session_id)
    
    def make_change(self, session_id: str, change_type: str, data: Dict[str, Any]) -> bool:
        """Apply a change to the decomposition plan"""
        if session_id not in self.edit_sessions:
            return False
        
        plan = self.edit_sessions[session_id]
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'change_type': change_type,
            'data': data,
            'previous_state': plan.copy()  # Simplified - in practice, you'd create deep copy
        }
        
        try:
            if change_type == 'add_sub_problem':
                if 'sub_problems' not in plan:
                    plan['sub_problems'] = []
                plan['sub_problems'].append(data)
            elif change_type == 'update_sub_problem':
                sub_problem_id = data.get('id')
                existing_idx = next((i for i, sp in enumerate(plan.get('sub_problems', [])) 
                                   if sp.get('id') == sub_problem_id), -1)
                if existing_idx >= 0:
                    plan['sub_problems'][existing_idx] = data
            elif change_type == 'delete_sub_problem':
                sub_problem_id = data.get('id')
                plan['sub_problems'] = [
                    sp for sp in plan.get('sub_problems', []) 
                    if sp.get('id') != sub_problem_id
                ]
            elif change_type == 'add_dependency':
                sub_problem_id = data.get('sub_problem_id')
                dependency_id = data.get('dependency_id')
                sub_problems = plan.get('sub_problems', [])
                for sp in sub_problems:
                    if sp.get('id') == sub_problem_id:
                        if 'dependencies' not in sp:
                            sp['dependencies'] = []
                        if dependency_id not in sp['dependencies']:
                            sp['dependencies'].append(dependency_id)
                        break
            elif change_type == 'update_metadata':
                # Update metadata fields in the plan
                for key, value in data.items():
                    plan[key] = value
            
            # Record the change
            self.changes_history[session_id].append(change_record)
            return True
        except Exception as e:
            logger.error(f"Error making change: {e}")
            return False
    
    def undo_last_change(self, session_id: str) -> bool:
        """Undo the last change made in the session"""
        if session_id not in self.changes_history or not self.changes_history[session_id]:
            return False
        
        try:
            last_change = self.changes_history[session_id].pop()
            self.edit_sessions[session_id] = last_change['previous_state']
            return True
        except Exception as e:
            logger.error(f"Error undoing change: {e}")
            return False
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the change history for a session"""
        return self.changes_history.get(session_id, [])
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End an edit session and return the final plan"""
        plan = self.edit_sessions.pop(session_id, None)
        self.changes_history.pop(session_id, None)
        
        if plan:
            logger.info(f"Ended edit session: {session_id}")
        
        return plan


class MultiModalAnalyzer:
    """Analyzes multi-modal content in problem statements"""
    
    def __init__(self, openevolve_client=None):
        self.diagram_analyzer = DiagramAnalyzer(openevolve_client)
        self.content_cache = {}  # Cache for analyzed content
        self._lock = threading.Lock()
    
    def analyze_multi_modal_content(self, content: MultiModalContent) -> Dict[str, Any]:
        """Analyze multi-modal content based on its type"""
        try:
            if content.content_type in [MultiModalType.IMAGE, MultiModalType.DIAGRAM]:
                analysis = self.diagram_analyzer.analyze_image(
                    content.data, 
                    content_type=content.content_type.value
                )
            elif content.content_type == MultiModalType.AUDIO:
                analysis = self._analyze_audio(content.data)
            elif content.content_type == MultiModalType.VIDEO:
                analysis = self._analyze_video(content.data)
            elif content.content_type == MultiModalType.TEXT:
                analysis = self._analyze_text(content.data)
            elif content.content_type == MultiModalType.STRUCTURED_DATA:
                analysis = self._analyze_structured_data(content.data)
            else:
                analysis = {"error": f"Unsupported content type: {content.content_type}"}
            
            # Store analysis result
            content.analysis_result = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing multi-modal content: {e}")
            return {"error": str(e)}
    
    def _analyze_audio(self, audio_data: Union[str, bytes]) -> Dict[str, Any]:
        """Analyze audio content"""
        import os
        import io
        import mimetypes
        import wave
        import base64

        analysis = {
            "analysis_type": "audio",
            "format": "unknown",
            "duration_seconds": None,
            "sample_rate_hz": None,
            "channels": None,
            "bitrate_kbps": None,
            "contains_speech": None,
            "transcription_available": False,
            "size_bytes": None
        }

        raw_bytes = None
        source_path = None

        if isinstance(audio_data, bytes):
            raw_bytes = audio_data
        elif isinstance(audio_data, str):
            if os.path.isfile(audio_data):
                source_path = audio_data
                analysis["size_bytes"] = os.path.getsize(audio_data)
                analysis["format"] = os.path.splitext(audio_data)[1].lstrip(".").lower() or "unknown"
            else:
                try:
                    raw_bytes = base64.b64decode(audio_data, validate=True)
                except (ValueError, TypeError):
                    raw_bytes = None
        else:
            return {"analysis_type": "audio", "error": "Unsupported audio input type"}

        if raw_bytes is not None:
            analysis["size_bytes"] = len(raw_bytes)
            if raw_bytes[:4] == b"RIFF" and b"WAVE" in raw_bytes[:16]:
                analysis["format"] = "wav"

        if analysis["format"] == "wav" or (raw_bytes and raw_bytes[:4] == b"RIFF"):
            try:
                if source_path:
                    with wave.open(source_path, "rb") as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                else:
                    with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        channels = wav_file.getnchannels()

                duration = frames / float(rate) if rate else None
                analysis.update({
                    "duration_seconds": duration,
                    "sample_rate_hz": rate,
                    "channels": channels,
                    "bitrate_kbps": (rate * channels * 16) / 1000 if rate and channels else None
                })
            except (wave.Error, IOError, OSError) as e:
                analysis["error"] = f"Failed to parse WAV audio: {e}"
        else:
            if source_path:
                mime_type, _ = mimetypes.guess_type(source_path)
                analysis["format"] = (mime_type or analysis["format"]).split("/")[-1]

            try:
                from mutagen import File as MutagenFile  # type: ignore
                media = MutagenFile(source_path or io.BytesIO(raw_bytes))
                if media:
                    analysis["duration_seconds"] = getattr(media.info, "length", None)
                    analysis["bitrate_kbps"] = getattr(media.info, "bitrate", 0) / 1000 if hasattr(media.info, "bitrate") else None
                    analysis["sample_rate_hz"] = getattr(media.info, "sample_rate", None)
                    analysis["channels"] = getattr(media.info, "channels", None)
            except (ImportError, AttributeError):
                analysis["analysis_notes"] = "Basic metadata only; install mutagen for richer analysis"

        return analysis
    
    def _analyze_video(self, video_data: Union[str, bytes]) -> Dict[str, Any]:
        """Analyze video content"""
        import os
        import tempfile
        import mimetypes

        analysis = {
            "analysis_type": "video",
            "format": "unknown",
            "duration_seconds": None,
            "frame_count": None,
            "fps": None,
            "resolution": None,
            "size_bytes": None,
            "contains_visual_elements": True
        }

        source_path = None
        temp_path = None

        if isinstance(video_data, str) and os.path.isfile(video_data):
            source_path = video_data
            analysis["size_bytes"] = os.path.getsize(video_data)
            analysis["format"] = os.path.splitext(video_data)[1].lstrip(".").lower() or "unknown"
        elif isinstance(video_data, bytes):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_data)
            temp_file.flush()
            temp_path = temp_file.name
            temp_file.close()
            source_path = temp_path
            analysis["size_bytes"] = len(video_data)
            analysis["format"] = "mp4"
        else:
            return {"analysis_type": "video", "error": "Unsupported video input type"}

        try:
            mime_type, _ = mimetypes.guess_type(source_path)
            if mime_type and "/" in mime_type:
                analysis["format"] = mime_type.split("/")[-1]
        except Exception as exc:
            logger.debug(f"Unable to infer video mime type: {exc}")

        try:
            import cv2  # type: ignore
            capture = cv2.VideoCapture(source_path)
            if capture.isOpened():
                fps = capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else None
                analysis.update({
                    "fps": fps or None,
                    "frame_count": frame_count,
                    "resolution": f"{width}x{height}" if width and height else None,
                    "duration_seconds": duration
                })
            capture.release()
        except (ImportError, RuntimeError, OSError):
            try:
                from moviepy.editor import VideoFileClip  # type: ignore
                with VideoFileClip(source_path) as clip:
                    analysis.update({
                        "duration_seconds": clip.duration,
                        "fps": clip.fps,
                        "resolution": f"{int(clip.w)}x{int(clip.h)}",
                        "frame_count": int(clip.duration * clip.fps) if clip.duration and clip.fps else None
                    })
            except (ImportError, RuntimeError, OSError):
                analysis["analysis_notes"] = "Basic metadata only; install opencv-python or moviepy for richer analysis"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (OSError, IOError) as exc:
                    logger.debug(f"Unable to clean up temporary video file {temp_path}: {exc}")

        return analysis
    
    def _analyze_text(self, text_data: str) -> Dict[str, Any]:
        """Analyze text content"""
        # Basic text analysis
        word_count = len(text_data.split())
        char_count = len(text_data)
        
        # Extract potential entities (simplified)
        emails = re.findall(r'\S+@\S+', text_data)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text_data)
        
        return {
            "analysis_type": "text",
            "word_count": word_count,
            "character_count": char_count,
            "email_count": len(emails),
            "url_count": len(urls),
            "has_complex_structure": word_count > 100
        }
    
    def _analyze_structured_data(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze structured data (JSON, CSV, etc.)"""
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        else:
            parsed_data = data
        
        if isinstance(parsed_data, dict):
            return {
                "analysis_type": "structured_data",
                "data_type": "object",
                "property_count": len(parsed_data),
                "nested_depth": self._calculate_nested_depth(parsed_data)
            }
        elif isinstance(parsed_data, list):
            return {
                "analysis_type": "structured_data",
                "data_type": "array",
                "item_count": len(parsed_data),
                "sample_item_keys": list(parsed_data[0].keys()) if parsed_data and isinstance(parsed_data[0], dict) else []
            }
        else:
            return {
                "analysis_type": "structured_data",
                "data_type": type(parsed_data).__name__,
                "value": str(parsed_data)[:100]  # Truncate long values
            }
    
    def _calculate_nested_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of an object"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_nested_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_nested_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth


class CollaborationManager:
    """Manages collaboration features for multi-user sessions"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> session_info
        self.session_users = {}  # session_id -> [user_ids]
        self.session_plans = {}  # session_id -> decomposition_plan
        self.change_queue = {}  # session_id -> queue of changes
        self._lock = threading.Lock()
    
    def create_collaboration_session(self, user_id: str, initial_plan: Dict[str, Any] = None) -> str:
        """Create a new collaboration session"""
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self.sessions[session_id] = {
                'id': session_id,
                'created_by': user_id,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'users': [user_id],
                'plan': initial_plan or {}
            }
            self.session_users[session_id] = [user_id]
            self.session_plans[session_id] = initial_plan or {}
            self.change_queue[session_id] = []
        
        logger.info(f"Created collaboration session {session_id} for user {user_id}")
        return session_id
    
    def join_session(self, session_id: str, user_id: str) -> bool:
        """Allow a user to join a collaboration session"""
        if session_id not in self.sessions:
            return False
        
        with self._lock:
            if user_id not in self.session_users[session_id]:
                self.session_users[session_id].append(user_id)
                self.sessions[session_id]['users'].append(user_id)
        
        logger.info(f"User {user_id} joined session {session_id}")
        return True
    
    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Allow a user to leave a collaboration session"""
        if session_id not in self.sessions:
            return False
        
        with self._lock:
            if user_id in self.session_users[session_id]:
                self.session_users[session_id].remove(user_id)
                self.sessions[session_id]['users'].remove(user_id)
                
                # If no users left, end session
                if not self.session_users[session_id]:
                    self.end_session(session_id)
        
        logger.info(f"User {user_id} left session {session_id}")
        return True
    
    def add_change_to_session(self, session_id: str, user_id: str, change: Dict[str, Any]) -> bool:
        """Add a change to the session change queue"""
        if session_id not in self.sessions or user_id not in self.session_users[session_id]:
            return False
        
        change_with_metadata = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'change': change
        }
        
        with self._lock:
            self.change_queue[session_id].append(change_with_metadata)
        
        logger.debug(f"Change added to session {session_id} by user {user_id}")
        return True
    
    def get_pending_changes(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get pending changes for a user in a session"""
        if session_id not in self.sessions or user_id not in self.session_users[session_id]:
            return []
        
        # Return all changes (in practice, you'd return only changes since last sync)
        with self._lock:
            return [c for c in self.change_queue[session_id] if c['user_id'] != user_id]
    
    def get_current_plan(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current plan for a session"""
        return self.session_plans.get(session_id)
    
    def update_plan_in_session(self, session_id: str, plan: Dict[str, Any]) -> bool:
        """Update the plan in a session"""
        if session_id not in self.sessions:
            return False
        
        with self._lock:
            self.session_plans[session_id] = plan
            self.sessions[session_id]['plan'] = plan
        
        logger.info(f"Plan updated in session {session_id}")
        return True
    
    def end_session(self, session_id: str) -> bool:
        """End a collaboration session"""
        if session_id not in self.sessions:
            return False
        
        with self._lock:
            del self.sessions[session_id]
            del self.session_users[session_id]
            del self.session_plans[session_id]
            del self.change_queue[session_id]
        
        logger.info(f"Ended collaboration session {session_id}")
        return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a collaboration session"""
        return self.sessions.get(session_id)


class VersionControlSystem:
    """Version control for decomposition plans"""
    
    def __init__(self, storage_path: str = "decomposition_versions.db"):
        self.storage_path = storage_path
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage for version history"""
        import sqlite3
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plan_versions (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    plan_data TEXT NOT NULL,
                    author TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    commit_message TEXT,
                    parent_version TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_versions_plan_id 
                ON plan_versions(plan_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_versions_timestamp 
                ON plan_versions(timestamp)
            """)
    
    def commit_version(self, plan_id: str, plan_data: Dict[str, Any], author: str, 
                      commit_message: str = "", parent_version: Optional[str] = None) -> str:
        """Commit a new version of a plan"""
        import sqlite3
        
        version_id = str(uuid.uuid4())
        version_number = self._get_next_version_number(plan_id)
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO plan_versions 
                (id, plan_id, version_number, plan_data, author, timestamp, commit_message, parent_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id, plan_id, version_number,
                json.dumps(plan_data), author, datetime.now().isoformat(),
                commit_message, parent_version
            ))
        
        logger.info(f"Committed version {version_number} for plan {plan_id}")
        return version_id
    
    def _get_next_version_number(self, plan_id: str) -> int:
        """Get the next version number for a plan"""
        import sqlite3
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(version_number) FROM plan_versions WHERE plan_id = ?
            """, (plan_id,))
            result = cursor.fetchone()
        
        last_version = result[0] or 0
        return last_version + 1
    
    def get_plan_version(self, plan_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific version of a plan"""
        import sqlite3
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT plan_data FROM plan_versions 
                WHERE plan_id = ? AND version_number = ?
            """, (plan_id, version_number))
            result = cursor.fetchone()
        
        if result:
            return json.loads(result[0])
        return None
    
    def get_plan_history(self, plan_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the version history for a plan"""
        import sqlite3
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, version_number, author, timestamp, commit_message
                FROM plan_versions 
                WHERE plan_id = ?
                ORDER BY version_number DESC
                LIMIT ?
            """, (plan_id, limit))
            results = cursor.fetchall()
        
        return [
            {
                'id': row[0],
                'version_number': row[1],
                'author': row[2],
                'timestamp': row[3],
                'commit_message': row[4]
            }
            for row in results
        ]
    
    def rollback_to_version(self, plan_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """Rollback a plan to a specific version"""
        plan_data = self.get_plan_version(plan_id, version_number)
        if plan_data:
            logger.info(f"Rolled back plan {plan_id} to version {version_number}")
        return plan_data


class NotificationSystem:
    """Notification system for workflow updates"""
    
    def __init__(self):
        self.subscribers = {}  # event_type -> [callbacks]
        self.notifications = []  # List of recent notifications
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to a specific event type"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """Unsubscribe from a specific event type"""
        with self._lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(callback)
                except ValueError:
                    logger.debug("Callback not found while unsubscribing from event '%s'.", event_type)
    
    def notify(self, event_type: str, data: Dict[str, Any]):
        """Notify all subscribers of an event"""
        with self._lock:
            # Store notification
            notification = {
                'id': str(uuid.uuid4()),
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.notifications.append(notification)
            
            # Keep only recent notifications (last 1000)
            if len(self.notifications) > 1000:
                self.notifications = self.notifications[-1000:]
            
            # Notify subscribers
            callbacks = self.subscribers.get(event_type, [])
        
        for callback in callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    def get_recent_notifications(self, event_type: Optional[str] = None, 
                               minutes_back: int = 60) -> List[Dict[str, Any]]:
        """Get recent notifications"""
        cutoff = datetime.now() - timedelta(minutes=minutes_back)
        
        with self._lock:
            notifications = [
                n for n in self.notifications
                if datetime.fromisoformat(n['timestamp']) > cutoff
                and (event_type is None or n['event_type'] == event_type)
            ]
        
        return notifications


class DomainSpecificTemplates:
    """Domain-specific templates and patterns for problem decomposition"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load default domain-specific templates"""
        return {
            'software_engineering': {
                'name': 'Software Engineering Template',
                'description': 'Template for decomposing software engineering problems',
                'strategies': [
                    {
                        'name': 'Architecture-First',
                        'description': 'Decompose by architectural layers and components',
                        'sub_problem_patterns': [
                            {
                                'title': 'Requirements Analysis',
                                'type': 'analysis',
                                'description': 'Analyze and understand requirements',
                                'dependencies': []
                            },
                            {
                                'title': 'System Design',
                                'type': 'design',
                                'description': 'Design system architecture',
                                'dependencies': ['Requirements Analysis']
                            },
                            {
                                'title': 'Component Implementation',
                                'type': 'implementation',
                                'description': 'Implement individual components',
                                'dependencies': ['System Design']
                            },
                            {
                                'title': 'Integration Testing',
                                'type': 'validation',
                                'description': 'Test component integration',
                                'dependencies': ['Component Implementation']
                            }
                        ]
                    },
                    {
                        'name': 'Feature-Driven',
                        'description': 'Decompose by features and user stories',
                        'sub_problem_patterns': [
                            {
                                'title': 'Feature Definition',
                                'type': 'analysis',
                                'description': 'Define and analyze features',
                                'dependencies': []
                            },
                            {
                                'title': 'Feature Implementation',
                                'type': 'implementation',
                                'description': 'Implement individual features',
                                'dependencies': ['Feature Definition']
                            }
                        ]
                    }
                ]
            },
            'research': {
                'name': 'Research Template',
                'description': 'Template for decomposing research problems',
                'strategies': [
                    {
                        'name': 'Hypothesis-Driven',
                        'description': 'Decompose by hypothesis formation and testing',
                        'sub_problem_patterns': [
                            {
                                'title': 'Literature Review',
                                'type': 'research',
                                'description': 'Review existing literature',
                                'dependencies': []
                            },
                            {
                                'title': 'Hypothesis Formation',
                                'type': 'analysis',
                                'description': 'Formulate hypotheses',
                                'dependencies': ['Literature Review']
                            },
                            {
                                'title': 'Experimental Design',
                                'type': 'design',
                                'description': 'Design experiments to test hypotheses',
                                'dependencies': ['Hypothesis Formation']
                            },
                            {
                                'title': 'Data Collection',
                                'type': 'implementation',
                                'description': 'Collect experimental data',
                                'dependencies': ['Experimental Design']
                            },
                            {
                                'title': 'Analysis & Validation',
                                'type': 'analysis',
                                'description': 'Analyze data and validate findings',
                                'dependencies': ['Data Collection']
                            }
                        ]
                    }
                ]
            },
            'business_strategy': {
                'name': 'Business Strategy Template',
                'description': 'Template for decomposing business strategy problems',
                'strategies': [
                    {
                        'name': 'Market-Driven',
                        'description': 'Decompose by market analysis and positioning',
                        'sub_problem_patterns': [
                            {
                                'title': 'Market Analysis',
                                'type': 'analysis',
                                'description': 'Analyze market conditions and competitors',
                                'dependencies': []
                            },
                            {
                                'title': 'Customer Research',
                                'type': 'research',
                                'description': 'Research customer needs and behaviors',
                                'dependencies': ['Market Analysis']
                            },
                            {
                                'title': 'Strategy Formulation',
                                'type': 'design',
                                'description': 'Formulate strategic approach',
                                'dependencies': ['Customer Research']
                            },
                            {
                                'title': 'Implementation Planning',
                                'type': 'planning',
                                'description': 'Plan implementation of strategy',
                                'dependencies': ['Strategy Formulation']
                            }
                        ]
                    }
                ]
            }
        }
    
    def get_template(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get a domain-specific template"""
        return self.templates.get(domain)
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available templates"""
        return self.templates
    
    def apply_template_to_problem(self, problem_statement: str, domain: str, 
                                 strategy_name: str) -> Optional[Dict[str, Any]]:
        """Apply a template to a problem statement"""
        template = self.get_template(domain)
        if not template:
            return None
        
        # Find the requested strategy
        strategy = None
        for strat in template['strategies']:
            if strat['name'].lower() == strategy_name.lower():
                strategy = strat
                break
        
        if not strategy:
            # Use the first strategy if none specified
            strategy = template['strategies'][0] if template['strategies'] else None
        
        if not strategy:
            return None
        
        # Create a decomposition plan based on the template
        decomposition_plan = {
            'id': str(uuid.uuid4()),
            'original_problem': problem_statement,
            'domain': domain,
            'template_used': template['name'],
            'strategy_used': strategy['name'],
            'sub_problems': [
                {
                    'id': f"sp_{i+1}",
                    'title': sp['title'],
                    'description': sp['description'],
                    'type': sp['type'],
                    'dependencies': sp['dependencies'],
                    'created_from_template': True
                }
                for i, sp in enumerate(strategy['sub_problem_patterns'])
            ],
            'created_at': datetime.now().isoformat()
        }
        
        return decomposition_plan


class AdvancedFeaturesManager:
    """Main manager for advanced features"""
    
    def __init__(self, openevolve_client=None):
        self.multi_modal_analyzer = MultiModalAnalyzer(openevolve_client)
        self.visual_renderer = VisualDecompositionRenderer()
        self.interactive_editor = InteractiveEditor()
        self.collaboration_manager = CollaborationManager()
        self.version_control = VersionControlSystem()
        self.notification_system = NotificationSystem()
        self.domain_templates = DomainSpecificTemplates()
    
    def analyze_multi_modal_problem(self, problem_statement: str, 
                                  media_content: List[MultiModalContent]) -> Dict[str, Any]:
        """Analyze a problem with multi-modal support"""
        results = {
            'problem_text_analysis': self.multi_modal_analyzer._analyze_text(problem_statement),
            'media_analyses': []
        }
        
        for media in media_content:
            analysis = self.multi_modal_analyzer.analyze_multi_modal_content(media)
            results['media_analyses'].append({
                'content_id': media.id,
                'content_type': media.content_type.value,
                'analysis': analysis
            })
        
        return results
    
    def create_visual_representation(self, decomposition_plan: Dict[str, Any], 
                                   format_type: str = 'mermaid') -> str:
        """Create a visual representation of a decomposition plan"""
        return self.visual_renderer.create_visual_decomposition(
            decomposition_plan, format_type
        )
    
    def start_interactive_session(self, decomposition_plan: Dict[str, Any]) -> str:
        """Start an interactive editing session"""
        return self.interactive_editor.start_edit_session(decomposition_plan)
    
    def make_interactive_change(self, session_id: str, change_type: str, 
                              data: Dict[str, Any]) -> bool:
        """Make a change in an interactive session"""
        return self.interactive_editor.make_change(session_id, change_type, data)
    
    def create_collaboration_session(self, user_id: str, 
                                   initial_plan: Dict[str, Any] = None) -> str:
        """Create a collaboration session"""
        return self.collaboration_manager.create_collaboration_session(user_id, initial_plan)
    
    def join_collaboration_session(self, session_id: str, user_id: str) -> bool:
        """Join a collaboration session"""
        return self.collaboration_manager.join_session(session_id, user_id)
    
    def commit_plan_version(self, plan_id: str, plan_data: Dict[str, Any], 
                          author: str, commit_message: str = "") -> str:
        """Commit a version of a plan"""
        return self.version_control.commit_version(plan_id, plan_data, author, commit_message)
    
    def apply_domain_template(self, problem_statement: str, domain: str, 
                            strategy_name: str) -> Optional[Dict[str, Any]]:
        """Apply a domain-specific template to a problem"""
        return self.domain_templates.apply_template_to_problem(
            problem_statement, domain, strategy_name
        )
    
    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get all available domain templates"""
        return self.domain_templates.get_all_templates()


# Global advanced features manager instance
_advanced_features_manager = None


def get_advanced_features_manager(openevolve_client=None) -> AdvancedFeaturesManager:
    """Get the advanced features manager instance"""
    global _advanced_features_manager
    if _advanced_features_manager is None:
        _advanced_features_manager = AdvancedFeaturesManager(openevolve_client)
    return _advanced_features_manager


def analyze_multi_modal_problem(problem_statement: str, 
                              media_content: List[MultiModalContent]) -> Dict[str, Any]:
    """Analyze a multi-modal problem"""
    return get_advanced_features_manager().analyze_multi_modal_problem(
        problem_statement, media_content
    )


def create_visual_representation(decomposition_plan: Dict[str, Any], 
                               format_type: str = 'mermaid') -> str:
    """Create a visual representation of a decomposition plan"""
    return get_advanced_features_manager().create_visual_representation(
        decomposition_plan, format_type
    )


def apply_domain_template(problem_statement: str, domain: str, 
                         strategy_name: str) -> Optional[Dict[str, Any]]:
    """Apply a domain-specific template"""
    return get_advanced_features_manager().apply_domain_template(
        problem_statement, domain, strategy_name
    )


def get_available_templates() -> Dict[str, Dict[str, Any]]:
    """Get available templates"""
    return get_advanced_features_manager().get_available_templates()


# Example usage
if __name__ == "__main__":
    # Create some sample multi-modal content
    sample_image = MultiModalContent(
        id="img_1",
        content_type=MultiModalType.IMAGE,
        data=b"fake_image_data",  # This would be actual image bytes
        metadata={"source": "problem_statement", "type": "diagram"}
    )
    
    # Analyze a multi-modal problem
    problem_text = "Design a system architecture for a multi-tenant SaaS application"
    media_content = [sample_image]
    
    results = analyze_multi_modal_problem(problem_text, media_content)
    print(f"Multi-modal analysis results: {json.dumps(results, indent=2)[:500]}...")
    
    # Get available templates
    templates = get_available_templates()
    print(f"Available templates: {list(templates.keys())}")
    
    # Apply a template
    plan = apply_domain_template(
        "Build a recommendation engine", 
        "software_engineering", 
        "Architecture-First"
    )
    if plan:
        print(f"Applied template result: {json.dumps(plan, indent=2)[:500]}...")
    
    # Create visual representation
    if plan:
        visual = create_visual_representation(plan, "mermaid")
        print(f"Visual representation:\n{visual[:200]}...")
    
    print("Advanced features implemented successfully!")
