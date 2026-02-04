"""Message primitives for LoongFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MimeType(str, Enum):
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    JSON = "application/json"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    RUNNING = "running"


@dataclass
class Element:
    type: str
    content: str
    mime_type: MimeType = MimeType.TEXT
    metadata: dict = field(default_factory=dict)


BaseElement = Element


@dataclass
class ContentElement(Element):
    pass


@dataclass
class ThinkElement(Element):
    pass


@dataclass
class ToolCallElement(Element):
    tool_name: Optional[str] = None


@dataclass
class ToolOutputElement(Element):
    status: ToolStatus = ToolStatus.SUCCESS


ElementT = Union[Element, ContentElement, ThinkElement, ToolCallElement, ToolOutputElement]


@dataclass
class Message:
    role: Role
    content: str
    elements: List[ElementT] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_element(self, element: ElementT) -> None:
        self.elements.append(element)


__all__ = [
    "Message",
    "Role",
    "MimeType",
    "Element",
    "ElementT",
    "ToolStatus",
    "BaseElement",
    "ContentElement",
    "ThinkElement",
    "ToolCallElement",
    "ToolOutputElement",
]
