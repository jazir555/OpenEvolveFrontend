"""
Dashboard UI Components Module

Reusable UI components for the decomposition engine dashboard.
Provides modular, customizable components for building dashboards.

Features:
- Reusable HTML/CSS components
- JavaScript widgets
- Theme support
- Responsive design
- Accessibility features
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class UIComponent:
    """Base class for UI components."""

    def __init__(self, component_id: str):
        """Initialize component with unique ID."""
        self.component_id = component_id
        self.classes = []
        self.attributes = {}
        self.styles = {}

    def add_class(self, class_name: str):
        """Add CSS class to component."""
        self.classes.append(class_name)
        return self

    def set_attribute(self, key: str, value: str):
        """Set HTML attribute."""
        self.attributes[key] = value
        return self

    def set_style(self, property: str, value: str):
        """Set CSS style property."""
        self.styles[property] = value
        return self

    def _build_attributes(self) -> str:
        """Build HTML attributes string."""
        attrs = []

        if self.classes:
            attrs.append(f'class="{" ".join(self.classes)}"')

        for key, value in self.attributes.items():
            attrs.append(f'{key}="{value}"')

        if self.styles:
            style_str = "; ".join([f"{k}: {v}" for k, v in self.styles.items()])
            attrs.append(f'style="{style_str}"')

        return " " + " ".join(attrs) if attrs else ""

    def render(self) -> str:
        """Render component to HTML."""
        raise NotImplementedError


class Button(UIComponent):
    """Button component with various styles and states."""

    def __init__(
        self,
        button_id: str,
        label: str,
        button_type: str = "button",
        variant: str = "primary",
        size: str = "medium"
    ):
        """
        Initialize button.

        Args:
            button_id: Unique button ID
            label: Button text
            button_type: HTML button type (button, submit, reset)
            variant: Style variant (primary, secondary, success, danger, warning)
            size: Button size (small, medium, large)
        """
        super().__init__(button_id)
        self.label = label
        self.button_type = button_type
        self.variant = variant
        self.size = size
        self.disabled = False
        self.icon = None
        self.onclick = None

    def set_icon(self, icon: str):
        """Set button icon."""
        self.icon = icon
        return self

    def set_onclick(self, handler: str):
        """Set click handler."""
        self.onclick = handler
        return self

    def set_disabled(self, disabled: bool):
        """Set disabled state."""
        self.disabled = disabled
        return self

    def render(self) -> str:
        """Render button HTML."""
        self.add_class(f"btn")
        self.add_class(f"btn-{self.variant}")
        self.add_class(f"btn-{self.size}")

        if self.disabled:
            self.add_class("disabled")
            self.set_attribute("disabled", "disabled")

        if self.onclick:
            self.set_attribute("onclick", self.onclick)

        attrs = self._build_attributes()

        icon_html = f'<i class="icon {self.icon}"></i> ' if self.icon else ""

        return f'<button{attrs} type="{self.button_type}">{icon_html}{self.label}</button>'


class Card(UIComponent):
    """Card container component."""

    def __init__(self, card_id: str, title: str = ""):
        """Initialize card."""
        super().__init__(card_id)
        self.title = title
        self.content = []
        self.actions = []
        self.variant = "default"

    def set_variant(self, variant: str):
        """Set card variant (default, primary, success, warning, danger)."""
        self.variant = variant
        return self

    def add_content(self, content: str):
        """Add content to card."""
        self.content.append(content)
        return self

    def add_action(self, button: Button):
        """Add action button to card header."""
        self.actions.append(button)
        return self

    def render(self) -> str:
        """Render card HTML."""
        self.add_class("card")
        self.add_class(f"card-{self.variant}")

        attrs = self._build_attributes()

        title_html = f""
        if self.title:
            title_html = f'<div class="card-header"><h3 class="card-title">{self.title}</h3></div>'

        actions_html = ""
        if self.actions:
            actions_html = '<div class="card-actions">' + \
                          "".join([btn.render() for btn in self.actions]) + \
                          '</div>'

        content_html = '<div class="card-body">' + \
                       "".join(self.content) + \
                       '</div>'

        return f'''
<div{attrs}>
    {title_html}
    {actions_html}
    {content_html}
</div>
        '''


class ProgressBar(UIComponent):
    """Progress bar component."""

    def __init__(
        self,
        progress_id: str,
        value: float = 0.0,
        max_value: float = 100.0,
        label: str = ""
    ):
        """Initialize progress bar."""
        super().__init__(progress_id)
        self.value = value
        self.max_value = max_value
        self.label = label
        self.variant = "primary"
        self.animated = False
        self.striped = False

    def set_variant(self, variant: str):
        """Set color variant."""
        self.variant = variant
        return self

    def set_animated(self, animated: bool):
        """Set animation."""
        self.animated = animated
        return self

    def set_striped(self, striped: bool):
        """Set striped style."""
        self.striped = striped
        return self

    def update(self, value: float):
        """Update progress value."""
        self.value = max(0, min(value, self.max_value))
        return self

    def render(self) -> str:
        """Render progress bar HTML."""
        percentage = (self.value / self.max_value) * 100 if self.max_value > 0 else 0

        self.add_class("progress")

        attrs = self._build_attributes()

        bar_classes = ["progress-bar"]
        bar_classes.append(f"bg-{self.variant}")

        if self.striped:
            bar_classes.append("progress-bar-striped")

        if self.animated:
            bar_classes.append("progress-bar-animated")

        bar_attrs = " ".join(bar_classes)

        label_html = f'<div class="progress-label">{self.label}</div>' if self.label else ""

        return f'''
<div{attrs}>
    {label_html}
    <div class="progress-bar-wrapper">
        <div class="{bar_attrs}" style="width: {percentage}%"
             role="progressbar"
             aria-valuenow="{self.value}"
             aria-valuemin="0"
             aria-valuemax="{self.max_value}">
            {percentage:.1f}%
        </div>
    </div>
</div>
        '''


class Badge(UIComponent):
    """Badge component for status indicators."""

    def __init__(self, badge_id: str, text: str, variant: str = "primary"):
        """Initialize badge."""
        super().__init__(badge_id)
        self.text = text
        self.variant = variant
        self.pill = False

    def set_pill(self, pill: bool):
        """Set pill shape."""
        self.pill = pill
        return self

    def render(self) -> str:
        """Render badge HTML."""
        self.add_class("badge")
        self.add_class(f"bg-{self.variant}")

        if self.pill:
            self.add_class("rounded-pill")

        attrs = self._build_attributes()

        return f'<span{attrs}>{self.text}</span>'


class MetricCard(UIComponent):
    """Metric display card with value and label."""

    def __init__(
        self,
        card_id: str,
        value: str,
        label: str,
        variant: str = "primary",
        icon: str = None
    ):
        """Initialize metric card."""
        super().__init__(card_id)
        self.value = value
        self.label = label
        self.variant = variant
        self.icon = icon
        self.trend = None
        self.change_percentage = None

    def set_trend(self, trend: str, percentage: float):
        """Set trend indicator (up, down, neutral)."""
        self.trend = trend
        self.change_percentage = percentage
        return self

    def render(self) -> str:
        """Render metric card HTML."""
        self.add_class("metric-card")
        self.add_class(f"metric-{self.variant}")

        attrs = self._build_attributes()

        icon_html = f'<div class="metric-icon">{self.icon}</div>' if self.icon else ""

        trend_html = ""
        if self.trend and self.change_percentage is not None:
            trend_icon = {
                'up': '^',
                'down': 'v',
                'neutral': '->'
            }.get(self.trend, '->')

            trend_class = f"trend-{self.trend}"
            trend_html = f'<div class="metric-trend {trend_class}">{trend_icon} {abs(self.change_percentage):.1f}%</div>'

        return f'''
<div{attrs}>
    {icon_html}
    <div class="metric-value">{self.value}</div>
    <div class="metric-label">{self.label}</div>
    {trend_html}
</div>
        '''


class Table(UIComponent):
    """Data table component."""

    def __init__(self, table_id: str, columns: List[str]):
        """Initialize table."""
        super().__init__(table_id)
        self.columns = columns
        self.rows = []
        self.striped = True
        self.hoverable = True
        self.sortable = False

    def add_row(self, row: List[Any]) -> 'Table':
        """Add row to table."""
        self.rows.append(row)
        return self

    def add_rows(self, rows: List[List[Any]]) -> 'Table':
        """Add multiple rows."""
        self.rows.extend(rows)
        return self

    def set_striped(self, striped: bool):
        """Set striped rows."""
        self.striped = striped
        return self

    def set_hoverable(self, hoverable: bool):
        """Set hover effect."""
        self.hoverable = hoverable
        return self

    def set_sortable(self, sortable: bool):
        """Set sortable columns."""
        self.sortable = sortable
        return self

    def render(self) -> str:
        """Render table HTML."""
        self.add_class("table")

        if self.striped:
            self.add_class("table-striped")

        if self.hoverable:
            self.add_class("table-hover")

        attrs = self._build_attributes()

        # Header
        header_html = "<thead><tr>"
        for col in self.columns:
            sortable_attr = f' data-sortable="true"' if self.sortable else ''
            header_html += f'<th{sortable_attr}>{col}</th>'
        header_html += "</tr></thead>"

        # Body
        body_html = "<tbody>"
        for row in self.rows:
            body_html += "<tr>"
            for cell in row:
                body_html += f'<td>{cell}</td>'
            body_html += "</tr>"
        body_html += "</tbody>"

        return f'<table{attrs}>{header_html}{body_html}</table>'


class ComponentLibrary:
    """
    Library of pre-built component templates.

    Provides ready-to-use components for common dashboard patterns.
    """

    @staticmethod
    def create_progress_card(
        card_id: str,
        title: str,
        progress: float,
        total: float
    ) -> Card:
        """Create progress tracking card."""
        card = Card(card_id, title)

        percentage = (progress / total * 100) if total > 0 else 0

        card.add_content(f'''
            <div class="progress-summary">
                <div class="progress-text">
                    <span class="progress-value">{progress}</span>
                    <span class="progress-divider">/</span>
                    <span class="progress-total">{total}</span>
                </div>
                <div class="progress-percentage">{percentage:.1f}%</div>
            </div>
        ''')

        bar = ProgressBar(f"{card_id}_bar", progress, total)
        card.add_content(bar.render())

        return card

    @staticmethod
    def create_status_card(
        card_id: str,
        title: str,
        status: str,
        count: int
    ) -> Card:
        """Create status summary card."""
        variant_map = {
            'completed': 'success',
            'in_progress': 'primary',
            'failed': 'danger',
            'blocked': 'warning',
            'pending': 'secondary'
        }

        variant = variant_map.get(status, 'secondary')

        card = Card(card_id)
        card.set_variant(variant)

        card.add_content(f'''
            <div class="status-summary">
                <div class="status-count">{count}</div>
                <div class="status-label">{title}</div>
                <div class="status-badge">{status.replace('_', ' ').title()}</div>
            </div>
        ''')

        return card

    @staticmethod
    def create_metrics_row(metrics: List[Dict[str, Any]]) -> str:
        """Create row of metric cards."""
        cards_html = '<div class="metrics-row">'

        for metric in metrics:
            card = MetricCard(
                metric.get('id', f'metric_{len(cards_html)}'),
                metric['value'],
                metric['label'],
                metric.get('variant', 'primary'),
                metric.get('icon')
            )

            if 'trend' in metric and 'change' in metric:
                card.set_trend(metric['trend'], metric['change'])

            cards_html += card.render()

        cards_html += '</div>'

        return cards_html


class CSSFramework:
    """
    CSS framework for component styling.

    Provides responsive, themeable styles for all components.
    """

    @staticmethod
    def get_styles(theme: str = "light") -> str:
        """Get CSS framework styles."""
        return f"""
/* {theme.upper()} Theme Variables */
:root {{
    --primary-color: #3498db;
    --secondary-color: #2c3e50;
    --success-color: #27ae60;
    --danger-color: #e74c3c;
    --warning-color: #f39c12;
    --info-color: #16a085;
    --light-color: #ecf0f1;
    --dark-color: #2c3e50;
    --border-color: #ddd;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
    --shadow-lg: 0 10px 25px rgba(0,0,0,0.15);
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
}}

.theme-dark {{
    --primary-color: #5dade2;
    --secondary-color: #34495e;
    --light-color: #2c3e50;
    --dark-color: #ecf0f1;
    --border-color: #555;
}}

/* Base Styles */
* {{
    box-sizing: border-box;
}}

body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--dark-color);
    background: #f5f7fa;
    margin: 0;
    padding: 0;
}}

/* Button Component */
.btn {{
    display: inline-block;
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-md);
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
}}

.btn:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}}

.btn-primary {{ background: var(--primary-color); color: white; }}
.btn-secondary {{ background: var(--secondary-color); color: white; }}
.btn-success {{ background: var(--success-color); color: white; }}
.btn-danger {{ background: var(--danger-color); color: white; }}
.btn-warning {{ background: var(--warning-color); color: white; }}

.btn-sm {{ padding: 6px 12px; font-size: 12px; }}
.btn-lg {{ padding: 14px 28px; font-size: 16px; }}
.btn.disabled {{ opacity: 0.6; cursor: not-allowed; }}

/* Card Component */
.card {{
    background: white;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    overflow: hidden;
    margin-bottom: 20px;
}}

.card-header {{
    padding: 20px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.card-title {{
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--secondary-color);
}}

.card-body {{
    padding: 20px;
}}

/* Metric Card */
.metric-card {{
    background: white;
    padding: 25px;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
    text-align: center;
    transition: transform 0.3s;
}}

.metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}}

.metric-icon {{ font-size: 40px; margin-bottom: 10px; }}
.metric-value {{ font-size: 36px; font-weight: bold; color: var(--primary-color); margin-bottom: 5px; }}
.metric-label {{ font-size: 14px; color: #6c757d; margin-bottom: 10px; }}
.metric-trend {{ font-size: 12px; font-weight: 600; }}
.trend-up {{ color: var(--success-color); }}
.trend-down {{ color: var(--danger-color); }}
.trend-neutral {{ color: var(--info-color); }}

/* Progress Bar */
.progress {{ margin-bottom: 20px; }}
.progress-label {{ display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: 500; }}
.progress-bar-wrapper {{ height: 25px; background: var(--light-color); border-radius: 12px; overflow: hidden; }}
.progress-bar {{
    height: 100%;
    background: var(--primary-color);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
    transition: width 0.5s ease;
}}

/* Badge */
.badge {{
    display: inline-block;
    padding: 4px 8px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    color: white;
}}

/* Table */
.table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}}

.table thead {{ background: var(--secondary-color); color: white; }}
.table th, .table td {{ padding: 12px 15px; text-align: left; }}
.table-striped tbody tr:nth-of-type(odd) {{ background: var(--light-color); }}
.table-hover tbody tr:hover {{ background: rgba(52, 152, 219, 0.1); }}

/* Responsive */
@media (max-width: 768px) {{
    .metrics-row {{ grid-template-columns: 1fr; }}
    .card {{ margin-bottom: 15px; }}
    .table {{ font-size: 12px; }}
}}
        """


class JavaScriptFramework:
    """
    JavaScript framework for component interactions.

    Provides interactive behaviors for components.
    """

    @staticmethod
    def get_scripts() -> str:
        """Get JavaScript framework code."""
        return """
// Component initialization
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    initializeModals();
    initializeAlerts();
});

// Tabs functionality
function initializeTabs() {
    document.querySelectorAll('.tabs-item a').forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            const tabId = this.getAttribute('data-tab');

            // Remove active from all tabs
            document.querySelectorAll('.tabs-item').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tabs-pane').forEach(p => p.classList.remove('active'));

            // Add active to clicked tab
            this.parentElement.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        });
    });
}

// Modal functionality
function initializeModals() {
    // Open modals
    document.querySelectorAll('[data-toggle="modal"]').forEach(trigger => {
        trigger.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Close modals
    document.querySelectorAll('[data-dismiss="modal"]').forEach(trigger => {
        trigger.addEventListener('click', function() {
            this.closest('.modal').classList.remove('active');
        });
    });

    // Close on escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.active').forEach(modal => {
                modal.classList.remove('active');
            });
        }
    });
}

// Alert functionality
function initializeAlerts() {
    document.querySelectorAll('.alert-close').forEach(btn => {
        btn.addEventListener('click', function() {
            this.closest('.alert').style.display = 'none';
        });
    });
}
        """
