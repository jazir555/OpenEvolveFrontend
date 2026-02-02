# ICR Analytics Dashboard - Integration Guide

## Overview

The ICR (Iterative Contextual Refinements) Analytics Dashboard provides a comprehensive visual interface for monitoring ICR performance across all integrated components. The dashboard displays real-time statistics, pattern analysis, VLM analytics, and refinement events.

## Dashboard Files

### Frontend Files
- `templates/icr_dashboard.html` - Main dashboard HTML template
- `static/css/icr_dashboard.css` - Dashboard styling
- `static/js/icr_dashboard.js` - Dashboard JavaScript with Chart.js visualizations

### Backend Files
- `api_server.py` - Contains all ICR analytics API endpoints

## Accessing the Dashboard

### Starting the API Server

```bash
# Start the FastAPI server
python api_server.py
```

The server will start on `http://localhost:8001` by default.

### Dashboard URL

Navigate to:
```
http://localhost:8001/icr/dashboard
```

## Dashboard Sections

### 1. Overview Section

Displays key metrics:
- **Total Patterns** - Total number of patterns stored across all components
- **Overall Success Rate** - Combined pass/fail rate across all components
- **Active Components** - Number of components with ICR enabled
- **Total Refinements** - Total number of refinements applied

### 2. Component Statistics

Breakdown by component:
- **QualityGateEngine** - Quality gate pattern statistics
- **SGDWorkflowOrchestrator** - Workflow orchestrator statistics
- **RobustnessCoordinator** - Robustness integration statistics
- **BubbleLab** - BubbleLab node statistics
- **ROMA Components** - ROMA module statistics

Each component shows:
- Pattern count
- Pass rate
- Quality score
- Active/inactive status

### 3. Pattern Analysis

Visualizations of pattern data:
- **Pattern Distribution** - Doughnut chart of pattern types
- **Success Rate Trend** - Line chart showing success rate over time
- **By Content Type** - Horizontal bar chart of content type distribution
- **By Quality Level** - Pie chart of quality level distribution
- **By Complexity** - Bar chart of complexity distribution (1-10)

### 4. VLM Analytics

Vision Language Model analysis statistics:
- **Total Analyses** - Number of VLM analyses performed
- **Tokens Used** - Total tokens consumed by VLM
- **Avg Confidence** - Average confidence score of VLM analyses
- **Cache Hit Rate** - Percentage of cached responses
- **Provider Performance** - Bar chart of analyses by provider
- **Configuration** - Current VLM configuration display

### 5. Heatmap Visualization

Interactive heatmap of ICR pattern intensity:
- Visual representation of pattern activity
- Color gradient from low (light) to high (dark) intensity
- Refresh button to update heatmap data

### 6. Refinement Events

Recent refinement events table showing:
- **Timestamp** - When the refinement occurred
- **Component** - Which component triggered the refinement
- **Type** - Type of refinement (e.g., threshold_adjustment)
- **Reason** - Why the refinement was needed
- **Success** - Whether the refinement was successful
- **Confidence** - Confidence score of the refinement

### 7. Configuration Status

Current ICR configuration:
- ICR enabled status
- Prediction and learning flags
- Component-specific enablement

## API Endpoints

### Dashboard Route
```
GET /icr/dashboard
```
Returns the dashboard HTML page.

### Analytics Endpoints

#### Overview Statistics
```
GET /icr/analytics/overview
```
Returns:
```json
{
  "icr_enabled": true,
  "total_patterns": 1234,
  "overall_success_rate": 0.85,
  "active_components": 5,
  "total_refinements": 42
}
```

#### Component Statistics
```
GET /icr/analytics/components
```
Returns statistics for each component:
```json
{
  "quality_gate_engine": {
    "active": true,
    "total_patterns": 234,
    "overall_pass_rate": 0.88,
    "overall_quality": 0.82
  },
  "workflow_orchestrator": { ... },
  "robustness_coordinator": { ... },
  "bubblelab": { ... },
  "roma": { ... }
}
```

#### Pattern Analysis
```
GET /icr/analytics/patterns
```
Returns pattern distribution and trends:
```json
{
  "pattern_types": {
    "content_type": 45,
    "quality_level": 38,
    "metric": 67,
    ...
  },
  "trends": {
    "timestamps": ["2026-02-01T10:00:00", ...],
    "values": [0.85, 0.87, 0.86, ...]
  },
  "by_content_type": { "code": 120, "text": 45, ... },
  "by_quality_level": { "standard": 80, "high": 60, ... },
  "by_complexity": { "1": 10, "2": 15, ... }
}
```

#### VLM Analytics
```
GET /icr/analytics/vlm
```
Returns VLM statistics:
```json
{
  "available": true,
  "enabled": true,
  "total_analyses": 156,
  "total_tokens": 45678,
  "avg_confidence": 0.92,
  "cache_hit_rate": 0.35,
  "by_provider": {
    "openai": 120,
    "anthropic": 36
  },
  "config": { ... }
}
```

#### Refinement Events
```
GET /icr/analytics/refinements?limit=10
```
Returns recent refinement events:
```json
{
  "events": [
    {
      "event_id": "uuid",
      "timestamp": "2026-02-02T00:00:00",
      "refinement_type": "threshold_adjustment",
      "component": "quality_gate_engine",
      "reason": "Low pass rate for code content",
      "success": true,
      "confidence": 0.85
    },
    ...
  ],
  "total_count": 42
}
```

#### Heatmap Data
```
GET /icr/analytics/heatmap
```
Returns heatmap points:
```json
{
  "points": [
    {
      "x": 0.5,
      "y": 0.3,
      "intensity": 0.8
    },
    ...
  ],
  "total_snapshots": 15
}
```

#### Configuration
```
GET /icr/config
```
Returns current ICR configuration:
```json
{
  "enabled": true,
  "enable_prediction": true,
  "enable_learning": true,
  "quality_gate_enabled": true,
  "workflow_orchestrator_enabled": true,
  "gauntlet_system_enabled": true,
  "robustness_enabled": true,
  "roma_modules_enabled": true
}
```

## Component Integration

To integrate a component with the ICR dashboard, use the helper functions in `api_server.py`:

### Update Component Statistics

```python
from api_server import update_icr_component_stats

update_icr_component_stats(
    component_name="quality_gate_engine",
    stats={
        "total_patterns": 100,
        "overall_pass_rate": 0.85,
        "overall_quality": 0.82,
        "active": True
    }
)
```

### Record Pattern Data

```python
from api_server import update_icr_pattern_data

update_icr_pattern_data(
    pattern_type="content_type",
    content_type="code",
    quality_level="high",
    complexity=7
)
```

### Record VLM Analysis

```python
from api_server import update_icr_vlm_stats

update_icr_vlm_stats(
    provider="openai",
    tokens_used=1024,
    confidence=0.92,
    cached=False
)
```

### Record Refinement Event

```python
from api_server import record_icr_refinement

record_icr_refinement(
    refinement_type="threshold_adjustment",
    component="quality_gate_engine",
    reason="Low pass rate detected",
    success=True,
    confidence=0.85
)
```

## Dashboard Features

### Auto-Refresh
The dashboard automatically refreshes every 30 seconds. This can be configured in `static/js/icr_dashboard.js`:

```javascript
let dashboardState = {
    refreshInterval: 30000,  // 30 seconds
    autoRefresh: true
};
```

### Manual Refresh
Click the "Refresh All" button in the navigation bar to manually refresh all data.

### Toast Notifications
The dashboard shows toast notifications for:
- Successful data refresh
- Error messages
- Warning messages

### Responsive Design
The dashboard is fully responsive and works on:
- Desktop (1920x1080 and above)
- Tablet (768x1024)
- Mobile (375x667 and above)

## Visualizations

All charts use Chart.js for rendering:
- **Doughnut Chart** - Pattern distribution
- **Line Chart** - Success rate trends
- **Bar Chart** - Content type, complexity, provider performance
- **Pie Chart** - Quality level distribution

Charts are interactive:
- Hover for tooltips
- Click for details
- Responsive resizing

## Customization

### Styling
Modify `static/css/icr_dashboard.css` to customize:
- Colors
- Layout
- Card styles
- Chart themes

### Data Sources
The dashboard uses in-memory data storage by default. For production:
1. Replace `ICR_ANALYTICS_DATA` with database storage
2. Update helper functions to persist data
3. Add authentication to API endpoints

### Adding New Visualizations
1. Add canvas element to `templates/icr_dashboard.html`
2. Initialize chart in `static/js/icr_dashboard.js`
3. Create API endpoint in `api_server.py`
4. Fetch data in JavaScript and update chart

## Troubleshooting

### Dashboard Not Loading
1. Check API server is running: `http://localhost:8001`
2. Check browser console for errors
3. Verify template file exists: `templates/icr_dashboard.html`

### Charts Not Displaying
1. Verify Chart.js CDN is accessible
2. Check browser console for JavaScript errors
3. Ensure data is being returned from API endpoints

### Data Not Updating
1. Check helper functions are being called
2. Verify `ICR_ANALYTICS_DATA` is being updated
3. Check browser network tab for API requests

### VLM Section Shows "Not Configured"
1. Set environment variable: `ICR_VLM_ENABLED=1`
2. Configure VLM provider: `ICR_VLM_PROVIDER=openai`
3. Set API key: `ICR_VLM_API_KEY=your-key`

## Security Considerations

For production deployment:
1. Add authentication to dashboard endpoints
2. Use HTTPS instead of HTTP
3. Implement rate limiting
4. Add CSRF protection
5. Sanitize user inputs

## Performance Optimization

To improve dashboard performance:
1. Enable data caching
2. Use pagination for large datasets
3. Implement WebSocket for real-time updates
4. Optimize chart rendering
5. Use CDN for static assets

## Future Enhancements

Potential improvements:
1. Real-time WebSocket updates
2. Export data to CSV/JSON
3. Custom date range filtering
4. Advanced pattern search
5. Component comparison views
6. Alert thresholds and notifications
7. Historical data comparison
8. Custom dashboard layouts

## Support

For issues or questions:
1. Check browser console for errors
2. Review API server logs
3. Verify environment configuration
4. Check component integration code
