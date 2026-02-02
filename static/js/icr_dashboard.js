/**
 * ICR Analytics Dashboard JavaScript
 * Handles all dashboard functionality including data fetching, visualization, and interactivity
 */

// Global chart instances
let patternDistributionChart = null;
let successRateTrendChart = null;
let contentTypeChart = null;
let qualityLevelChart = null;
let complexityChart = null;
let vlmProviderChart = null;
let heatmapChart = null;

// API base URL
const API_BASE = window.location.origin;

// Dashboard state
let dashboardState = {
    lastUpdated: null,
    refreshInterval: 30000, // 30 seconds
    autoRefresh: true
};

/**
 * Initialize the dashboard
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadAllData();
    setupEventListeners();
    startAutoRefresh();
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
    document.getElementById('refreshAllBtn').addEventListener('click', loadAllData);
    document.getElementById('heatmapRefreshBtn').addEventListener('click', loadHeatmapData);
    document.getElementById('refinementRefreshBtn').addEventListener('click', loadRefinementEvents);
}

/**
 * Load all dashboard data
 */
async function loadAllData() {
    try {
        await Promise.all([
            loadOverviewData(),
            loadComponentData(),
            loadPatternAnalysis(),
            loadVLMAnalytics(),
            loadRefinementEvents(),
            loadConfiguration()
        ]);
        updateLastUpdated();
        showNotification('Data refreshed successfully', 'success');
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Error loading data: ' + error.message, 'error');
    }
}

/**
 * Load overview statistics
 */
async function loadOverviewData() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/overview`);
        if (!response.ok) throw new Error('Failed to load overview data');
        
        const data = await response.json();
        
        // Update overview cards
        document.getElementById('totalPatterns').textContent = data.total_patterns || 0;
        document.getElementById('overallSuccessRate').textContent = 
            (data.overall_success_rate ? (data.overall_success_rate * 100).toFixed(1) : '0') + '%';
        document.getElementById('activeComponents').textContent = data.active_components || 0;
        document.getElementById('totalRefinements').textContent = data.total_refinements || 0;
        
        // Update ICR status badge
        const statusBadge = document.getElementById('icrStatusBadge');
        if (data.icr_enabled) {
            statusBadge.className = 'badge bg-success me-3';
            statusBadge.innerHTML = '<i class="bi bi-check-circle me-1"></i> ICR Enabled';
        } else {
            statusBadge.className = 'badge bg-secondary me-3';
            statusBadge.innerHTML = '<i class="bi bi-dash-circle me-1"></i> ICR Disabled';
        }
    } catch (error) {
        console.error('Error loading overview data:', error);
        // Set default values
        document.getElementById('totalPatterns').textContent = '0';
        document.getElementById('overallSuccessRate').textContent = '0%';
        document.getElementById('activeComponents').textContent = '0';
        document.getElementById('totalRefinements').textContent = '0';
    }
}

/**
 * Load component statistics
 */
async function loadComponentData() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/components`);
        if (!response.ok) throw new Error('Failed to load component data');
        
        const data = await response.json();
        
        // Update QualityGateEngine
        updateComponentCard('qg', data.quality_gate_engine);
        
        // Update SGDWorkflowOrchestrator
        updateComponentCard('sgd', data.workflow_orchestrator);
        
        // Update RobustnessCoordinator
        updateComponentCard('rob', data.robustness_coordinator);
        
        // Update BubbleLab
        updateComponentCard('bl', data.bubblelab);
        
        // Update ROMA
        updateComponentCard('roma', data.roma);
        
    } catch (error) {
        console.error('Error loading component data:', error);
    }
}

/**
 * Update a component card
 */
function updateComponentCard(prefix, componentData) {
    if (!componentData) {
        document.getElementById(`${prefix}Patterns`).textContent = '0';
        document.getElementById(`${prefix}PassRate`).textContent = '0%';
        document.getElementById(`${prefix}Quality`).textContent = '0.0';
        document.getElementById(`${prefix}Status`).className = 'badge bg-secondary';
        document.getElementById(`${prefix}Status`).textContent = 'Inactive';
        return;
    }
    
    document.getElementById(`${prefix}Patterns`).textContent = componentData.total_patterns || 0;
    document.getElementById(`${prefix}PassRate`).textContent = 
        (componentData.overall_pass_rate ? (componentData.overall_pass_rate * 100).toFixed(1) : '0') + '%';
    document.getElementById(`${prefix}Quality`).textContent = 
        (componentData.overall_quality || 0).toFixed(2);
    
    const statusBadge = document.getElementById(`${prefix}Status`);
    if (componentData.active) {
        statusBadge.className = 'badge bg-success';
        statusBadge.textContent = 'Active';
    } else {
        statusBadge.className = 'badge bg-secondary';
        statusBadge.textContent = 'Inactive';
    }
}

/**
 * Load pattern analysis data
 */
async function loadPatternAnalysis() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/patterns`);
        if (!response.ok) throw new Error('Failed to load pattern data');
        
        const data = await response.json();
        
        // Update pattern distribution chart
        updatePatternDistributionChart(data.pattern_types);
        
        // Update success rate trend chart
        updateSuccessRateTrendChart(data.trends);
        
        // Update content type chart
        updateContentTypeChart(data.by_content_type);
        
        // Update quality level chart
        updateQualityLevelChart(data.by_quality_level);
        
        // Update complexity chart
        updateComplexityChart(data.by_complexity);
        
    } catch (error) {
        console.error('Error loading pattern analysis:', error);
    }
}

/**
 * Load VLM analytics
 */
async function loadVLMAnalytics() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/vlm`);
        if (!response.ok) throw new Error('Failed to load VLM data');
        
        const data = await response.json();
        
        // Update VLM stats
        document.getElementById('vlmTotalAnalyses').textContent = data.total_analyses || 0;
        document.getElementById('vlmTokensUsed').textContent = formatNumber(data.total_tokens || 0);
        document.getElementById('vlmAvgConfidence').textContent = 
            (data.avg_confidence ? (data.avg_confidence * 100).toFixed(1) : '0') + '%';
        document.getElementById('vlmCached').textContent = 
            (data.cache_hit_rate ? (data.cache_hit_rate * 100).toFixed(1) : '0') + '%';
        
        // Update VLM status
        const vlmStatus = document.getElementById('vlmStatus');
        if (data.available && data.enabled) {
            vlmStatus.className = 'badge bg-success';
            vlmStatus.textContent = 'Active';
        } else if (data.available && !data.enabled) {
            vlmStatus.className = 'badge bg-warning';
            vlmStatus.textContent = 'Disabled';
        } else {
            vlmStatus.className = 'badge bg-danger';
            vlmStatus.textContent = 'Not Configured';
        }
        
        // Update VLM provider chart
        updateVLMProviderChart(data.by_provider);
        
        // Update VLM configuration display
        updateVLMConfigDisplay(data.config);
        
    } catch (error) {
        console.error('Error loading VLM analytics:', error);
    }
}

/**
 * Update VLM configuration display
 */
function updateVLMConfigDisplay(config) {
    const configDiv = document.getElementById('vlmConfig');
    
    if (!config) {
        configDiv.innerHTML = `
            <div class="text-center text-muted">
                <i class="bi bi-gear fs-1 mb-2"></i>
                <p>VLM not configured</p>
            </div>
        `;
        return;
    }
    
    configDiv.innerHTML = `
        <div class="config-item">
            <div class="config-label">Provider</div>
            <div class="config-value">${config.provider || 'N/A'}</div>
        </div>
        <div class="config-item">
            <div class="config-label">Model</div>
            <div class="config-value">${config.model || 'N/A'}</div>
        </div>
        <div class="config-item">
            <div class="config-label">Temperature</div>
            <div class="config-value">${config.temperature || 'N/A'}</div>
        </div>
        <div class="config-item">
            <div class="config-label">Max Tokens</div>
            <div class="config-value">${config.max_tokens || 'N/A'}</div>
        </div>
        <div class="config-item">
            <div class="config-label">Caching</div>
            <div class="config-value ${config.enable_caching ? 'enabled' : 'disabled'}">
                ${config.enable_caching ? 'Enabled' : 'Disabled'}
            </div>
        </div>
    `;
}

/**
 * Load refinement events
 */
async function loadRefinementEvents() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/refinements?limit=10`);
        if (!response.ok) throw new Error('Failed to load refinement events');
        
        const data = await response.json();
        
        // Update refinement table
        updateRefinementTable(data.events || []);
        
    } catch (error) {
        console.error('Error loading refinement events:', error);
    }
}

/**
 * Update refinement table
 */
function updateRefinementTable(events) {
    const tbody = document.getElementById('refinementTableBody');
    
    if (!events || events.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">No refinement events found</td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = events.map(event => `
        <tr>
            <td>${formatTimestamp(event.timestamp)}</td>
            <td>${event.component || 'N/A'}</td>
            <td><span class="badge bg-info">${formatRefinementType(event.refinement_type)}</span></td>
            <td class="text-truncate-2" style="max-width: 200px;">${event.reason || 'N/A'}</td>
            <td>
                <span class="badge ${event.success ? 'bg-success' : 'bg-danger'}">
                    ${event.success ? 'Success' : 'Failed'}
                </span>
            </td>
            <td>${(event.confidence * 100).toFixed(1)}%</td>
        </tr>
    `).join('');
}

/**
 * Load heatmap data
 */
async function loadHeatmapData() {
    try {
        const response = await fetch(`${API_BASE}/icr/analytics/heatmap`);
        if (!response.ok) throw new Error('Failed to load heatmap data');
        
        const data = await response.json();
        
        // Update heatmap visualization
        updateHeatmap(data);
        
    } catch (error) {
        console.error('Error loading heatmap data:', error);
    }
}

/**
 * Load configuration
 */
async function loadConfiguration() {
    try {
        const response = await fetch(`${API_BASE}/icr/config`);
        if (!response.ok) throw new Error('Failed to load configuration');
        
        const data = await response.json();
        
        // Update configuration display
        updateConfigurationDisplay(data);
        
    } catch (error) {
        console.error('Error loading configuration:', error);
    }
}

/**
 * Update configuration display
 */
function updateConfigurationDisplay(config) {
    const configDiv = document.getElementById('icrConfig');
    
    if (!config) {
        configDiv.innerHTML = `
            <div class="text-center text-muted">
                <i class="bi bi-gear fs-1 mb-2"></i>
                <p>Configuration not available</p>
            </div>
        `;
        return;
    }
    
    configDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="config-item">
                    <div class="config-label">ICR Enabled</div>
                    <div class="config-value ${config.enabled ? 'enabled' : 'disabled'}">
                        ${config.enabled ? 'Yes' : 'No'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Prediction</div>
                    <div class="config-value ${config.enable_prediction ? 'enabled' : 'disabled'}">
                        ${config.enable_prediction ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Learning</div>
                    <div class="config-value ${config.enable_learning ? 'enabled' : 'disabled'}">
                        ${config.enable_learning ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="config-item">
                    <div class="config-label">Quality Gate</div>
                    <div class="config-value ${config.quality_gate_enabled ? 'enabled' : 'disabled'}">
                        ${config.quality_gate_enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Workflow Orchestrator</div>
                    <div class="config-value ${config.workflow_orchestrator_enabled ? 'enabled' : 'disabled'}">
                        ${config.workflow_orchestrator_enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Robustness</div>
                    <div class="config-value ${config.robustness_enabled ? 'enabled' : 'disabled'}">
                        ${config.robustness_enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">ROMA Modules</div>
                    <div class="config-value ${config.roma_modules_enabled ? 'enabled' : 'disabled'}">
                        ${config.roma_modules_enabled ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Initialize all charts
 */
function initializeCharts() {
    // Pattern Distribution Chart
    const patternCtx = document.getElementById('patternDistributionChart').getContext('2d');
    patternDistributionChart = new Chart(patternCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#4a6fa5', '#28a745', '#17a2b8', '#ffc107', '#dc3545',
                    '#6c757d', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Success Rate Trend Chart
    const trendCtx = document.getElementById('successRateTrendChart').getContext('2d');
    successRateTrendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Success Rate',
                data: [],
                borderColor: '#4a6fa5',
                backgroundColor: 'rgba(74, 111, 165, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Content Type Chart
    const contentTypeCtx = document.getElementById('contentTypeChart').getContext('2d');
    contentTypeChart = new Chart(contentTypeCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Count',
                data: [],
                backgroundColor: '#4a6fa5'
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Quality Level Chart
    const qualityCtx = document.getElementById('qualityLevelChart').getContext('2d');
    qualityLevelChart = new Chart(qualityCtx, {
        type: 'pie',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: ['#28a745', '#17a2b8', '#ffc107']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Complexity Chart
    const complexityCtx = document.getElementById('complexityChart').getContext('2d');
    complexityChart = new Chart(complexityCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Count',
                data: [],
                backgroundColor: '#17a2b8'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // VLM Provider Chart
    const vlmCtx = document.getElementById('vlmProviderChart').getContext('2d');
    vlmProviderChart = new Chart(vlmCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Analyses',
                data: [],
                backgroundColor: ['#4a6fa5', '#28a745', '#ffc107', '#dc3545']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Update pattern distribution chart
 */
function updatePatternDistributionChart(data) {
    if (!data) return;
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    patternDistributionChart.data.labels = labels;
    patternDistributionChart.data.datasets[0].data = values;
    patternDistributionChart.update();
}

/**
 * Update success rate trend chart
 */
function updateSuccessRateTrendChart(data) {
    if (!data || !data.timestamps || !data.values) return;
    
    successRateTrendChart.data.labels = data.timestamps.map(ts => formatTimestamp(ts));
    successRateTrendChart.data.datasets[0].data = data.values;
    successRateTrendChart.update();
}

/**
 * Update content type chart
 */
function updateContentTypeChart(data) {
    if (!data) return;
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    contentTypeChart.data.labels = labels;
    contentTypeChart.data.datasets[0].data = values;
    contentTypeChart.update();
}

/**
 * Update quality level chart
 */
function updateQualityLevelChart(data) {
    if (!data) return;
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    qualityLevelChart.data.labels = labels;
    qualityLevelChart.data.datasets[0].data = values;
    qualityLevelChart.update();
}

/**
 * Update complexity chart
 */
function updateComplexityChart(data) {
    if (!data) return;
    
    const labels = Object.keys(data).sort((a, b) => parseInt(a) - parseInt(b));
    const values = labels.map(key => data[key]);
    
    complexityChart.data.labels = labels;
    complexityChart.data.datasets[0].data = values;
    complexityChart.update();
}

/**
 * Update VLM provider chart
 */
function updateVLMProviderChart(data) {
    if (!data) return;
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    vlmProviderChart.data.labels = labels;
    vlmProviderChart.data.datasets[0].data = values;
    vlmProviderChart.update();
}

/**
 * Update heatmap visualization
 */
function updateHeatmap(data) {
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = canvas.parentElement.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!data || !data.points || data.points.length === 0) {
        ctx.fillStyle = '#e9ecef';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6c757d';
        ctx.font = '16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No heatmap data available', canvas.width / 2, canvas.height / 2);
        return;
    }
    
    // Draw heatmap points
    data.points.forEach(point => {
        const x = point.x * canvas.width;
        const y = point.y * canvas.height;
        const intensity = point.intensity || 0.5;
        
        // Create radial gradient for each point
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 30);
        gradient.addColorStop(0, `rgba(74, 111, 165, ${intensity})`);
        gradient.addColorStop(1, 'rgba(74, 111, 165, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, 30, 0, Math.PI * 2);
        ctx.fill();
    });
}

/**
 * Start auto-refresh
 */
function startAutoRefresh() {
    if (dashboardState.autoRefresh) {
        setInterval(loadAllData, dashboardState.refreshInterval);
    }
}

/**
 * Update last updated timestamp
 */
function updateLastUpdated() {
    const now = new Date();
    document.getElementById('lastUpdated').textContent = now.toLocaleString();
    dashboardState.lastUpdated = now;
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    const toast = document.getElementById('notificationToast');
    const icon = document.getElementById('toastIcon');
    const title = document.getElementById('toastTitle');
    const body = document.getElementById('toastBody');
    
    // Set icon based on type
    switch (type) {
        case 'success':
            icon.className = 'bi bi-check-circle me-2 text-success';
            title.textContent = 'Success';
            break;
        case 'error':
            icon.className = 'bi bi-exclamation-circle me-2 text-danger';
            title.textContent = 'Error';
            break;
        case 'warning':
            icon.className = 'bi bi-exclamation-triangle me-2 text-warning';
            title.textContent = 'Warning';
            break;
        default:
            icon.className = 'bi bi-info-circle me-2 text-info';
            title.textContent = 'Info';
    }
    
    body.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Format refinement type
 */
function formatRefinementType(type) {
    if (!type) return 'N/A';
    
    // Convert snake_case to Title Case
    return type.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Format number with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
