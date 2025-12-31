/**
 * Sovereign Decomposition System Visualization
 * D3.js based visualization for problem decomposition graphs
 */

class SovereignVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = null;
        this.width = 800;
        this.height = 500;
        this.simulation = null;
        
        this.init();
    }
    
    init() {
        // Set up SVG container
        const container = d3.select(`#${this.containerId}`);
        container.html(''); // Clear existing content
        
        this.svg = container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height])
            .attr('style', 'max-width: 100%; height: auto; font: 12px sans-serif;');
        
        // Set up zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 8])
            .on('zoom', (event) => {
                this.svg.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create groups for links and nodes
        this.linkGroup = this.svg.append('g')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6);
        
        this.nodeGroup = this.svg.append('g')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5);
    }
    
    update(plan) {
        if (!plan || !plan.dependency_graph) {
            this.svg.selectAll('*').remove();
            // Show message if no plan
            this.svg.append('text')
                .attr('x', this.width / 2)
                .attr('y', this.height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#6c757d')
                .text('No decomposition plan available for visualization');
            return;
        }
        
        const nodes = this.convertSubProblemsToNodes(plan.sub_problems);
        const links = this.extractDependencies(plan.sub_problems);
        
        if (!nodes || nodes.length === 0) {
            this.svg.selectAll('*').remove();
            this.svg.append('text')
                .attr('x', this.width / 2)
                .attr('y', this.height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#6c757d')
                .text('No sub-problems to visualize');
            return;
        }
        
        // Create simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(40));
        
        // Update links
        const link = this.linkGroup.selectAll('line')
            .data(links, d => [d.source, d.target]);
        
        link.exit().remove();
        
        const linkEnter = link.enter()
            .append('line')
            .attr('stroke-width', 2)
            .attr('stroke', '#6c757d');
        
        this.link = link.merge(linkEnter);
        
        // Update nodes
        const node = this.nodeGroup.selectAll('g')
            .data(nodes, d => d.id);
        
        // Remove old nodes
        node.exit().remove();
        
        // Add new nodes
        const nodeEnter = node.enter()
            .append('g')
            .attr('cursor', 'pointer')
            .on('click', (event, d) => {
                this.onNodeClick(d);
            });
        
        // Add circles to nodes
        nodeEnter.append('circle')
            .attr('r', 20)
            .attr('fill', d => this.getNodeColor(d));
        
        // Add labels to nodes
        nodeEnter.append('text')
            .attr('x', 0)
            .attr('y', 5)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', '#fff')
            .attr('font-size', '10px')
            .text(d => d.title.substring(0, 12) + (d.title.length > 12 ? '...' : ''));
        
        // Combine old and new nodes
        this.node = node.merge(nodeEnter);
        
        // Update simulation
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(links);
        
        // Update positions when simulation ticks
        this.simulation.on('tick', () => {
            this.link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            this.node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Add drag behavior
        this.node.call(d3.drag()
            .on('start', this.dragstarted.bind(this))
            .on('drag', this.dragged.bind(this))
            .on('end', this.dragended.bind(this)));
    }
    
    convertSubProblemsToNodes(subProblems) {
        if (!subProblems) return [];
        
        return subProblems.map(sp => ({
            id: sp.id,
            title: sp.title || 'Untitled',
            type: sp.type || 'general',
            complexity: sp.complexity_score?.overall_complexity || 5,
            status: sp.status || 'pending',
            priority: sp.priority || 5
        }));
    }
    
    extractDependencies(subProblems) {
        if (!subProblems) return [];
        
        const links = [];
        subProblems.forEach(sp => {
            if (sp.dependencies && Array.isArray(sp.dependencies)) {
                sp.dependencies.forEach(depId => {
                    links.push({
                        source: depId,
                        target: sp.id
                    });
                });
            }
        });
        
        return links;
    }
    
    getNodeColor(d) {
        // Color based on sub-problem type
        const typeColors = {
            'research': '#4361ee',
            'analysis': '#3a0ca3',
            'implementation': '#4cc9f0',
            'validation': '#7209b7',
            'integration': '#f72585'
        };
        
        return typeColors[d.type] || '#6c757d';
    }
    
    onNodeClick(nodeData) {
        console.log('Node clicked:', nodeData);
        // In a real implementation, you might open a details panel
        // or highlight related elements
    }
    
    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Method to update visualization with new data
    updateWithPlan(plan) {
        this.update(plan);
    }
    
    // Method to resize the visualization
    resize(width, height) {
        this.width = width;
        this.height = height;
        
        this.svg
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height]);
            
        if (this.simulation) {
            this.simulation.force('center', d3.forceCenter(width / 2, height / 2));
        }
    }
}

/**
 * Performance Chart for displaying metrics over time
 */
class PerformanceChart {
    constructor(containerId) {
        this.containerId = containerId;
        this.chart = null;
    }
    
    init() {
        const ctx = document.getElementById(this.containerId).getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Quality Score',
                        data: [],
                        borderColor: '#4361ee',
                        backgroundColor: 'rgba(67, 97, 238, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Complexity',
                        data: [],
                        borderColor: '#f72585',
                        backgroundColor: 'rgba(247, 37, 133, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Decomposition Performance Metrics'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10
                    },
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                }
            }
        });
    }
    
    update(data) {
        if (!this.chart) {
            this.init();
        }
        
        // In a real implementation, this would update with actual metric data
        // For now, we'll just show placeholder data
        this.chart.data.labels = data.labels || ['Start', 'Iteration 1', 'Iteration 2', 'Iteration 3', 'Final'];
        this.chart.data.datasets[0].data = data.qualityScores || [6, 7, 8, 8.5, 9];
        this.chart.data.datasets[1].data = data.complexityScores || [7, 6.5, 6, 5.5, 5];
        
        this.chart.update();
    }
}

// Initialize visualizations when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the main visualization
    const viz = new SovereignVisualization('decompositionGraph');
    
    // Make it globally available for other scripts to update
    window.sovereignViz = viz;
    
    // Initialize performance chart if the element exists
    if (document.getElementById('performanceChart')) {
        window.performanceChart = new PerformanceChart('performanceChart');
    }
});