/**
 * Knowledge Graph Visualization Library
 * Production-grade JavaScript following CLAUDE.md principles
 */

(function(global) {
    'use strict';

    class GraphVisualization {
        constructor(containerId, options = {}) {
            this.containerId = containerId;
            this.options = {
                width: options.width || 1200,
                height: options.height || 800,
                showLabels: options.showLabels !== false,
                enableZoom: options.enableZoom !== false,
                enablePhysics: options.enablePhysics !== false,
                enableSelection: options.enableSelection !== false,
                nodeSize: options.nodeSize || 'centrality',
                colorScheme: options.colorScheme || 'colorblind',
                ...options
            };

            this.data = null;
            this.simulation = null;
            this.svg = null;
            this.zoom = null;

            this.init();
        }

        init() {
            const container = document.getElementById(this.containerId);
            if (!container) {
                console.error(`Container ${this.containerId} not found`);
                return;
            }

            // Create SVG
            this.svg = d3.select(`#${this.containerId}`)
                .append('svg')
                .attr('width', this.options.width)
                .attr('height', this.options.height);

            // Create groups
            this.linksGroup = this.svg.append('g').attr('class', 'links');
            this.nodesGroup = this.svg.append('g').attr('class', 'nodes');
            this.labelsGroup = this.svg.append('g').attr('class', 'labels');

            // Setup zoom
            if (this.options.enableZoom) {
                this.setupZoom();
            }
        }

        setupZoom() {
            this.zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    this.linksGroup.attr('transform', event.transform);
                    this.nodesGroup.attr('transform', event.transform);
                    this.labelsGroup.attr('transform', event.transform);
                });

            this.svg.call(this.zoom);
        }

        loadData(graphData) {
            this.data = graphData;
            this.render();
        }

        render() {
            if (!this.data) {
                console.error('No data loaded');
                return;
            }

            // Clear existing
            this.linksGroup.selectAll('*').remove();
            this.nodesGroup.selectAll('*').remove();
            this.labelsGroup.selectAll('*').remove();

            // Create links
            const link = this.linksGroup.selectAll('line')
                .data(this.data.edges)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('stroke', d => d.type === 'dashed' ? '#999' : '#666')
                .attr('stroke-width', d => (d.confidence || 1) * 2)
                .attr('stroke-dasharray', d => d.type === 'dashed' ? '5,5' : 'none');

            // Create nodes
            const node = this.nodesGroup.selectAll('circle')
                .data(this.data.nodes)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('r', d => d.size || 10)
                .attr('fill', d => d.color || '#1f77b4')
                .call(d3.drag()
                    .on('start', this.dragstarted.bind(this))
                    .on('drag', this.dragged.bind(this))
                    .on('end', this.dragended.bind(this)));

            // Create labels
            if (this.options.showLabels) {
                const label = this.labelsGroup.selectAll('text')
                    .data(this.data.nodes)
                    .enter()
                    .append('text')
                    .attr('class', 'label')
                    .text(d => d.id)
                    .attr('font-size', '12px')
                    .attr('font-weight', '500')
                    .attr('fill', '#333')
                    .attr('text-anchor', 'middle')
                    .attr('dy', d => (d.size || 10) + 15)
                    .style('pointer-events', 'none');
            }

            // Setup physics simulation
            if (this.options.enablePhysics) {
                this.setupSimulation(node, link);
            } else {
                // Static positioning
                node.attr('cx', d => d.x || this.options.width / 2)
                    .attr('cy', d => d.y || this.options.height / 2);

                link.attr('x1', d => d.source.x || this.options.width / 2)
                    .attr('y1', d => d.source.y || this.options.height / 2)
                    .attr('x2', d => d.target.x || this.options.width / 2)
                    .attr('y2', d => d.target.y || this.options.height / 2);
            }

            // Setup interactions
            this.setupInteractions(node, link);
        }

        setupSimulation(node, link) {
            this.simulation = d3.forceSimulation(this.data.nodes)
                .force('link', d3.forceLink(this.data.edges)
                    .id(d => d.id)
                    .distance(100))
                .force('charge', d3.forceManyBody()
                    .strength(-300))
                .force('center', d3.forceCenter(
                    this.options.width / 2,
                    this.options.height / 2
                ))
                .force('collision', d3.forceCollide()
                    .radius(d => (d.size || 10) + 5))
                .on('tick', () => {
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);

                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);

                    if (this.options.showLabels) {
                        this.labelsGroup.selectAll('text')
                            .attr('x', d => d.x)
                            .attr('y', d => d.y);
                    }
                });
        }

        setupInteractions(node, link) {
            // Hover effects
            node.on('mouseover', (event, d) => {
                this.onNodeHover(event, d);
            }).on('mouseout', (event, d) => {
                this.onNodeLeave(event, d);
            }).on('click', (event, d) => {
                this.onNodeClick(event, d);
            });
        }

        onNodeHover(event, d) {
            // Highlight connected nodes and edges
            this.nodesGroup.selectAll('.node')
                .style('opacity', n => {
                    if (n.id === d.id) return 1;
                    const isConnected = this.data.edges.some(e =>
                        (e.source.id === d.id && e.target.id === n.id) ||
                        (e.target.id === d.id && e.source.id === n.id)
                    );
                    return isConnected ? 1 : 0.2;
                });

            this.linksGroup.selectAll('.link')
                .style('opacity', e => {
                    return (e.source.id === d.id || e.target.id === d.id) ? 1 : 0.1;
                });

            // Show tooltip
            this.showTooltip(event, d);
        }

        onNodeLeave(event, d) {
            // Reset opacity
            this.nodesGroup.selectAll('.node')
                .style('opacity', 1);

            this.linksGroup.selectAll('.link')
                .style('opacity', 0.6);

            // Hide tooltip
            this.hideTooltip();
        }

        onNodeClick(event, d) {
            if (this.options.enableSelection) {
                d3.select(event.currentTarget)
                    .classed('selected', !d3.select(event.currentTarget).classed('selected'));

                // Trigger custom event
                const customEvent = new CustomEvent('nodeSelected', {
                    detail: { node: d }
                });
                document.dispatchEvent(customEvent);
            }
        }

        showTooltip(event, d) {
            let tooltip = document.querySelector('.tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                document.body.appendChild(tooltip);
            }

            const html = `
                <strong>${d.id}</strong><br/>
                ${d.community !== undefined ? `Community: ${d.community}<br/>` : ''}
                ${d.centrality !== undefined ? `Centrality: ${d.centrality.toFixed(3)}<br/>` : ''}
                ${d.degree !== undefined ? `Degree: ${d.degree}` : ''}
            `;

            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
        }

        hideTooltip() {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) {
                tooltip.style.display = 'none';
            }
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

        zoomIn() {
            this.svg.transition().call(this.zoom.scaleBy, 1.3);
        }

        zoomOut() {
            this.svg.transition().call(this.zoom.scaleBy, 0.7);
        }

        resetZoom() {
            this.svg.transition().call(this.zoom.transform, d3.zoomIdentity);
        }

        filterNodes(predicate) {
            this.nodesGroup.selectAll('.node')
                .style('display', d => predicate(d) ? 'block' : 'none');

            this.linksGroup.selectAll('.link')
                .style('display', d => predicate(d.source) && predicate(d.target) ? 'block' : 'none');

            if (this.options.showLabels) {
                this.labelsGroup.selectAll('.label')
                    .style('display', d => predicate(d) ? 'block' : 'none');
            }
        }

        highlightNodes(nodeIds) {
            const nodeSet = new Set(nodeIds);

            this.nodesGroup.selectAll('.node')
                .classed('highlighted', d => nodeSet.has(d.id))
                .classed('dimmed', d => !nodeSet.has(d.id));
        }

        resetHighlight() {
            this.nodesGroup.selectAll('.node')
                .classed('highlighted', false)
                .classed('dimmed', false);
        }

        exportAsSVG() {
            const svgData = new XMLSerializer().serializeToString(this.svg.node());
            const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
            const url = URL.createObjectURL(blob);

            const link = document.createElement('a');
            link.href = url;
            link.download = 'graph-visualization.svg';
            link.click();

            URL.revokeObjectURL(url);
        }

        destroy() {
            if (this.simulation) {
                this.simulation.stop();
            }

            this.svg.remove();
        }
    }

    // Export to global
    global.GraphVisualization = GraphVisualization;

})(typeof window !== 'undefined' ? window : global);
