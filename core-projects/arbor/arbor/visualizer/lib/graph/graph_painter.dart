import 'dart:math';
import 'package:flutter/material.dart';
import '../core/providers.dart';
import '../core/theme.dart';

/// Custom painter for rendering the code graph.
///
/// This draws edges as bezier curves and nodes as glowing circles.
/// The visual style is cinematic with bloom effects and depth cues.
class GraphPainter extends CustomPainter {
  final List<GraphNode> nodes;
  final List<GraphEdge> edges;
  final String? selectedNodeId;
  final String? hoveredNodeId;
  final String? failureNodeId;
  final Set<String> entangledNodeIds;
  final Offset offset;
  final double scale;
  final bool isLowGpuMode;
  final double interactionSpeed;
  final double pulse;

  GraphPainter({
    required this.nodes,
    required this.edges,
    this.selectedNodeId,
    this.hoveredNodeId,
    this.failureNodeId,
    this.entangledNodeIds = const {},
    this.offset = Offset.zero,
    this.scale = 1.0,
    this.isLowGpuMode = false,
    this.interactionSpeed = 0.0,
    this.pulse = 0.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Build node map for edge drawing
    final nodeMap = {for (var n in nodes) n.id: n};
    
    // Viewport culling bounds (with margin for nodes near edges)
    const cullMargin = 100.0;
    final cullRect = Rect.fromLTWH(
      -cullMargin, 
      -cullMargin, 
      size.width + cullMargin * 2, 
      size.height + cullMargin * 2,
    );
    
    // Filter visible nodes for LOD/culling
    final visibleNodes = nodes.where((node) {
      final pos = _transformPoint(node.x, node.y);
      return cullRect.contains(pos);
    }).toList();
    
    // Create set of visible node IDs for edge culling
    final visibleNodeIds = visibleNodes.map((n) => n.id).toSet();

    // Draw edges first (below nodes) - only for visible nodes
    _drawEdges(canvas, nodeMap, size, visibleNodeIds);

    // Draw nodes with LOD based on scale
    for (final node in visibleNodes) {
      _drawNode(canvas, node, size);
    }
  }

  void _drawEdges(Canvas canvas, Map<String, GraphNode> nodeMap, Size size, Set<String> visibleNodeIds) {
    final screenCenter = size.center(Offset.zero);
    final maxDist = size.shortestSide * 0.8;

    for (final edge in edges) {
      // Edge culling: skip if neither endpoint is visible
      if (!visibleNodeIds.contains(edge.source) && !visibleNodeIds.contains(edge.target)) {
        continue;
      }
      
      final from = nodeMap[edge.source];
      final to = nodeMap[edge.target];
      if (from == null || to == null) continue;

      final start = _transformPoint(from.x, from.y);
      final end = _transformPoint(to.x, to.y);

      // Edge Fading: Depth of field effect
      final edgeCenter = Offset((start.dx + end.dx) / 2, (start.dy + end.dy) / 2);
      final dist = (edgeCenter - screenCenter).distance;
      // Fade out as it gets further from center
      final depthOpacity = (1.0 - (dist / maxDist)).clamp(0.1, 1.0);

      final isHighlighted = edge.source == selectedNodeId || 
                           edge.target == selectedNodeId;
      final isEntangledEdge = entangledNodeIds.contains(edge.source) && entangledNodeIds.contains(edge.target);

      final paint = Paint()
        ..color = isHighlighted 
            ? ArborTheme.function.withOpacity(0.8 * depthOpacity)
            : isEntangledEdge
                ? ArborTheme.variable.withOpacity(0.6 * depthOpacity)
                : ArborTheme.border.withOpacity(0.4 * depthOpacity)
        ..strokeWidth = isHighlighted ? 2.0 : (isEntangledEdge ? 1.6 : 1.0)
        ..style = PaintingStyle.stroke;

      // Draw a curved line for visual interest
      final controlX = (start.dx + end.dx) / 2;
      final controlY = (start.dy + end.dy) / 2 - 20;

      final path = Path()
        ..moveTo(start.dx, start.dy)
        ..quadraticBezierTo(controlX, controlY, end.dx, end.dy);

      canvas.drawPath(path, paint);

      // Draw arrow head
      if (isHighlighted) {
        _drawArrow(canvas, Offset(controlX, controlY), end, paint);
      }
    }
  }

  void _drawArrow(Canvas canvas, Offset from, Offset to, Paint paint) {
    final angle = atan2(to.dy - from.dy, to.dx - from.dx);
    const arrowSize = 10.0;

    final path = Path()
      ..moveTo(to.dx, to.dy)
      ..lineTo(
        to.dx - arrowSize * cos(angle - 0.4),
        to.dy - arrowSize * sin(angle - 0.4),
      )
      ..lineTo(
        to.dx - arrowSize * cos(angle + 0.4),
        to.dy - arrowSize * sin(angle + 0.4),
      )
      ..close();

    canvas.drawPath(path, paint..style = PaintingStyle.fill);
  }

  void _drawNode(Canvas canvas, GraphNode node, Size size) {
    final isSelected = node.id == selectedNodeId;
    final isHovered = node.id == hoveredNodeId;
    final isFailure = node.id == failureNodeId;
    final isEntangled = entangledNodeIds.contains(node.id);

    final jitterPhase = (node.id.hashCode % 360) / 360.0;
    final jitterAmplitude = isEntangled ? 1.5 : 0.0;
    final jitterX = sin((pulse + jitterPhase) * 2 * pi) * jitterAmplitude;
    final jitterY = cos((pulse + jitterPhase) * 2 * pi) * jitterAmplitude;
    final center = _transformPoint(node.x + jitterX, node.y + jitterY);
    
    // Size based on centrality (more important = bigger)
    final baseRadius = 8 + node.centrality * 16;
    final radius = isHovered ? baseRadius * 1.2 : baseRadius;
    
    // LOD: Skip fancy effects for very small nodes at low zoom OR high speed
    // If interactionSpeed > 100, we consider it "rapid movement"
    final isLOD = (scale < 0.3 && radius * scale < 4) || (interactionSpeed > 30.0);

    final color = isFailure ? ArborTheme.importType : ArborTheme.colorForKind(node.kind);
    
    // Ultra-simplified rendering for LOD mode
    if (isLOD && !isSelected && !isHovered) {
      final simplePaint = Paint()..color = color;
      canvas.drawCircle(center, max(2, radius * 0.5), simplePaint);
      return;
    }

    // Cinematic Bloom (Persistent for important nodes) - Disable in Low GPU Mode
    if (!isLowGpuMode && node.centrality > 0.3) {
       final bloomPaint = Paint()
        ..color = color.withOpacity(0.4)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 15);
       canvas.drawCircle(center, radius * 1.5, bloomPaint);
    }

    // Draw glow effect (Interactive)
    if (isSelected || isHovered || isFailure) {
      // Simple ring in Low GPU mode, Blur in High GPU
      if (isLowGpuMode) {
         final simpleGlowPaint = Paint()
          ..color = color.withOpacity(isFailure ? 0.6 : 0.3)
          ..style = PaintingStyle.stroke
          ..strokeWidth = isFailure ? 6 : 4;
         canvas.drawCircle(center, radius * (isFailure ? 1.8 : 1.5), simpleGlowPaint);
      } else {
         final glowPaint = Paint()
          ..color = color.withOpacity(isFailure ? 0.6 : 0.3)
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 20);
         canvas.drawCircle(center, radius * (isFailure ? 2.4 : 2), glowPaint);
      }
    }

    // Draw outer ring
    final ringPaint = Paint()
      ..color = color.withOpacity(isEntangled ? 0.7 : 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = isEntangled ? 2.6 : 2;
    canvas.drawCircle(center, radius, ringPaint);

    if (isEntangled) {
      final pulseRadius = radius * (1.2 + 0.15 * sin(pulse * 2 * pi));
      final entangledPaint = Paint()
        ..color = ArborTheme.variable.withOpacity(0.25)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
      canvas.drawCircle(center, pulseRadius, entangledPaint);
    }

    // Draw filled center
    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, radius * 0.6, fillPaint);

    // Draw label if selected or hovered
    if (isSelected || isHovered) {
      _drawLabel(canvas, center, node.name, color);
    }
  }

  void _drawLabel(Canvas canvas, Offset center, String text, Color color) {
    final textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(
              color: ArborTheme.background,
              blurRadius: 4,
            ),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    textPainter.paint(
      canvas,
      Offset(
        center.dx - textPainter.width / 2,
        center.dy + 20,
      ),
    );
  }

  Offset _transformPoint(double x, double y) {
    return Offset(
      x * scale + offset.dx,
      y * scale + offset.dy,
    );
  }

  @override
  bool shouldRepaint(covariant GraphPainter oldDelegate) {
    return nodes != oldDelegate.nodes ||
        edges != oldDelegate.edges ||
        selectedNodeId != oldDelegate.selectedNodeId ||
        hoveredNodeId != oldDelegate.hoveredNodeId ||
        failureNodeId != oldDelegate.failureNodeId ||
        entangledNodeIds != oldDelegate.entangledNodeIds ||
        offset != oldDelegate.offset ||
        scale != oldDelegate.scale ||
        isLowGpuMode != oldDelegate.isLowGpuMode ||
        interactionSpeed != oldDelegate.interactionSpeed ||
        pulse != oldDelegate.pulse;
  }
}
