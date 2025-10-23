"""
Health Check Endpoint for Sovereign System
Provides HTTP endpoint for health monitoring
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import logging
from sovereign_reliability import get_health_monitor, get_error_handler
from sovereign_performance_optimization import get_performance_stats

logger = logging.getLogger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/health/detailed':
            self.handle_detailed_health()
        elif self.path == '/metrics':
            self.handle_metrics()
        elif self.path == '/errors':
            self.handle_errors()
        else:
            self.send_error(404, "Not Found")
    
    def handle_health(self):
        """Basic health check."""
        try:
            monitor = get_health_monitor()
            results = monitor.run_health_checks()
            
            if results['overall_healthy']:
                self.send_json_response(200, {
                    'status': 'healthy',
                    'timestamp': results['timestamp']
                })
            else:
                self.send_json_response(503, {
                    'status': 'unhealthy',
                    'failed_checks': [
                        name for name, result in results['checks'].items()
                        if not result.get('healthy', False)
                    ],
                    'timestamp': results['timestamp']
                })
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.send_json_response(500, {
                'status': 'error',
                'message': str(e)
            })
    
    def handle_detailed_health(self):
        """Detailed health check with all checks."""
        try:
            monitor = get_health_monitor()
            results = monitor.run_health_checks()
            self.send_json_response(200, results)
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            self.send_json_response(500, {
                'status': 'error',
                'message': str(e)
            })
    
    def handle_metrics(self):
        """Performance metrics endpoint."""
        try:
            stats = get_performance_stats()
            self.send_json_response(200, {
                'metrics': stats,
                'status': 'ok'
            })
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            self.send_json_response(500, {
                'status': 'error',
                'message': str(e)
            })
    
    def handle_errors(self):
        """Error statistics endpoint."""
        try:
            handler = get_error_handler()
            stats = handler.get_error_stats()
            self.send_json_response(200, stats)
        except Exception as e:
            logger.error(f"Error stats retrieval failed: {e}")
            self.send_json_response(500, {
                'status': 'error',
                'message': str(e)
            })
    
    def send_json_response(self, status_code, data):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")


def start_health_server(port=8000):
    """
    Start health check HTTP server.
    
    Args:
        port: Port to listen on
    """
    server_address = ('', port)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    
    logger.info(f"Health check server starting on port {port}")
    logger.info(f"Endpoints:")
    logger.info(f"  GET /health - Basic health check")
    logger.info(f"  GET /health/detailed - Detailed health check")
    logger.info(f"  GET /metrics - Performance metrics")
    logger.info(f"  GET /errors - Error statistics")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Health check server stopping")
        httpd.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    start_health_server()
