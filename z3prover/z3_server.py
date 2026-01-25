"""
Z3 Prover HTTP Server
Simple HTTP server that wraps Z3 Python API for BubbleLab integration
"""

import http.server
import json
import socket
import subprocess
import sys
import os
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

# Z3 imports
try:
    from z3 import *
except ImportError:
    print("ERROR: z3-solver not installed. Run: pip install z3-solver", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

Z3_PORT = int(os.environ.get("Z3_PORT", 7655))
Z3_HOST = os.environ.get("Z3_HOST", "0.0.0.0")

# ============================================================================
# Z3 SERVICE
# ============================================================================

class Z3ServiceError(Exception):
    """Custom Z3 service error"""
    pass

class Z3Service:
    """Z3 Solver service wrapper"""

    def __init__(self):
        pass

    def solve_smt(self, smtlib2: str, logic: Optional[str] = None, timeout: int = 30000) -> Dict[str, Any]:
        """Solve SMT problem expressed in SMTLIB2 format"""
        try:
            s = Solver()

            # Set logic if specified
            if logic:
                s.set(logic=logic)

            # Parse and add assertions
            s.from_string(smtlib2)

            # Check satisfiability
            result = s.check()

            response = {
                'result': 'sat' if result == sat else 'unsat' if result == unsat else 'unknown',
            }

            # Extract model if SAT
            if result == sat:
                model = s.model()
                model_dict = {}
                for decl in model:
                    try:
                        val = model[decl]
                        model_dict[str(decl)] = str(val) if val else None
                    except:
                        pass
                response['model'] = model_dict

            # Get statistics
            response['statistics'] = str(s.statistics())

            return response

        except Exception as e:
            raise Z3ServiceError(f"Z3 solving failed: {str(e)}")

    def optimize(self, objectives: List[Dict[str, str]],
                 constraints: Optional[List[str]] = None,
                 timeout: int = 30000) -> Dict[str, Any]:
        """Solve optimization problem"""
        try:
            opt = Optimize()

            # Add constraints if provided
            if constraints:
                for constraint in constraints:
                    # Parse constraint in a safe context
                    try:
                        opt.add(eval(constraint, {
                            'Int': Int,
                            'Real': Real,
                            'Bool': Bool,
                            'BitVec': BitVec,
                            'Array': Array,
                        }, {}))
                    except Exception as e:
                        raise Z3ServiceError(f"Failed to parse constraint '{constraint}': {str(e)}")

            # Add objectives
            handles = {}
            for obj in objectives:
                try:
                    expr = eval(obj['expression'], {
                        'Int': Int,
                        'Real': Real,
                        'Bool': Bool,
                        'BitVec': BitVec,
                        'Array': Array,
                    }, {})
                    if obj['type'] == 'maximize':
                        handles[obj['expression']] = opt.maximize(expr)
                    else:
                        handles[obj['expression']] = opt.minimize(expr)
                except Exception as e:
                    raise Z3ServiceError(f"Failed to parse objective '{obj}': {str(e)}")

            # Optimize
            result = opt.check()

            response = {
                'status': 'optimal' if result == optimal else 'unsat' if result == unsat else 'unknown',
            }

            # Extract model and objective values
            if result == optimal:
                model = opt.model()
                model_dict = {}
                for decl in model:
                    try:
                        val = model[decl]
                        model_dict[str(decl)] = str(val) if val else None
                    except:
                        pass
                response['model'] = model_dict

                # Get objective values
                obj_values = {}
                for expr, handle in handles.items():
                    obj_values[expr] = opt.value(handle)
                response['objective_values'] = obj_values

            return response

        except Exception as e:
            raise Z3ServiceError(f"Z3 optimization failed: {str(e)}")

    def simplify(self, expression: str,
                 assumptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Simplify expression"""
        try:
            # Parse assumptions
            ctx = None
            assm_list = []
            if assumptions:
                for a in assumptions:
                    try:
                        assm_list.append(eval(a, {
                            'Int': Int,
                            'Real': Real,
                            'Bool': Bool,
                            'BitVec': BitVec,
                        }, {}))
                    except Exception as e:
                        raise Z3ServiceError(f"Failed to parse assumption '{a}': {str(e)}")

            # Simplify
            expr = eval(expression, {
                'Int': Int,
                'Real': Real,
                'Bool': Bool,
                'BitVec': BitVec,
            }, {})

            simplified = simplify(expr, *assm_list)

            return {
                'result': str(simplified)
            }

        except Exception as e:
            raise Z3ServiceError(f"Z3 simplification failed: {str(e)}")

    def apply_tactic(self, goal: str, tactic: str,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply tactic to goal"""
        try:
            g = Goal()
            g.add(eval(goal, {
                'Int': Int,
                'Real': Real,
                'Bool': Bool,
                'BitVec': BitVec,
                'Array': Array,
            }, {}))

            # Create tactic with optional parameters
            if params:
                t = Tactic(tactic, **params)
            else:
                t = Tactic(tactic)

            result = t(g)

            response = {
                'status': str(result),
            }

            # Extract subgoals
            goals = []
            for subgoal in result:
                goals.append(str(subgoal.as_expr()))
            response['goals'] = goals

            return response

        except Exception as e:
            raise Z3ServiceError(f"Z3 tactic failed: {str(e)}")

    def get_tactics(self) -> List[Dict[str, str]]:
        """Get list of available tactics"""
        tactics = [
            {'name': 'simplify', 'description': 'Simplify expression'},
            {'name': 'sat', 'description': 'SAT solver'},
            {'name': 'sat-preprocess', 'description': 'SAT preprocessing'},
            {'name': 'solve-eqs', 'description': 'Solve equations'},
            {'name': 'bit-blast', 'description': 'Bitvector blasting'},
            {'name': 'pb-lemma', 'description': 'PB lemma'},
            {'name': 'nlsat', 'description': 'Non-linear arithmetic'},
            {'name': 'qff', 'description': 'Quantifier-free floating point'},
            {'name': 'snf', 'description': 'Skolem normal form'},
            {'name': 'tseitin-cnf', 'description': 'Tseitin CNF conversion'},
            {'name': 'der', 'description': 'Destructive equality resolution'},
            {'name': 'factor-sleep', 'description': 'Factorization'},
            {'name': 'fm', 'description': 'Fourier-Motzkin'},
            {'name': 'lift-ite', 'description': 'Lift if-then-else'},
            {'name': 'max-bv-sharding', 'description': 'Maximize bitvector sharding'},
            {'name': 'pb-rewrite', 'description': 'Pseudo-boolean rewriting'},
            {'name': 'propagate-values', 'description': 'Propagate values'},
            {'name': 'recover-01', 'description': 'Recover 01 values'},
            {'name': 'smt', 'description': 'SMT solver'},
            {'name': 'subst-cov', 'description': 'Substitution coverage'},
            {'name': 'ujf', 'description': 'Justification filter'},
            {'name': 'ctx-solver-simplify', 'description': 'Context solver simplification'},
            {'name': 'bv1-bv', 'description': 'Bitvector solver'},
            {'name': 'aig', 'description': 'And-inverter graph'},
            {'name': 'qfnia', 'description': 'Quantifier-free nonlinear integer arithmetic'},
            {'name': 'qfaufbv', 'description': 'Quantifier-free arrays, bitvectors, uninterpreted functions'},
            {'name': 'uf2bv', 'description': 'Uninterpreted functions to bitvectors'},
            {'name': 'bfm', 'description': 'Brute force bitvector model finding'},
        ]
        return tactics

    def get_logics(self) -> List[str]:
        """Get list of supported logics"""
        return [
            'AUFLIRA', 'AUFLIRF', 'AUFNIRA', 'BV', 'BVREF',
            'HORN', 'LIA', 'LRA', 'NIA', 'NRA', 'QF_ABV',
            'QF_AUFBV', 'QF_AUFLIA', 'QF_BV', 'QF_IDL',
            'QF_LIA', 'QF_LRA', 'QF_NIA', 'QF_NRA',
            'QF_UF', 'QF_UFBV', 'UFLRA', 'UF', 'UFBV',
            'QF_AX', 'QF_S', 'SMT', 'ALL',
        ]

    def get_version(self) -> Dict[str, str]:
        """Get Z3 version information"""
        try:
            return {
                'version': get_version_string(),
                'full_version': get_full_version(),
            }
        except Exception as e:
            raise Z3ServiceError(f"Failed to get version: {str(e)}")

# ============================================================================
# HTTP SERVER
# ============================================================================

class Z3RequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Z3 server"""

    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, message: str, status: int = 500):
        """Send error response"""
        self._send_json_response({'error': message}, status)

    def _read_json_body(self) -> Dict[str, Any]:
        """Read JSON request body"""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def do_POST(self):
        """Handle POST requests"""
        try:
            # Get request path
            if self.path == '/solve':
                self._handle_solve()
            elif self.path == '/optimize':
                self._handle_optimize()
            elif self.path == '/simplify':
                self._handle_simplify()
            elif self.path == '/tactic':
                self._handle_tactic()
            elif self.path == '/fixedpoint':
                self._handle_fixedpoint()
            else:
                self._send_error(f'Unknown endpoint: {self.path}', 404)

        except Z3ServiceError as e:
            self._send_error(str(e), 400)
        except Exception as e:
            self._send_error(f"Internal error: {str(e)}", 500)

    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/tactics':
                self._handle_get_tactics()
            elif self.path == '/logics':
                self._handle_get_logics()
            elif self.path == '/version':
                self._handle_get_version()
            elif self.path == '/health':
                self._handle_health()
            else:
                self._send_error(f'Unknown endpoint: {self.path}', 404)

        except Exception as e:
            self._send_error(f"Internal error: {str(e)}", 500)

    def _handle_solve(self):
        """Handle SMT solving"""
        data = self._read_json_body()
        service = Z3Service()

        smtlib2 = data.get('smtlib2')
        timeout = data.get('timeout', 30000)
        logic = data.get('logic')

        if not smtlib2:
            raise Z3ServiceError('Missing required field: smtlib2')

        result = service.solve_smt(smtlib2, logic, timeout)
        self._send_json_response(result)

    def _handle_optimize(self):
        """Handle optimization"""
        data = self._read_json_body()
        service = Z3Service()

        objectives = data.get('objectives', [])
        constraints = data.get('constraints')
        timeout = data.get('timeout', 30000)

        if not objectives:
            raise Z3ServiceError('Missing required field: objectives')

        result = service.optimize(objectives, constraints, timeout)
        self._send_json_response(result)

    def _handle_simplify(self):
        """Handle simplification"""
        data = self._read_json_body()
        service = Z3Service()

        expression = data.get('expression')
        assumptions = data.get('assumptions')
        timeout = data.get('timeout', 10000)

        if not expression:
            raise Z3ServiceError('Missing required field: expression')

        result = service.simplify(expression, assumptions)
        self._send_json_response(result)

    def _handle_tactic(self):
        """Handle tactic application"""
        data = self._read_json_body()
        service = Z3Service()

        goal = data.get('goal')
        tactic = data.get('tactic')
        params = data.get('params')
        timeout = data.get('timeout', 30000)

        if not goal:
            raise Z3ServiceError('Missing required field: goal')
        if not tactic:
            raise Z3ServiceError('Missing required field: tactic')

        result = service.apply_tactic(goal, tactic, params)
        self._send_json_response(result)

    def _handle_fixedpoint(self):
        """Handle fixedpoint query"""
        data = self._read_json_body()
        service = Z3Service()

        # Fixedpoint is complex, for now return a simplified response
        rules = data.get('rules', [])
        query = data.get('query', '')

        try:
            fp = Fixedpoint()
            for rule in rules:
                fp.add_rule(eval(rule, {
                    'Int': Int,
                    'Real': Real,
                    'Bool': Bool,
                }, {}))

            result = fp.query(eval(query, {
                'Int': Int,
                'Real': Real,
                'Bool': Bool,
            }, {}))

            self._send_json_response({
                'result': str(result),
                'answer': str(result) if result else None
            })
        except Exception as e:
            self._send_json_response({
                'result': 'error',
                'error': str(e)
            })

    def _handle_get_tactics(self):
        """Handle get tactics"""
        service = Z3Service()
        tactics = service.get_tactics()
        self._send_json_response({'tactics': tactics})

    def _handle_get_logics(self):
        """Handle get logics"""
        service = Z3Service()
        logics = service.get_logics()
        self._send_json_response({'logics': logics})

    def _handle_get_version(self):
        """Handle get version"""
        service = Z3Service()
        version = service.get_version()
        self._send_json_response(version)

    def _handle_health(self):
        """Handle health check"""
        try:
            service = Z3Service()
            version = service.get_version()
            self._send_json_response({
                'status': 'healthy',
                'z3_available': True,
                'version': version['version']
            })
        except Exception as e:
            self._send_json_response({
                'status': 'unhealthy',
                'z3_available': False,
                'error': str(e)
            })

    def log_message(self, format: str, *args):
        """Log message to console"""
        print(f"[Z3 Server] {format % args}")

# ============================================================================
# SERVER STARTUP
# ============================================================================

def run_server():
    """Run the Z3 HTTP server"""
    server_address = (Z3_HOST, Z3_PORT)
    httpd = HTTPServer(server_address, Z3RequestHandler)
    httpd.serve_forever()

if __name__ == '__main__':
    print(f"Starting Z3 Server on {Z3_HOST}:{Z3_PORT}")
    print(f"Z3 version: {get_version_string()}")
    print("Press Ctrl+C to stop the server")
    run_server()
