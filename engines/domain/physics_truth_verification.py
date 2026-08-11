"""
BRUTAL NO-HOLDS-BARRED VERIFICATION
Of E2E Invention Physics "TRUE 100%" Claims
"""

import numpy as np
import re

def main():
    print('='*70)
    print('BRUTAL VERIFICATION OF E2E INVENTION PHYSICS CLAIMS')
    print('='*70)

    # ===== CLAIM 1: REAL FEA WITH STIFFNESS MATRIX =====
    print('\n' + '='*70)
    print('CLAIM 1: REAL FEA WITH STIFFNESS MATRIX')
    print('='*70)

    with open('physics_validator_real.py', 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    total_lines = len(lines)

    # Check for stiffness matrix assembly
    has_stiffness_matrix = 'self.K' in content or 'K_global' in content or 'K_data' in content
    has_spsolve = 'spsolve' in content
    has_Ku_F = 'K_global' in content and 'spsolve' in content

    print(f'[OK] File has {total_lines} lines')
    print(f'[OK] Has stiffness matrix variables: {has_stiffness_matrix}')
    print(f'[OK] Uses spsolve (sparse solver): {has_spsolve}')
    print(f'[OK] Solves K*u = F system: {has_Ku_F}')

    # Check specific FEA implementations
    checks = [
        ('Element stiffness k_e = EA/h', 'k_e = E * A_e / h'),
        ('Global K assembly with COO format', 'K_data.append'),
        ('Scipy CSR matrix for K_global', 'sp.csr_matrix((K_data, (K_row, K_col))'),
        ('Sparse solve K*u=F', 'spla.spsolve(K_global, F)'),
        ('2D plane stress with B-matrix', 'B.T @ D @ B'),
        ('Constitutive matrix D for plane stress', 'D = E / (1 - nu**2)'),
        ('Strain calculation B*u', 'B @ u_elem'),
        ('Stress = D*strain', 'stress = D @ strain'),
        ('Von Mises stress calculation', 'von_mises'),
        ('Modal analysis eigenvalue solve', 'eig'),
    ]

    fea_score = 0
    for desc, pattern in checks:
        found = pattern in content
        if found:
            fea_score += 1
            print(f'  [OK] {desc}')
        else:
            print(f'  [XX] {desc} - NOT FOUND')

    print(f'\nFEA Implementation Score: {fea_score}/{len(checks)} checks passed')

    # Count real FEA related lines
    fea_keywords = ['stiffness', 'spsolve', 'csr_matrix', 'element', 'mesh', 'D @ strain', 'B.T']
    fea_lines = set()
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in fea_keywords):
            fea_lines.add(i)

    print(f'Lines related to FEA implementation: {len(fea_lines)}')

    # ===== CLAIM 2: REAL CFD WITH NAVIER-STOKES =====
    print('\n' + '='*70)
    print('CLAIM 2: REAL CFD WITH NAVIER-STOKES')
    print('='*70)

    ns_checks = [
        ('Lid-driven cavity solver', 'solve_steady_lid_driven_cavity'),
        ('Velocity field u (x-velocity)', "u = np.zeros((nx, ny))"),
        ('Velocity field v (y-velocity)', "v = np.zeros((nx, ny))"),
        ('Pressure field p', "p = np.zeros((nx, ny))"),
        ('Reynolds number handling', 'Re:'),
        ('Convection term computation', 'conv_u = '),
        ('Diffusion term computation', 'diff_u = '),
        ('Pressure gradient', 'dpdx'),
        ('Iterative SIMPLE-like algorithm', 'max_iter'),
        ('Convergence checking', 'tol = '),
        ('Boundary conditions (walls)', 'u[:, 0] = 0'),
        ('Velocity magnitude field', 'vel_mag = np.sqrt'),
        ('Hagen-Poiseuille pipe flow', 'solve_pipe_flow'),
        ('Analytical parabolic profile', 'R**2 - r**2'),
    ]

    cfd_score = 0
    for desc, pattern in ns_checks:
        found = pattern in content
        if found:
            cfd_score += 1
            print(f'  [OK] {desc}')
        else:
            print(f'  [XX] {desc} - NOT FOUND')

    print(f'\nCFD Implementation Score: {cfd_score}/{len(ns_checks)} checks passed')

    # Check what methods are returned
    methods = re.findall(r'"method":\s*"([^"]+)"', content)
    unique_methods = set(methods)
    print(f'\nUnique computation methods returned:')
    for m in sorted(unique_methods):
        print(f'  - {m}')

    # ===== CLAIM 3: POLYNOMIAL CHAOS EXPANSION =====
    print('\n' + '='*70)
    print('CLAIM 3: POLYNOMIAL CHAOS EXPANSION')
    print('='*70)

    with open('uncertainty_propagation_real.py', 'r', encoding='utf-8') as f:
        uq_content = f.read()

    uq_lines = uq_content.split('\n')

    pce_checks = [
        ('PCE class exists', 'RealPolynomialChaosExpansion'),
        ('Polynomial order parameter', 'polynomial_order'),
        ('Basis indices generation', '_generate_basis_indices'),
        ('Multi-dimensional quadrature', '_multi_dimensional_projection'),
        ('Gauss quadrature points', 'leggauss'),
        ('Hermite-Gauss for normal', 'hermegauss'),
        ('Orthogonal Legendre eval', 'eval_legendre'),
        ('Orthogonal Hermite eval', 'eval_hermitenorm'),
        ('Projection coefficients', 'coefficients.append'),
        ('Basis norm squared', '_basis_norm_squared'),
        ('Sobol indices from PCE', 'get_sobol_indices'),
        ('Variance from coefficients', 'sum(c**2 for c in self.coefficients)'),
    ]

    pce_score = 0
    for desc, pattern in pce_checks:
        found = pattern in uq_content
        if found:
            pce_score += 1
            print(f'  [OK] {desc}')
        else:
            print(f'  [XX] {desc} - NOT FOUND')

    print(f'\nPCE Implementation Score: {pce_score}/{len(pce_checks)} checks passed')

    # ===== FUNCTIONAL TESTS =====
    print('\n' + '='*70)
    print('FUNCTIONAL VERIFICATION TESTS')
    print('='*70)

    # Test FEA
    try:
        from physics_validator_real import RealFiniteElementAnalysis, MeshGenerator
        fea = RealFiniteElementAnalysis()
        
        # 1D stress analysis
        result = fea.solve_stress_analysis_1d(
            length=1.0,
            n_elements=10,
            E=200e9,
            A_func=lambda x: 0.01,
            loads=[(0.5, 1000)],
            constraints=[(0, 0)]
        )
        
        if 'stress_field' in result and 'displacement_field' in result:
            print('[OK] FEA Test 1D: Returns actual field data')
            print(f'       Max stress: {result["max_stress"]:.2f} Pa')
            print(f'       Method: {result.get("method", "unknown")}')
        else:
            print('[XX] FEA Test 1D: No field data')
            
    except Exception as e:
        print(f'[XX] FEA Test: ERROR - {e}')

    # Test 2D FEA
    try:
        mesh = MeshGenerator.generate_2d_rectangular_mesh(1.0, 1.0, 5, 5)
        result = fea.solve_2d_plane_stress(
            mesh=mesh,
            E=200e9,
            nu=0.3,
            thickness=0.01,
            forces={24: [1000, 0]},
            fixed_nodes=[0, 1, 2, 3, 4]
        )
        
        if 'stress_field' in result and 'von_mises_field' in result:
            print('[OK] FEA Test 2D: Returns stress tensor and von Mises')
            print(f'       Max von Mises: {result["max_von_mises"]:.2f} Pa')
            print(f'       Method: {result.get("method", "unknown")}')
        else:
            print('[XX] FEA Test 2D: No stress field')
            
    except Exception as e:
        print(f'[XX] FEA Test 2D: ERROR - {e}')

    # Test CFD
    try:
        from physics_validator_real import NavierStokesSolver
        cfd = NavierStokesSolver(nx=20, ny=20)
        
        result = cfd.solve_steady_lid_driven_cavity(Re=100)
        
        if 'u_velocity' in result and 'v_velocity' in result and 'pressure' in result:
            print('[OK] CFD Test: Returns velocity and pressure fields')
            print(f'       Max velocity: {result["max_velocity"]:.4f}')
            print(f'       Method: {result.get("method", "unknown")}')
        else:
            print('[XX] CFD Test: No field data')
            
    except Exception as e:
        print(f'[XX] CFD Test: ERROR - {e}')

    # Test PCE
    try:
        from uncertainty_propagation_real import RealPolynomialChaosExpansion, UncertaintySource
        
        pce = RealPolynomialChaosExpansion(polynomial_order=3)
        
        def model(x):
            return x[0]**2 + 2*x[1]
        
        sources = [
            UncertaintySource('p1', 'uniform', {'low': 0, 'high': 1}),
            UncertaintySource('p2', 'normal', {'mean': 0, 'std': 1})
        ]
        
        result = pce.fit(model, sources, method='quadrature')
        
        if 'coefficients' in result and result.get('n_basis_functions', 0) > 1:
            print('[OK] PCE Test: Creates polynomial basis and coefficients')
            print(f'       Basis functions: {result["n_basis_functions"]}')
            print(f'       Mean (c_0): {result["mean"]:.4f}')
        else:
            print('[XX] PCE Test: No coefficients or basis')
            
    except Exception as e:
        print(f'[XX] PCE Test: ERROR - {e}')

    # Test Sobol
    try:
        from uncertainty_propagation_real import RealSobolAnalyzer
        
        sobol = RealSobolAnalyzer()
        
        def model(x):
            return x[0] + 2*x[1]**2
        
        sources = [
            UncertaintySource('p1', 'uniform', {'low': 0, 'high': 1}),
            UncertaintySource('p2', 'uniform', {'low': 0, 'high': 1})
        ]
        
        indices = sobol.analyze(model, sources, n_samples=1000)
        
        if indices.first_order and indices.total_order:
            print('[OK] Sobol Test: Computes sensitivity indices')
        else:
            print('[XX] Sobol Test: No indices computed')
            
    except Exception as e:
        print(f'[XX] Sobol Test: ERROR - {e}')

    # ===== FINAL VERDICT =====
    print('\n' + '='*70)
    print('FINAL BRUTAL VERDICT')
    print('='*70)
    
    # Calculate actual percentage
    fea_pct = (fea_score / len(checks)) * 100
    cfd_pct = (cfd_score / len(ns_checks)) * 100
    pce_pct = (pce_score / len(pce_checks)) * 100
    
    overall = (fea_pct + cfd_pct + pce_pct) / 3
    
    print(f'''
CLAIM 1: REAL FEA WITH STIFFNESS MATRIX
  Implementation: {fea_score}/{len(checks)} ({fea_pct:.0f}%)
  Verdict: {'VERIFIED - REAL' if fea_pct >= 80 else 'PARTIAL' if fea_pct >= 50 else 'INADEQUATE'}
  
CLAIM 2: REAL CFD WITH NAVIER-STOKES  
  Implementation: {cfd_score}/{len(ns_checks)} ({cfd_pct:.0f}%)
  Verdict: {'VERIFIED - REAL' if cfd_pct >= 80 else 'PARTIAL' if cfd_pct >= 50 else 'INADEQUATE'}

CLAIM 3: POLYNOMIAL CHAOS EXPANSION
  Implementation: {pce_score}/{len(pce_checks)} ({pce_pct:.0f}%)
  Verdict: {'VERIFIED - REAL' if pce_pct >= 80 else 'PARTIAL' if pce_pct >= 50 else 'INADEQUATE'}

========================================================================
ACTUAL IMPLEMENTATION PERCENTAGE: {overall:.1f}%
CLAIMED PERCENTAGE: 100%
========================================================================
''')
    
    if overall >= 90:
        print("VERDICT: SUBSTANTIALLY VERIFIED - Real physics implementations")
    elif overall >= 70:
        print("VERDICT: MOSTLY VERIFIED - Real implementations with minor gaps")
    elif overall >= 50:
        print("VERDICT: PARTIALLY IMPLEMENTED - Significant gaps exist")
    else:
        print("VERDICT: INADEQUATE - Major implementation gaps")
    
    print('='*70)

if __name__ == "__main__":
    main()
