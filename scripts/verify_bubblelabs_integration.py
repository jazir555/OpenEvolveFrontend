"""
OpenEvolve-BubbleLabs Integration Verification

This script verifies that the BubbleLabs integration properly connects with OpenEvolve.
"""

def verify_integration():
    print("Verifying OpenEvolve-BubbleLabs Integration...")
    print("=" * 50)
    
    # Test 1: Check that required imports work
    try:
        from workflow_structures import WorkflowState
        print("[OK] WorkflowState import successful")
    except ImportError as e:
        print(f"[ERROR] WorkflowState import failed: {e}")
        return False
    
    try:
        from workflow_engine import run_sovereign_workflow
        print("[OK] run_sovereign_workflow import successful")
    except ImportError as e:
        print(f"[ERROR] run_sovereign_workflow import failed: {e}")
        return False
    
    try:
        from team_manager import TeamManager
        from gauntlet_manager import GauntletManager
        print("[OK] Team and Gauntlet managers import successful")
    except ImportError as e:
        print(f"[ERROR] Team/Gauntlet managers import failed: {e}")
        return False
    
    # Test 2: Check BubbleLabs UI component
    try:
        from bubblelabs_ui_component import BubbleLabsWorkflowUI
        ui = BubbleLabsWorkflowUI()
        print("[OK] BubbleLabsWorkflowUI instantiation successful")
    except ImportError as e:
        print(f"[ERROR] BubbleLabsWorkflowUI import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] BubbleLabsWorkflowUI instantiation failed: {e}")
        return False
    
    # Test 3: Check that all required files exist and import
    required_files = [
        'bubblelabs_integration.py',
        'bubblelabs_ui_component.py',
        'start_bubblelabs_integration.py'
    ]
    
    import os
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} exists")
        else:
            print(f"[ERROR] {file} does not exist")
            return False
    
    # Test 4: Check that main.py has proper integration
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'BubbleLabs Workflows' in content and 'render_bubblelabs_workflow_ui' in content:
                print("[OK] Main UI integration verified")
            else:
                print("[ERROR] Main UI integration not found")
                return False
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open('main.py', 'r', encoding='latin-1') as f:
                content = f.read()
                if 'BubbleLabs Workflows' in content and 'render_bubblelabs_workflow_ui' in content:
                    print("[OK] Main UI integration verified")
                else:
                    print("[ERROR] Main UI integration not found")
                    return False
        except Exception as e:
            print(f"[ERROR] Error checking main.py integration: {e}")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking main.py integration: {e}")
        return False
    
    print("=" * 50)
    print("[OK] All integration checks passed!")
    print("The BubbleLabs integration is properly connected to OpenEvolve.")
    return True

if __name__ == "__main__":
    verify_integration()