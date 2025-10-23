"""
Test the fixed problem decomposition system
"""

from problem_decomposition import (
    ProblemDecomposer, 
    DecompositionStrategy, 
    ComponentType
)


def test_basic_functionality():
    """Test basic decomposition functionality"""
    print("Testing basic functionality...")
    
    decomposer = ProblemDecomposer()
    
    test_content = """
# Introduction
This is the introduction section with some content.

# Main Content
This is the main content section with more detailed information.
It has multiple paragraphs and explains the core concepts.

# Conclusion
This is the conclusion section that wraps up the content.
"""
    
    result = decomposer.decompose_content(
        content=test_content,
        strategy=DecompositionStrategy.HIERARCHICAL,
        max_components=5,
        min_component_size=20
    )
    
    print(f"✅ Decomposition completed")
    print(f"   - Original content length: {len(result.original_content)}")
    print(f"   - Components created: {len(result.components)}")
    print(f"   - Quality score: {result.quality_score:.2f}")
    print(f"   - Strategy used: {result.decomposition_strategy.value}")
    
    for i, component in enumerate(result.components):
        print(f"   - Component {i+1}: {component.title} ({len(component.content)} chars)")
    
    return True


def test_different_strategies():
    """Test different decomposition strategies"""
    print("\nTesting different strategies...")
    
    decomposer = ProblemDecomposer()
    
    test_content = """
def function_one():
    return "Hello World"

def function_two():
    return "Goodbye World"

This is some documentation text.
It explains how the functions work.
"""
    
    strategies = [
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
        DecompositionStrategy.SEMANTIC
    ]
    
    for strategy in strategies:
        result = decomposer.decompose_content(
            content=test_content,
            strategy=strategy,
            max_components=5,
            min_component_size=10
        )
        
        print(f"✅ {strategy.value}: {len(result.components)} components, quality: {result.quality_score:.2f}")
    
    return True


def test_history_management():
    """Test decomposition history"""
    print("\nTesting history management...")
    
    decomposer = ProblemDecomposer()
    
    # Perform multiple decompositions
    for i in range(3):
        content = f"Test content number {i+1} with some text."
        decomposer.decompose_content(content)
    
    history = decomposer.get_decomposition_history()
    print(f"✅ History contains {len(history)} entries")
    
    decomposer.clear_history()
    history = decomposer.get_decomposition_history()
    print(f"✅ History cleared, now contains {len(history)} entries")
    
    return True


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_different_strategies()
        test_history_management()
        print("\n🎉 All tests passed! Problem decomposition system is working.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()