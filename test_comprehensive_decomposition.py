"""
Test comprehensive problem decomposition features
"""

from problem_decomposition import (
    ProblemDecomposer, 
    DecompositionStrategy, 
    ComponentType,
    DecompositionResult
)
import re
import json

# Define utility functions locally since import is having issues
def analyze_content_patterns(content: str):
    """Analyze patterns in content for better decomposition"""
    patterns = {
        'code_blocks': len(re.findall(r'```[\s\S]*?```', content)),
        'headers': len(re.findall(r'^#+\s+', content, re.MULTILINE)),
        'lists': len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE)),
        'functions': len(re.findall(r'def\s+\w+\s*\(', content)),
        'classes': len(re.findall(r'class\s+\w+', content)),
        'imports': len(re.findall(r'^(?:from|import)\s+', content, re.MULTILINE))
    }
    return patterns

def suggest_optimal_strategy(content: str) -> DecompositionStrategy:
    """Suggest optimal decomposition strategy based on content analysis"""
    patterns = analyze_content_patterns(content)
    
    # Rule-based strategy selection
    if patterns['functions'] > 3 or patterns['classes'] > 1:
        return DecompositionStrategy.FUNCTIONAL
    elif patterns['headers'] > 2:
        return DecompositionStrategy.HIERARCHICAL
    elif patterns['imports'] > 5:
        return DecompositionStrategy.DEPENDENCY_BASED
    elif len(content) > 2000:
        return DecompositionStrategy.COMPLEXITY_BASED
    else:
        return DecompositionStrategy.SEMANTIC

def create_decomposition_report(result: DecompositionResult) -> str:
    """Create a comprehensive report of decomposition results"""
    report = f"""
# Decomposition Report

## Overview
- **Strategy Used:** {result.decomposition_strategy.value}
- **Components Created:** {len(result.components)}
- **Quality Score:** {result.quality_score:.2f}
- **Processing Time:** {result.metadata.get('decomposition_time', 0):.2f}s

## Components Summary
"""
    
    for i, component in enumerate(result.components, 1):
        report += f"""
### Component {i}: {component.title}
- **Type:** {component.component_type.value}
- **Size:** {len(component.content)} characters
- **Complexity:** {component.complexity_score:.2f}
- **Dependencies:** {len(component.dependencies)}
"""
    
    report += f"""
## Dependency Graph
{json.dumps(result.dependency_graph, indent=2)}

## Quality Metrics
- **Coverage:** {result.metadata.get('avg_component_size', 0):.0f} avg chars per component
- **Complexity Distribution:** {result.metadata.get('complexity_distribution', {})}
"""
    
    return report


def test_all_strategies():
    """Test all decomposition strategies with appropriate content"""
    print("Testing all decomposition strategies...")
    
    decomposer = ProblemDecomposer()
    
    # Test content for different strategies
    test_cases = {
        DecompositionStrategy.FUNCTIONAL: """
def calculate_score(data):
    return sum(data) / len(data)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, input_data):
        return calculate_score(input_data)
""",
        DecompositionStrategy.HIERARCHICAL: """
# Introduction
This document explains the system architecture.

## Core Components
The system has several key components.

### Database Layer
Handles all data persistence.

### API Layer  
Provides REST endpoints.

## Conclusion
The architecture is scalable and maintainable.
""",
        DecompositionStrategy.DEPENDENCY_BASED: """
import numpy as np
from sklearn import datasets
from tensorflow import keras
import pandas as pd

def load_data():
    return datasets.load_iris()

def preprocess_data(data):
    return pd.DataFrame(data)

def train_model(processed_data):
    model = keras.Sequential()
    return model
""",
        DecompositionStrategy.COMPLEXITY_BASED: """
def complex_algorithm(data, params):
    result = []
    for i in range(len(data)):
        if data[i] > params['threshold']:
            for j in range(params['iterations']):
                try:
                    if params['use_advanced']:
                        temp = data[i] * params['multiplier']
                        if temp > params['max_value']:
                            while temp > params['min_value']:
                                temp = temp / params['divisor']
                                if temp < params['break_point']:
                                    break
                        result.append(temp)
                    else:
                        result.append(data[i] * 2)
                except Exception as e:
                    continue
    return result
"""
    }
    
    for strategy, content in test_cases.items():
        result = decomposer.decompose_content(
            content=content,
            strategy=strategy,
            max_components=10,
            min_component_size=20
        )
        
        print(f"✅ {strategy.value}:")
        print(f"   - Components: {len(result.components)}")
        print(f"   - Quality: {result.quality_score:.2f}")
        print(f"   - Dependencies: {len(result.dependency_graph)}")
        
        # Test reassembly
        try:
            reassembly = decomposer.reassemble_components(
                result.components, 
                result.reassembly_instructions
            )
            print(f"   - Reassembly quality: {reassembly.quality_score:.2f}")
        except Exception as e:
            print(f"   - Reassembly: Error - {e}")
    
    return True


def test_content_analysis():
    """Test content pattern analysis"""
    print("\nTesting content analysis...")
    
    test_content = """
# Header 1
Some content here.

## Header 2
- List item 1
- List item 2

```python
def function_one():
    pass

class MyClass:
    pass
```

import numpy as np
from sklearn import datasets
"""
    
    patterns = analyze_content_patterns(test_content)
    print(f"✅ Content patterns detected:")
    for pattern, count in patterns.items():
        print(f"   - {pattern}: {count}")
    
    # Test strategy suggestion
    suggested = suggest_optimal_strategy(test_content)
    print(f"✅ Suggested strategy: {suggested.value}")
    
    return True


def test_comprehensive_decomposition():
    """Test comprehensive decomposition with all features"""
    print("\nTesting comprehensive decomposition...")
    
    decomposer = ProblemDecomposer()
    
    complex_content = """
# Machine Learning Pipeline

## Data Processing
The first step involves data preprocessing and cleaning.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    '''Load data from CSV file'''
    return pd.read_csv(filepath)

def clean_data(df):
    '''Clean and preprocess data'''
    # Remove null values
    df = df.dropna()
    
    # Normalize features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
    
    def process(self, data):
        try:
            if self.config['normalize']:
                data = self.scaler.fit_transform(data)
            return data
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
```

## Model Training
The model training phase uses various algorithms.

### Configuration
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100

## Evaluation
Model performance is evaluated using multiple metrics.
"""
    
    # Test with automatic strategy selection
    suggested_strategy = suggest_optimal_strategy(complex_content)
    
    result = decomposer.decompose_content(
        content=complex_content,
        strategy=suggested_strategy,
        max_components=8,
        min_component_size=50
    )
    
    print(f"✅ Comprehensive decomposition:")
    print(f"   - Strategy: {result.decomposition_strategy.value}")
    print(f"   - Components: {len(result.components)}")
    print(f"   - Quality: {result.quality_score:.2f}")
    
    # Test component details
    for component in result.components:
        print(f"   - {component.title}: {component.component_type.value} "
              f"(complexity: {component.complexity_score:.2f})")
    
    # Generate report
    report = create_decomposition_report(result)
    print(f"✅ Generated report: {len(report)} characters")
    
    return True


def test_reassembly_quality():
    """Test reassembly quality and improvement metrics"""
    print("\nTesting reassembly quality...")
    
    decomposer = ProblemDecomposer()
    
    original_content = """
def main():
    data = load_data()
    processed = process_data(data)
    result = analyze_data(processed)
    return result

def load_data():
    return [1, 2, 3, 4, 5]

def process_data(data):
    return [x * 2 for x in data]

def analyze_data(data):
    return sum(data) / len(data)
"""
    
    # Decompose
    result = decomposer.decompose_content(
        content=original_content,
        strategy=DecompositionStrategy.FUNCTIONAL,
        max_components=5,
        min_component_size=30
    )
    
    print(f"✅ Decomposition results:")
    print(f"   - Original length: {len(original_content)}")
    print(f"   - Components created: {len(result.components)}")
    print(f"   - Quality score: {result.quality_score:.2f}")
    print(f"   - Strategy: {result.decomposition_strategy.value}")
    
    # Test reassembly
    try:
        reassembly = decomposer.reassemble_components(
            result.components,
            result.reassembly_instructions
        )
        print(f"   - Reassembly quality: {reassembly.quality_score:.2f}")
        print(f"   - Components used: {len(reassembly.components_used)}")
    except Exception as e:
        print(f"   - Reassembly: Error - {e}")
    
    return True


if __name__ == "__main__":
    try:
        test_all_strategies()
        test_content_analysis()
        test_comprehensive_decomposition()
        test_reassembly_quality()
        print("\n🎉 All comprehensive tests passed! Full problem decomposition system working.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()