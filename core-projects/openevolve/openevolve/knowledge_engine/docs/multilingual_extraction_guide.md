# Bilingual Extraction Guide

Complete guide for multilingual knowledge extraction using OneKE integration.

## Overview

The OneKE integration enables bilingual (English/Chinese) knowledge extraction with:
- Cross-lingual entity recognition
- Multilingual relationship extraction
- Language-agnostic deduplication
- Unified knowledge graph generation

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Language Detection](#language-detection)
6. [Cross-Lingual Operations](#cross-lingual-operations)
7. [Examples](#examples)

## Features

### Supported Languages

- **English**: Full support
- **Chinese (Simplified)**: Full support
- **Chinese (Traditional)**: Full support
- **Other Languages**: Experimental support via translation

### Capabilities

```python
# Extract from English
text_en = "Apple Inc. is headquartered in Cupertino, California."
graph_en = await extract_bilingual(text_en)

# Extract from Chinese
text_zh = "苹果公司总部位于加利福尼亚州库比蒂诺。"
graph_zh = await extract_bilingual(text_zh)

# Both produce equivalent graphs
```

## Installation

```bash
# Install OneKE dependencies
pip install oneke
pip install jieba  # Chinese tokenization
pip install google-trans  # Translation (optional)
```

## Configuration

```yaml
# knowledge_engine/config/oneke.yaml
languages:
  primary: en
  secondary: zh
  fallback_translation: true

extraction:
  cross_lingual: true
  preserve_language: true
  merge_cross_lingual: true

models:
  en: english-bert-base
  zh: chinese-bert-base
  multilingual: mbert-base

translation:
  enabled: true
  service: google  # or bing, azure
  cache_translations: true
```

## Usage

### Basic Extraction

```python
from knowledge_engine.integrations.oneke_integration import OneKEIntegration

# Initialize
oneke = OneKEIntegration()

# Extract from English text
text_en = """
Tesla Inc. is an electric vehicle and clean energy company
based in Palo Alto, California.
"""

graph = await oneke.extract(text_en, language='en')

# Extract from Chinese text
text_zh = """
特斯拉公司是一家位于加利福尼亚州帕洛阿尔托的
电动汽车和清洁能源公司。
"""

graph = await oneke.extract(text_zh, language='zh')
```

### Automatic Language Detection

```python
# Auto-detect language
text = "This is mixed English和中文 text."

graph = await oneke.extract(text, language='auto')

# Detected language is stored
print(f"Detected: {graph.metadata['detected_language']}")
```

## Language Detection

```python
from langdetect import detect, detect_langs

def detect_language(text: str) -> str:
    """Detect primary language."""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English

def detect_languages(text: str) -> dict:
    """Detect all languages with confidence."""
    try:
        langs = detect_langs(text)
        return {lang.lang: lang.prob for lang in langs}
    except:
        return {'en': 1.0}

# Examples
detect_language("This is English")  # 'en'
detect_language("这是中文")  # 'zh-cn'

detect_languages("Mixed English和中文")
# {'en': 0.6, 'zh-cn': 0.4}
```

## Cross-Lingual Operations

### Entity Alignment

```python
# Align entities across languages
graph_en = await oneke.extract("Apple is a company", language='en')
graph_zh = await oneke.extract("苹果是一家公司", language='zh')

# Align entities
aligned = await oneke.align_entities(graph_en, graph_zh)

# Result:
# {
#     "Apple": "苹果",
#     "company": "公司"
# }
```

### Cross-Lingual Query

```python
# Query English knowledge with Chinese
results = await oneke.search(
    query="苹果",  # Chinese query
    knowledge_language='en'  # Search English knowledge
)

# Or vice versa
results = await oneke.search(
    query="Apple",
    knowledge_language='zh'
)
```

## Examples

### Example 1: Bilingual Document Processing

```python
document = """
Apple Inc. (苹果公司) is a multinational technology company
headquartered in Cupertino (库比蒂诺), California.
"""

# Extract bilingual knowledge
graph = await oneke.extract(document, language='auto')

# Entities from both languages
print(graph.entities)
# ['Apple Inc.', '苹果公司', 'Cupertino', '库比蒂诺', 'California']

# Relationships
for subj, pred, obj in graph.relationships:
    print(f"{subj} --[{pred}]--> {obj}")
# Apple Inc. --[headquartered_in]--> Cupertino
# 苹果公司 --[总部位于]--> 库比蒂诺
```

### Example 2: Cross-Lingual Deduplication

```python
# Extract from two sources
text_en = "Tim Cook is the CEO of Apple"
text_zh = "蒂姆·库克是苹果公司的首席执行官"

graph_en = await oneke.extract(text_en, language='en')
graph_zh = await oneke.extract(text_zh, language='zh')

# Merge and deduplicate
merged = await oneke.merge_cross_lingual(graph_en, graph_zh)

# Result: Tim Cook and 蒂姆·库ck are merged
# Apple and 苹果公司 are merged
```

### Example 3: Translation-Aware Extraction

```python
# Extract with translation
text = "Elon Musk founded SpaceX"

# Extract in English
graph_en = await oneke.extract(text, language='en')

# Translate and extract in Chinese
text_zh = translate_to_chinese(text)
# "埃隆·马斯克创立了SpaceX"

graph_zh = await oneke.extract(text_zh, language='zh')

# Link translations
await oneke.link_translations(graph_en, graph_zh)

# Query can now find results in both languages
results = await oneke.search("马斯克")  # Finds "Elon Musk"
```

## Best Practices

1. **Specify language when known**:
```python
# Good
graph = await oneke.extract(text, language='zh')

# Less accurate
graph = await oneke.extract(text, language='auto')
```

2. **Use language metadata**:
```python
graph = await oneke.extract(text)
graph.metadata['source_language'] = 'zh'
graph.metadata['target_language'] = 'en'
```

3. **Handle mixed content**:
```python
# Detect and process separately
for paragraph in text.split('\n'):
    lang = detect_language(paragraph)
    graph = await oneke.extract(paragraph, language=lang)
```

## Troubleshooting

**Issue**: Poor extraction from Chinese text

**Solution**: Ensure proper tokenization
```python
import jieba

# Preprocess Chinese text
def preprocess_chinese(text):
    return ' '.join(jieba.cut(text))

text_zh = preprocess_chinese("苹果公司总部在库比蒂诺")
# "苹果 公司 总部 在 库比蒂诺"

graph = await oneke.extract(text_zh, language='zh')
```

## Next Steps

- [Extraction Pipeline Guide](kg_generation_pipeline_guide.md)
- [API Reference](api/bilingual_extraction_api.md)
