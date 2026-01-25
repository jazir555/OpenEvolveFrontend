# Bilingual Extraction Tutorial

**OpenEvolve Knowledge Engine - OneKE Bilingual Processing**

Complete tutorial for extracting knowledge from English, Chinese, and mixed-language documents using OneKE.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [English Document Extraction](#english-document-extraction)
3. [Chinese Document Extraction](#chinese-document-extraction)
4. [Mixed-Language Documents](#mixed-language-documents)
5. [Schema Definition for Bilingual](#schema-definition-for-bilingual)
6. [Code Examples](#code-examples)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)

---

## Getting Started

### Prerequisites

```bash
# Install dependencies
pip install transformers torch pydantic pyyaml

# Set environment variables
export ONEKE_MODEL_NAME="oneke/OneKE-13B"
export ONEKE_DEVICE="cuda"
export ONEKE_QUANTIZATION="int4"
```

### Basic Setup

```python
import asyncio
from knowledge_engine.integrations.oneke.model_adapter import (
    OneKEModelAdapter, ModelConfig, Language
)

async def setup():
    # Initialize adapter
    config = ModelConfig(
        model_name="oneke/OneKE-13B",
        device="cuda",
        quantization="int4"
    )

    adapter = OneKEModelAdapter(config)
    await adapter.load_model()

    return adapter

# Run setup
adapter = asyncio.run(setup())
```

---

## English Document Extraction

### Example 1: News Article

```python
async def extract_english_news():
    text = """
    Apple Inc. unveiled its latest iPhone today, featuring a revolutionary
    A17 Pro chip and titanium design. The tech giant, headquartered in
    Cupertino, California, announced that the new device will start at $799.

    CEO Tim Cook emphasized the company's commitment to innovation during
    the launch event at Apple Park. The iPhone 15 Pro represents a significant
    leap forward in smartphone technology, Cook said.
    """

    # Extract entities
    result = await adapter.extract_entities(
        text=text,
        language=Language.ENGLISH,
        correlation_id="news_en_001"
    )

    print("Extracted Entities:")
    for entity in result.entities:
        print(f"  {entity['name']} ({entity.get('type', 'Unknown')})")

    # Expected output:
    # Apple Inc. (Organization)
    # iPhone (Product)
    # A17 Pro (Product)
    # Cupertino (Location)
    # California (Location)
    # Tim Cook (Person)
    # Apple Park (Location)

    return result

asyncio.run(extract_english_news())
```

### Example 2: Academic Paper

```python
async def extract_academic_paper():
    text = """
    Deep learning has revolutionized natural language processing. In 2017,
    Vaswani et al. introduced the Transformer architecture, which replaced
    recurrent neural networks in most NLP tasks.

    Researchers at Google Brain proposed the self-attention mechanism,
    allowing models to process sequences in parallel. BERT, released by
    Google in 2018, achieved state-of-the-art results on 11 NLP benchmarks.
    """

    # Extract relations
    result = await adapter.extract_relations(
        text=text,
        language=Language.ENGLISH,
        correlation_id="academic_en_001"
    )

    print("Extracted Relations:")
    for rel in result.relations:
        print(f"  {rel}")

    # Expected relations:
    # (Deep learning) -> [revolutionized] -> (NLP)
    # (Vaswani et al.) -> [introduced] -> (Transformer)
    # (Transformer) -> [replaced] -> (RNN)
    # (Google Brain) -> [proposed] -> (self-attention)
    # (Google) -> [released] -> (BERT)

    return result

asyncio.run(extract_academic_paper())
```

### Example 3: Financial Report

```python
async def extract_financial_data():
    text = """
    Microsoft Corporation reported revenue of $56.2 billion for the fiscal
    quarter ending September 30, 2023, representing an 13% increase from
    the prior year.

    Azure cloud services grew 29% year-over-year, reaching $32.3 billion
    in revenue. LinkedIn revenue increased by 8% to $3.7 billion.

    The company returned $9.1 billion to shareholders through dividends
    and share repurchases during the quarter.
    """

    # Extract triples (subject, predicate, object)
    result = await adapter.extract_triples(
        text=text,
        language=Language.ENGLISH,
        correlation_id="financial_en_001"
    )

    print("Extracted Triples:")
    for triple in result.triples:
        print(f"  ({triple['subject']}, {triple['predicate']}, {triple['object']})")

    # Expected triples:
    # (Microsoft, reported, $56.2 billion revenue)
    # (Azure, grew, 29%)
    # (Azure, reached, $32.3 billion)
    # (LinkedIn, increased, 8%)
    # (LinkedIn, reached, $3.7 billion)
    # (Microsoft, returned, $9.1 billion)

    return result

asyncio.run(extract_financial_data())
```

---

## Chinese Document Extraction

### Example 4: Chinese News Article

```python
async def extract_chinese_news():
    text = """
    中国科学技术大学潘建伟团队在量子通信领域取得重大突破。
    该团队成功实现了千公里级星地双向量子通信，为构建全球量子通信网络
    奠定了基础。

    这一成果发表在《自然》杂志上，标志着中国在量子科技领域的
    世界领先地位。潘建伟是中国科学院院士，被誉为"量子之父"。
    """

    # Extract entities
    result = await adapter.extract_entities(
        text=text,
        language=Language.CHINESE,
        correlation_id="news_zh_001"
    )

    print("提取的实体:")
    for entity in result.entities:
        print(f"  {entity['name']} ({entity.get('type', 'Unknown')})")

    # Expected output:
    # 中国科学技术大学 (Organization)
    # 潘建伟 (Person)
    # 量子通信 (Technology)
    # 自然 (Publication)
    # 中国科学院 (Organization)

    return result

asyncio.run(extract_chinese_news())
```

### Example 5: Chinese Academic Paper

```python
async def extract_chinese_academic():
    text = """
    深度学习在计算机视觉领域取得了显著进展。2012年，AlexNet模型
    在ImageNet竞赛中取得了突破性成绩，引发了深度学习的研究热潮。

    卷积神经网络（CNN）成为图像识别的主流方法。随后，ResNet、
    DenseNet等网络结构不断刷新图像分类的准确率记录。

    在目标检测领域，Faster R-CNN、YOLO和SSD等算法相继提出，
    显著提升了检测速度和精度。
    """

    # Extract relations
    result = await adapter.extract_relations(
        text=text,
        language=Language.CHINESE,
        correlation_id="academic_zh_001"
    )

    print("提取的关系:")
    for rel in result.relations:
        print(f"  {rel}")

    # Expected relations:
    # (深度学习) -> [取得进展] -> (计算机视觉)
    # (AlexNet) -> [取得成绩] -> (ImageNet)
    # (CNN) -> [成为方法] -> (图像识别)
    # (ResNet) -> [刷新记录] -> (图像分类)
    # (Faster R-CNN) -> [提升] -> (检测速度)

    return result

asyncio.run(extract_chinese_academic())
```

### Example 6: Chinese Financial Report

```python
async def extract_chinese_financial():
    text = """
    腾讯控股有限公司发布2023年第三季度财报，营收达1546亿元人民币，
    同比增长10%。

    游戏业务收入为765亿元，占总收入的49%。广告收入增长20%至
    257亿元。金融科技及企业服务收入达到520亿元，同比增长16%。

    本季度公司净利润为401亿元人民币，同比增长39%。
    """

    # Extract triples
    result = await adapter.extract_triples(
        text=text,
        language=Language.CHINESE,
        correlation_id="financial_zh_001"
    )

    print("提取的三元组:")
    for triple in result.triples:
        print(f"  ({triple['subject']}, {triple['predicate']}, {triple['object']})")

    # Expected triples:
    # (腾讯, 发布, 财报)
    # (腾讯, 营收, 1546亿元)
    # (游戏, 收入, 765亿元)
    # (广告, 增长, 20%)
    # (广告, 收入, 257亿元)
    # (腾讯, 净利润, 401亿元)

    return result

asyncio.run(extract_chinese_financial())
```

---

## Mixed-Language Documents

### Example 7: Code-Switching Content

```python
async def extract_code_switching():
    text = """
    华为 (Huawei) 今天发布了新款旗舰手机 Mate 60 Pro。
    The device features the company's latest Kirin 9000s processor
    and HarmonyOS 4.0 operating system.

    该设备的起售价为6999元。Pre-orders start from September 10th
    in China (中国). Richard Yu, CEO of Huawei's Consumer Business Group,
    said the phone represents a "major breakthrough" (重大突破).
    """

    # Extract with bilingual mode
    result = await adapter.extract_entities(
        text=text,
        language=Language.BILINGUAL,
        correlation_id="mixed_001"
    )

    print("Bilingual Extraction Results:")
    for entity in result.entities:
        print(f"  {entity['name']} ({entity.get('type', 'Unknown')}) - {entity.get('language', 'unknown')}")

    # Expected:
    # 华为 - Organization
    # Mate 60 Pro - Product
    # Kirin 9000s - Product
    # HarmonyOS - Software
    # China - Location
    # 中国 - Location
    # Richard Yu - Person

    return result

asyncio.run(extract_code_switching())
```

### Example 8: Technical Documentation

```python
async def extract_technical_docs():
    text = """
    PyTorch is an open-source machine learning library developed by
    Facebook's AI Research lab (FAIR). PyTorch 是基于 Torch 库的
    Python 实现，提供了两个主要功能：张量计算 (tensor computation)
    和深度神经网络。

    The library supports both CPU and GPU computations. 该库广泛用于
    计算机视觉和自然语言处理领域。主要应用包括：
    - Tesla Autopilot (自动驾驶)
    - Uber's Pyro (概率编程)
    - Hugging Face Transformers (自然语言处理)
    """

    result = await adapter.extract_entities(
        text=text,
        language=Language.BILINGUAL,
        correlation_id="tech_001"
    )

    print("Technical Entities:")
    for entity in result.entities:
        print(f"  {entity['name']} - {entity.get('type', 'Unknown')}")

    # Expected:
    # PyTorch - Library
    # Facebook - Organization
    # FAIR - Organization
    # Torch - Library
    # Python - Language
    # Tesla - Organization
    # Uber - Organization
    # Hugging Face - Organization

    return result

asyncio.run(extract_technical_docs())
```

---

## Schema Definition for Bilingual

### English Schema

```python
english_schema = {
    "name": "tech_news_en",
    "version": "1.0.0",
    "description": "Schema for English tech news extraction",
    "entity_types": [
        {
            "name": "Company",
            "description": "Technology company",
            "examples": ["Apple", "Microsoft", "Google", "Amazon"]
        },
        {
            "name": "Product",
            "description": "Product or service",
            "examples": ["iPhone", "Azure", "Android", "Kindle"]
        },
        {
            "name": "Person",
            "description": "Person (CEO, founder, etc.)",
            "examples": ["Tim Cook", "Satya Nadella", "Sundar Pichai"]
        },
        {
            "name": "Technology",
            "description": "Technology or framework",
            "examples": ["AI", "Machine Learning", "Cloud Computing"]
        }
    ],
    "relation_types": [
        {
            "name": "released",
            "description": "Company released product",
            "domain": "Company",
            "range": "Product"
        },
        {
            "name": "led_by",
            "description": "Company led by person",
            "domain": "Company",
            "range": "Person"
        },
        {
            "name": "uses",
            "description": "Product uses technology",
            "domain": "Product",
            "range": "Technology"
        }
    ]
}
```

### Chinese Schema

```python
chinese_schema = {
    "name": "tech_news_zh",
    "version": "1.0.0",
    "description": "科技新闻中文抽取模式",
    "entity_types": [
        {
            "name": "公司",
            "description": "技术公司",
            "examples": ["苹果", "微软", "谷歌", "亚马逊", "华为", "腾讯", "阿里巴巴"]
        },
        {
            "name": "产品",
            "description": "产品或服务",
            "examples": ["iPhone", "Azure", "安卓", "Kindle", "微信", "支付宝"]
        },
        {
            "name": "人物",
            "description": "人物（CEO、创始人等）",
            "examples": ["蒂姆·库克", "萨提亚·纳德拉", "马云", "马化腾"]
        },
        {
            "name": "技术",
            "description": "技术或框架",
            "examples": ["人工智能", "机器学习", "云计算", "区块链"]
        }
    ],
    "relation_types": [
        {
            "name": "发布",
            "description": "公司发布产品",
            "domain": "公司",
            "range": "产品"
        },
        {
            "name": "领导",
            "description": "人物领导公司",
            "domain": "人物",
            "range": "公司"
        },
        {
            "name": "使用",
            "description": "产品使用技术",
            "domain": "产品",
            "range": "技术"
        }
    ]
}
```

### Bilingual Schema

```python
bilingual_schema = {
    "name": "tech_news_bilingual",
    "version": "1.0.0",
    "description": "Bilingual tech news schema",
    "entity_types": [
        {
            "name": "Company/公司",
            "description": "Technology company / 技术公司",
            "examples_en": ["Apple", "Microsoft", "Huawei", "Tencent"],
            "examples_zh": ["苹果", "微软", "华为", "腾讯"],
            "translations": {
                "Apple": "苹果",
                "Microsoft": "微软",
                "Google": "谷歌",
                "Huawei": "华为",
                "Tencent": "腾讯",
                "Alibaba": "阿里巴巴"
            }
        },
        {
            "name": "Product/产品",
            "description": "Product or service / 产品或服务",
            "examples_en": ["iPhone", "WeChat", "Android"],
            "examples_zh": ["iPhone", "微信", "安卓"],
            "translations": {
                "iPhone": "iPhone",
                "WeChat": "微信",
                "Android": "安卓"
            }
        },
        {
            "name": "Person/人物",
            "description": "Person / 人物",
            "examples_en": ["Tim Cook", "Pony Ma"],
            "examples_zh": ["蒂姆·库克", "马云"],
            "translations": {
                "Tim Cook": "蒂姆·库克",
                "Pony Ma": "马化腾",
                "Richard Yu": "余承东"
            }
        }
    ],
    "relation_types": [
        {
            "name": "released/发布",
            "description": "Company released product / 公司发布产品",
            "domain": "Company/公司",
            "range": "Product/产品"
        }
    ]
}
```

---

## Code Examples

### Example 9: Complete Bilingual Pipeline

```python
import asyncio
from knowledge_engine.integrations.oneke.model_adapter import OneKEModelAdapter, ModelConfig, Language
from knowledge_engine.integrations.oneke.schema_manager import OneKESchemaManager

async def bilingual_pipeline():
    # Initialize
    adapter = OneKEModelAdapter(ModelConfig(quantization="int4"))
    await adapter.load_model()

    schema_manager = OneKESchemaManager()
    schema = await schema_manager.load_schema("general")

    # Process bilingual document
    text = """
    脸书 (Facebook) 母公司 Meta 发布了最新季度财报。
    Revenue increased by 23% to $34.1 billion, beating analyst expectations.
    广告收入达到288亿美元，占总收入的84%。
    """

    # Extract entities
    result = await adapter.extract_entities(
        text=text,
        schema=schema.dict(),
        language=Language.BILINGUAL
    )

    # Print results
    print("Entities Extracted:")
    for entity in result.entities:
        lang_tag = entity.get('language', 'unknown')
        print(f"  [{lang_tag}] {entity.get('name')} - {entity.get('type', 'Unknown')}")

    # Extract relations
    result_rel = await adapter.extract_relations(
        text=text,
        schema=schema.dict(),
        language=Language.BILINGUAL
    )

    print("\nRelations Extracted:")
    for rel in result_rel.relations:
        print(f"  {rel}")

    # Cleanup
    await adapter.unload()

    return result, result_rel

# Run
entities, relations = asyncio.run(bilingual_pipeline())
```

### Example 10: Batch Processing

```python
async def batch_process_bilingual(documents):
    """
    Process multiple documents in parallel.

    Args:
        documents: List of (text, language) tuples
    """
    adapter = OneKEModelAdapter(ModelConfig())
    await adapter.load_model()

    # Process all documents
    results = []
    for text, language in documents:
        result = await adapter.extract_entities(
            text=text,
            language=language
        )
        results.append(result)

    await adapter.unload()
    return results

# Example usage
documents = [
    ("Apple released new iPhone today", Language.ENGLISH),
    ("华为今天发布新手机", Language.CHINESE),
    ("Google announced Pixel 8 / 谷歌发布Pixel 8", Language.BILINGUAL)
]

results = asyncio.run(batch_process_bilingual(documents))
```

---

## Best Practices

### 1. Language Specification

```python
# ✅ Good: Explicit language specification
result = await adapter.extract_entities(
    text=text,
    language=Language.CHINESE  # Force Chinese
)

# ❌ Bad: Let model guess for ambiguous text
result = await adapter.extract_entities(
    text=text  # May get wrong language
)
```

### 2. Schema-Guided Extraction

```python
# ✅ Good: Use schema for better results
schema = {
    "entity_types": [
        {"name": "Company", "examples": ["Apple", "华为"]}
    ]
}
result = await adapter.extract_entities(text, schema=schema)

# ❌ Bad: No schema guidance
result = await adapter.extract_entities(text)
```

### 3. Few-Shot Examples

```python
# ✅ Good: Provide examples for complex patterns
examples = [
    {
        "text": "Apple CEO Tim Cook announced iPhone 15",
        "entities": [
            {"name": "Apple", "type": "Company"},
            {"name": "Tim Cook", "type": "Person"},
            {"name": "iPhone 15", "type": "Product"}
        ]
    }
]
result = await adapter.extract_entities(
    text="华为CEO余承东发布Mate 60",
    few_shot_examples=examples
)
```

### 4. Temperature Tuning

```python
# Low temperature for deterministic extraction
config = ModelConfig(temperature=0.01)  # Consistent results

# Higher temperature for more diverse extraction
config = ModelConfig(temperature=0.3)   # More variation
```

### 5. Text Preprocessing

```python
# ✅ Good: Clean text before extraction
def clean_text(text):
    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')

    # Remove special characters if needed
    # text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

    return text

cleaned = clean_text(raw_text)
result = await adapter.extract_entities(cleaned)
```

---

## Common Pitfalls

### Pitfall 1: Mixed Language Without Explicit Mode

```python
# ❌ Wrong: Using EN mode for mixed text
text = "Apple 发布了 iPhone"
result = await adapter.extract_entities(
    text=text,
    language=Language.ENGLISH  # Will miss Chinese
)

# ✅ Correct: Use bilingual mode
result = await adapter.extract_entities(
    text=text,
    language=Language.BILINGUAL
)
```

### Pitfall 2: Not Using Schema

```python
# ❌ Wrong: No schema
result = await adapter.extract_entities(text)

# ✅ Correct: With schema
schema = {
    "entity_types": [
        {"name": "Company", "examples": ["Apple", "华为"]}
    ]
}
result = await adapter.extract_entities(text, schema=schema)
```

### Pitfall 3: Ignoring Confidence Scores

```python
# ❌ Wrong: Using all results blindly
entities = result.entities

# ✅ Correct: Filter by confidence
high_conf = [e for e in result.entities if e.get('confidence', 0) > 0.8]
```

### Pitfall 4: Not Handling Errors

```python
# ❌ Wrong: No error handling
result = await adapter.extract_entities(text)

# ✅ Correct: Handle errors
try:
    result = await adapter.extract_entities(text)
except RuntimeError as e:
    logger.error(f"Extraction failed: {e}")
    result = None
```

---

## Next Steps

- [OneKE Integration Guide](ONEKE_INTEGRATION_GUIDE.md) - Complete integration documentation
- [Schema Definition Guide](SCHEMA_DEFINITION_GUIDE.md) - Create custom schemas
- [Event Extraction Guide](EVENT_EXTRACTION_GUIDE.md) - Extract events
- [API Reference](ONEKE_API_REFERENCE.md) - Complete API documentation

---

**Version:** 1.0.0
**Last Updated:** 2026-01-08
