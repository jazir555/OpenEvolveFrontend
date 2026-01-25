# Bilingual Extraction Tutorial

Hands-on tutorial for bilingual (English/Chinese) knowledge extraction using OneKE.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Language Detection](#language-detection)
3. [Bilingual Entity Extraction](#bilingual-entity-extraction)
4. [Cross-Lingual Entity Linking](#cross-lingual-entity-linking)
5. [Bilingual Relation Extraction](#bilingual-relation-extraction)
6. [Bilingual Event Extraction](#bilingual-event-extraction)
7. [Building Bilingual Knowledge Graphs](#building-bilingual-knowledge-graphs)
8. [Advanced Topics](#advanced-topics)
9. [Real-World Examples](#real-world-examples)

---

## Getting Started

### Prerequisites

```bash
# Install dependencies
pip install torch transformers scikit-learn rapidfuzz

# Set environment variables
export ONEKE_MODEL_NAME=oneke/OneKE-13B
export ONEKE_DEVICE=cuda
```

### Basic Setup

```python
import asyncio
from knowledge_engine.integrations.oneke import (
    OneKEModelAdapter,
    ModelConfig,
    Language
)

async def setup():
    config = ModelConfig(model_name="oneke/OneKE-13B")
    adapter = OneKEModelAdapter(config)
    await adapter.load_model()
    return adapter

adapter = asyncio.run(setup())
```

---

## Language Detection

### Detecting Document Language

```python
from knowledge_engine.integrations.oneke import CrossLingualEntityLinker

async def detect_language_example():
    linker = CrossLingualEntityLinker()

    # English text
    text_en = "Apple announced the new iPhone today."
    lang_en = await linker.detect_language(text_en)
    print(f"Language: {lang_en}")  # Language.ENGLISH

    # Chinese text
    text_zh = "苹果今天发布了新iPhone。"
    lang_zh = await linker.detect_language(text_zh)
    print(f"Language: {lang_zh}")  # Language.CHINESE

    # Bilingual text
    text_bi = "Apple announced iPhone. 苹果发布iPhone。"
    lang_bi = await linker.detect_language(text_bi)
    print(f"Language: {lang_bi}")  # Language.CHINESE (mixed)

asyncio.run(detect_language_example())
```

### Mixed Language Documents

```python
async def process_mixed_document():
    linker = CrossLingualEntityLinker()

    # Document with both languages
    text = """
    苹果公司 (Apple Inc.) 今天宣布了新产品。
    Apple Inc. announced new products today.
    """

    # Detect language
    lang = await linker.detect_language(text)

    # Can still extract (model handles bilingual)
    result = await adapter.extract(
        text=text,
        language=Language.BILINGUAL
    )

    return result
```

---

## Bilingual Entity Extraction

### Example 1: Extracting Bilingual Entities

```python
async def extract_bilingual_entities():
    # Input text (bilingual)
    text = """
    苹果公司 (Apple Inc.) is headquartered in
    库比蒂诺 (Cupertino), California.
    史蒂夫·乔布斯 (Steve Jobs) founded the company.
    """

    # Extract with general schema
    result = await adapter.extract(
        text=text,
        schema=general_schema,
        language=Language.BILINGUAL
    )

    # Process entities
    for entity in result.entities:
        print(f"Entity: {entity['text']}")
        print(f"Type: {entity['type']}")
        print(f"Confidence: {entity.get('confidence', 0.0)}")
        print()

asyncio.run(extract_bilingual_entities())
```

**Expected Output:**
```
Entity: 苹果公司
Type: ORGANIZATION
Confidence: 0.95

Entity: Apple Inc.
Type: ORGANIZATION
Confidence: 0.95

Entity: 库比蒂诺
Type: LOCATION
Confidence: 0.92

Entity: Steve Jobs
Type: PERSON
Confidence: 0.98
```

### Example 2: Creating Bilingual Entities

```python
from knowledge_engine.integrations.oneke import Entity, LinkerLanguage

async def create_bilingual_entities():
    # Create bilingual entity
    apple = Entity(
        entity_id="E1",
        name_en=["Apple Inc.", "Apple"],
        name_zh=["苹果公司", "苹果"],
        aliases_en=["Apple Computer"],
        aliases_zh=["苹果电脑"],
        type="ORGANIZATION",
        language=LinkerLanguage.BILINGUAL,
        confidence=0.95
    )

    # Get all names
    names = apple.get_all_names()
    print(f"English names: {names['en']}")
    print(f"Chinese names: {names['zh']}")

    return apple

asyncio.run(create_bilingual_entities())
```

---

## Cross-Lingual Entity Linking

### Example 1: Matching Across Languages

```python
from knowledge_engine.integrations.oneke import (
    CrossLingualEntityLinker,
    Entity,
    MatchStrategy,
    LinkerLanguage
)

async def cross_lingual_matching():
    linker = CrossLingualEntityLinker()

    # Create entities in different languages
    entity_en = Entity(
        entity_id="E1",
        name_en=["Microsoft Corporation"],
        type="ORGANIZATION",
        language=LinkerLanguage.ENGLISH
    )

    entity_zh = Entity(
        entity_id="E2",
        name_zh=["微软公司"],
        type="ORGANIZATION",
        language=LinkerLanguage.CHINESE
    )

    # Match across languages
    match_result = await linker.match_entities(
        entity_en,
        entity_zh,
        strategy=MatchStrategy.HYBRID  # Try all strategies
    )

    print(f"Matched: {match_result.matched}")
    print(f"Confidence: {match_result.confidence}")
    print(f"Cross-lingual: {match_result.cross_lingual}")
    print(f"Evidence: {match_result.evidence}")

    return match_result

asyncio.run(cross_lingual_matching())
```

### Example 2: Building Bilingual Knowledge Base

```python
async def build_bilingual_kb():
    linker = CrossLingualEntityLinker()

    # Companies
    companies = [
        Entity(
            entity_id="C1",
            name_en=["Apple Inc."],
            name_zh=["苹果公司"],
            type="ORGANIZATION"
        ),
        Entity(
            entity_id="C2",
            name_en=["Microsoft Corporation"],
            name_zh=["微软公司"],
            type="ORGANIZATION"
        ),
        Entity(
            entity_id="C3",
            name_en=["Google LLC"],
            name_zh=["谷歌"],
            type="ORGANIZATION"
        )
    ]

    # Add to index
    for company in companies:
        await linker.add_entity(company)

    # Find duplicates
    clusters = await linker.deduplicate_entities(
        companies,
        strategy=MatchStrategy.HYBRID
    )

    print(f"Found {len(clusters)} duplicate clusters")

    # Export to bilingual KG format
    kg = linker.to_bilingual_kg(companies)

    return kg

asyncio.run(build_bilingual_kb())
```

### Example 3: Translation-Aware Matching

```python
async def translation_aware_matching():
    # Enable translation
    config = LinkerConfig(
        enable_translation=True,
        translation_model="google"
    )
    linker = CrossLingualEntityLinker(config)

    # Entity with only English name
    entity_en = Entity(
        entity_id="E1",
        name_en=["International Business Machines"],
        type="ORGANIZATION"
    )

    # Entity with only Chinese name
    entity_zh = Entity(
        entity_id="E2",
        name_zh=["国际商业机器公司"],
        type="ORGANIZATION"
    )

    # Match with translation
    match_result = await linker.match_entities(
        entity_en,
        entity_zh,
        strategy=MatchStrategy.TRANSLATION
    )

    print(f"Translation used: {match_result.translation_used}")
    print(f"Match confidence: {match_result.confidence}")

asyncio.run(translation_aware_matching())
```

---

## Bilingual Relation Extraction

### Example 1: Extracting Relations

```python
async def extract_bilingual_relations():
    # Bilingual text
    text = """
    史蒂夫·乔布斯 (Steve Jobs) 创立了 (founded)
    苹果公司 (Apple Inc.) in 1976.
    """

    # Extract
    result = await adapter.extract(
        text=text,
        schema=general_schema,
        language=Language.BILINGUAL
    )

    # Process relations
    for relation in result.relations:
        subject = relation.get('subject', '')
        obj = relation.get('object', '')
        rel_type = relation.get('type', '')

        print(f"{subject} --[{rel_type}]--> {obj}")

asyncio.run(extract_bilingual_relations())
```

**Expected Output:**
```
Steve Jobs --[FOUNDED_BY]--> Apple Inc.
史蒂夫·乔布斯 --[FOUNDED_BY]--> 苹果公司
```

### Example 2: Cross-Lingual Relation Alignment

```python
async def align_relations():
    linker = CrossLingualEntityLinker()

    # English relations
    relations_en = [
        {
            "type": "WORKS_FOR",
            "subject": "Tim Cook",
            "object": "Apple Inc."
        }
    ]

    # Chinese relations
    relations_zh = [
        {
            "type": "WORKS_FOR",
            "subject": "蒂姆·库克",
            "object": "苹果公司"
        }
    ]

    # Align across languages
    alignments = await linker.align_relations(
        relations_en,
        relations_zh
    )

    print(f"Found {len(alignments)} alignments")

    for rel1, rel2, similarity in alignments:
        print(f"Aligned: {rel1} ~ {rel2} (similarity: {similarity:.2f})")

asyncio.run(align_relations())
```

---

## Bilingual Event Extraction

### Example 1: Extracting Events

```python
from knowledge_engine.integrations.oneke import EventExtractionPipeline

async def extract_bilingual_events():
    pipeline = EventExtractionPipeline()

    # Bilingual text
    text = """
    2007年1月，苹果公司 (Apple Inc.) 宣布了 (announced) iPhone.
    In January 2007, Apple Inc. announced the iPhone.
    2007年6月，iPhone 发布了 (released).
    In June 2007, the iPhone was released.
    """

    # Extract events
    result = await pipeline.extract_complete_pipeline(
        text=text,
        language=Language.BILINGUAL
    )

    # Print events
    for event_data in result['events']:
        print(f"Event: {event_data['event_type']}")
        print(f"Trigger: {event_data['trigger']}")
        print(f"Language: {event_data['language']}")
        print()

asyncio.run(extract_bilingual_events())
```

### Example 2: Building Bilingual Event Chains

```python
async def build_bilingual_event_chains():
    pipeline = EventExtractionPipeline()

    text = """
    首先苹果宣布了产品。
    Then Apple announced the product.
    接着产品发布了。
    Then the product was released.
    最后销量增长了。
    Finally sales increased.
    """

    result = await pipeline.extract_complete_pipeline(
        text=text,
        language=Language.BILINGUAL
    )

    # Print event chains
    for chain_data in result['event_chains']:
        print(f"Chain: {chain_data['chain_id']}")
        print(f"Events: {len(chain_data['events'])}")

        for event in chain_data['events']:
            print(f"  - {event['trigger']} ({event['language']})")

asyncio.run(build_bilingual_event_chains())
```

---

## Building Bilingual Knowledge Graphs

### Example 1: Complete Bilingual KG Construction

```python
async def build_complete_bilingual_kg():
    # Initialize components
    linker = CrossLingualEntityLinker()
    pipeline = EventExtractionPipeline()

    # Input document (bilingual)
    text = """
    苹果公司 (Apple Inc.) 是一家科技公司。
    Apple Inc. is a technology company.

    史蒂夫·乔布斯 (Steve Jobs) 在1976年创立了苹果公司。
    Steve Jobs founded Apple Inc. in 1976.

    2007年，苹果发布了iPhone。
    In 2007, Apple launched the iPhone.

    蒂姆·库克 (Tim Cook) 是现任CEO。
    Tim Cook is the current CEO.
    """

    # Step 1: Extract entities
    result = await adapter.extract(
        text=text,
        schema=general_schema,
        language=Language.BILINGUAL
    )

    # Step 2: Create entity objects
    entities = []
    for entity_data in result.entities:
        entity = Entity(
            entity_id=entity_data['id'],
            name_en=entity_data.get('names_en', []),
            name_zh=entity_data.get('names_zh', []),
            type=entity_data['type']
        )
        entities.append(entity)
        await linker.add_entity(entity)

    # Step 3: Extract events
    event_result = await pipeline.extract_complete_pipeline(
        text=text,
        language=Language.BILINGUAL
    )

    # Step 4: Build bilingual KG
    kg = linker.to_bilingual_kg(entities)

    # Add relations and events
    kg['relations'] = result.relations
    kg['events'] = event_result['events']
    kg['event_chains'] = event_result['event_chains']

    # Save KG
    import json
    with open('bilingual_kg.json', 'w', encoding='utf-8') as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    print("Bilingual KG saved to bilingual_kg.json")
    return kg

asyncio.run(build_complete_bilingual_kg())
```

**Output Structure:**
```json
{
  "nodes": [
    {
      "id": "E1",
      "names": {
        "en": ["Apple Inc."],
        "zh": ["苹果公司"]
      },
      "type": "ORGANIZATION",
      "language": "bilingual"
    }
  ],
  "relations": [
    {
      "subject": "Steve Jobs",
      "object": "Apple Inc.",
      "type": "FOUNDED_BY"
    }
  ],
  "events": [
    {
      "event_id": "EV1",
      "event_type": "LAUNCH",
      "trigger": "announced",
      "language": "bilingual"
    }
  ]
}
```

---

## Advanced Topics

### Handling Code-Switching

```python
async def handle_code_switching():
    linker = CrossLingualEntityLinker()

    # Text with code-switching (mixing languages in sentence)
    text = "Apple 发布了 new iPhone 在 Cupertino."

    # Detect language (will identify as mixed)
    lang = await linker.detect_language(text)

    # Extract (handles code-switching)
    result = await adapter.extract(
        text=text,
        language=Language.BILINGUAL
    )

    return result
```

### Entity Disambiguation

```python
async def disambiguate_entities():
    linker = CrossLingualEntityLinker()

    # Ambiguous entity
    entity = Entity(
        entity_id="E1",
        name_en=["Apple"],
        name_zh=["苹果"],
        type="UNKNOWN"
    )

    # Find candidates
    candidates = await linker.find_candidates(entity, limit=10)

    # Context-based disambiguation
    context = "technology company iPhone Mac"

    # Score candidates based on context
    for candidate_id, score in candidates:
        candidate = linker.entity_index[candidate_id]

        # Check if context matches
        if candidate.type == "ORGANIZATION":
            print(f"Likely match: {candidate.entity_id}")

asyncio.run(disambiguate_entities())
```

### Multi-Lingual Document Processing

```python
async def process_multilingual_corpus():
    linker = CrossLingualEntityLinker()

    # Corpus with multiple languages
    documents = [
        {"text": "Apple announced iPhone.", "lang": "en"},
        {"text": "苹果宣布iPhone。", "lang": "zh"},
        {"text": "Google released Android.", "lang": "en"},
        {"text": "谷歌发布Android。", "lang": "zh"}
    ]

    # Extract from all documents
    all_entities = []

    for doc in documents:
        result = await adapter.extract(
            text=doc['text'],
            language=Language(doc['lang'])
        )

        # Add to linker
        for entity_data in result.entities:
            entity = Entity(
                entity_id=entity_data['id'],
                name_en=entity_data.get('names_en', []),
                name_zh=entity_data.get('names_zh', []),
                type=entity_data['type']
            )
            await linker.add_entity(entity)
            all_entities.append(entity)

    # Deduplicate across all documents
    clusters = await linker.deduplicate_entities(all_entities)

    print(f"Processed {len(documents)} documents")
    print(f"Found {len(clusters)} duplicate clusters")

asyncio.run(process_multilingual_corpus())
```

---

## Real-World Examples

### Example 1: Financial News Analysis

```python
async def analyze_financial_news():
    # Financial news (bilingual)
    news = """
    苹果公司 (Apple Inc.) 今天发布了财报。
    Apple Inc. released earnings report today.

    收入达到 943亿美元 ($94.3 billion)。
    Revenue reached $94.3 billion.

    iPhone销量增长了 (iPhone sales increased).
    """

    # Use general schema
    result = await adapter.extract(
        text=news,
        schema=general_schema,
        language=Language.BILINGUAL
    )

    # Extract monetary values
    money_entities = [
        e for e in result.entities
        if e['type'] == 'MONEY'
    ]

    # Extract events
    pipeline = EventExtractionPipeline()
    events = await pipeline.extract_events(news, Language.BILINGUAL)

    print(f"Financial entities: {len(money_entities)}")
    print(f"Events: {len(events)}")

asyncio.run(analyze_financial_news())
```

### Example 2: Biomedical Literature Processing

```python
async def process_biomedical_text():
    # Load biomedical schema
    manager = OneKESchemaManager()
    bio_schema = await manager.load_schema("schemas/biomedical_schema.json")

    # Biomedical text (bilingual)
    text = """
    COVID-19 (新冠肺炎) is caused by SARS-CoV-2 virus.

    Common symptoms (症状) include fever (发烧), cough (咳嗽).

    Vaccines (疫苗) from Pfizer (辉瑞) and Moderna are effective.
    """

    # Extract with biomedical schema
    result = await adapter.extract(
        text=text,
        schema=bio_schema,
        language=Language.BILINGUAL
    )

    # Extract diseases and drugs
    diseases = [e for e in result.entities if e['type'] == 'DISEASE']
    drugs = [e for e in result.entities if e['type'] == 'DRUG']

    print(f"Diseases: {len(diseases)}")
    print(f"Drugs: {len(drugs)}")

    # Extract treatment relations
    treatments = [
        r for r in result.relations
        if r['type'] == 'TREATS'
    ]

    print(f"Treatments found: {len(treatments)}")

asyncio.run(process_biomedical_text())
```

### Example 3: Legal Document Analysis

```python
async def analyze_legal_document():
    # Load legal schema
    manager = OneKESchemaManager()
    legal_schema = await manager.load_schema("schemas/legal_schema.json")

    # Legal text (bilingual)
    text = """
    苹果公司 (Apple Inc.) 起诉了 (sued) 三星 (Samsung).

    案件编号 (Case number): 11-1846.

    指控 (alleges) 专利侵权 (patent infringement).
    """

    # Extract with legal schema
    result = await adapter.extract(
        text=text,
        schema=legal_schema,
        language=Language.BILINGUAL
    )

    # Extract legal events
    pipeline = EventExtractionPipeline()
    events = await pipeline.extract_events(text, Language.BILINGUAL)

    # Find lawsuit events
    lawsuits = [
        e for e in events
        if e.event_type == EventType.LEGAL
    ]

    print(f"Legal events: {len(lawsuits)}")

asyncio.run(analyze_legal_document())
```

---

## Summary

This tutorial covered:

1. **Language Detection**: Identifying English, Chinese, and bilingual text
2. **Bilingual Entity Extraction**: Extracting entities from both languages
3. **Cross-Lingual Entity Linking**: Matching entities across languages
4. **Bilingual Relation Extraction**: Extracting relationships in both languages
5. **Bilingual Event Extraction**: Detecting events in bilingual documents
6. **Building Bilingual Knowledge Graphs**: Constructing complete bilingual KGs
7. **Advanced Topics**: Code-switching, disambiguation, multi-lingual corpora
8. **Real-World Examples**: Financial, biomedical, and legal domains

### Key Takeaways

- OneKE handles bilingual text natively
- Cross-lingual entity linking matches entities across languages
- Schema-guided extraction works for both languages
- Event extraction preserves temporal information
- Bilingual KGs can represent knowledge in multiple languages

### Next Steps

- Explore domain-specific schemas (biomedical, legal)
- Customize schemas for your use case
- Integrate with your knowledge pipeline
- Scale to document collections

For more details, see:
- [ONEKE_INTEGRATION_GUIDE.md](ONEKE_INTEGRATION_GUIDE.md)
- [SCHEMA_DEFINITION_GUIDE.md](SCHEMA_DEFINITION_GUIDE.md)
