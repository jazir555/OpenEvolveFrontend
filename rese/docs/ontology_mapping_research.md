# Ontology Mapping Research Document

**Agent**: G2 (Ψ₂ Specialist - Ontology Mapping)
**Created**: 2025-12-31
**Status**: 🔄 Research Phase
**Version**: 1.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Definition](#problem-definition)
3. [Ontology Alignment Techniques](#ontology-alignment-techniques)
4. [Knowledge Graph Resources](#knowledge-graph-resources)
5. [Semantic Similarity Methods](#semantic-similarity-methods)
6. [Graph Matching Algorithms](#graph-matching-algorithms)
7. [Proposed Architecture](#proposed-architecture)
8. [Implementation Strategy](#implementation-strategy)
9. [Integration with I_mech](#integration-with-i_mech)
10. [Validation & Testing](#validation--testing)
11. [References](#references)

---

## Executive Summary

**Objective**: Design and implement a semantic ontology mapping system that enables cross-domain knowledge transfer for the I_mech isomorphic resonance engine.

**Key Challenge**: Map concepts and relationships between heterogeneous problem domains (e.g., fluid dynamics → electromagnetism) to enable solution transfer.

**Approach**: Multi-layered semantic similarity combining:
- Lexical similarity (string matching)
- Semantic similarity (embeddings)
- Structural similarity (graph isomorphism)
- Knowledge graph validation (external KGs)

**Target Performance**: >80% mapping accuracy for domain pairs with known isomorphisms.

---

## Problem Definition

### What is Ontology Mapping?

Ontology mapping (also called ontology alignment) is the process of finding correspondences between entities (concepts, relations, attributes) in different ontologies.

### Why is it Critical for I_mech?

The **I_mech** (Isomorphic Mechanism) requires identifying which elements in a source domain correspond to elements in a target domain. For example:

**Example**: Fluid Dynamics → Electricity
- "flow rate" (fluid) ↔ "current" (electricity)
- "pressure" (fluid) ↔ "voltage" (electricity)
- "pipe resistance" (fluid) ↔ "electrical resistance" (electricity)

Without accurate ontology mapping, I_mech cannot:
1. Identify isomorphic domain pairs
2. Transfer solutions between domains
3. Validate mechanistic similarity

### Core Requirements

1. **Concept Mapping**: Map individual concepts between domains
2. **Relation Mapping**: Map relationships between concepts
3. **Structure Alignment**: Align graph structures
4. **Confidence Scoring**: Quantify mapping confidence
5. **Real-Time Performance**: Map domains in <10 seconds

---

## Ontology Alignment Techniques

### 1. Lexical-Based Techniques

#### String Similarity Metrics

**Levenshtein Distance**
- Measures edit operations between strings
- Formula: `Lev(a, b) = minimum edits to transform a → b`
- Normalized: `sim(a, b) = 1 - Lev(a, b) / max(|a|, |b|)`
- Use Case: Exact/near-exact concept names

**Jaro-Winkler Similarity**
- Weighted string matching
- Gives higher weight to prefix matches
- Formula: `JW = J + (l * p * (1 - J))`
  - J = Jaro similarity
  - l = common prefix length (up to 4)
  - p = prefix scaling factor (typically 0.1)
- Use Case: Names with common prefixes (e.g., "velocity" vs "velocity_x")

**N-Gram Overlap**
- Compare character n-grams (typically 2-3 grams)
- Formula: `overlap = |ngrams(a) ∩ ngrams(b)| / |ngrams(a) ∪ ngrams(b)|`
- Use Case: Partial word matching

#### Pros & Cons

**Pros**:
- Fast computation
- No training required
- Works well for similar naming conventions

**Cons**:
- Fails on synonyms (e.g., "fast" ↔ "rapid")
- Language-dependent
- No semantic understanding

---

### 2. Semantic Similarity (Embeddings)

#### Word Embeddings

**Word2Vec** (2013)
- Dense vector representations of words
- Captures semantic relationships via distributional hypothesis
- Architecture: Skip-gram or CBOW
- Dimensionality: 100-300
- Training: Large text corpora (Wikipedia, Common Crawl)
- Formula: `sim(w1, w2) = cos(vec(w1), vec(w2))`

**GloVe** (2014)
- Global Vectors for Word Representation
- Combines matrix factorization and local context
- Captures both global and local semantics
- Formula: `sim(w1, w2) = cos(vec(w1), vec(w2))`

**FastText** (2016)
- Extension of Word2Vec with subword information
- Handles out-of-vocabulary words via character n-grams
- Formula: `vec(word) = Σ vec(ngrams) / |ngrams|`

#### Sentence/Document Embeddings

**Sentence-BERT (SBERT)** (2019)
- Siamese BERT network for sentence embeddings
- Fine-tuned on NLI (Natural Language Inference) datasets
- Dimensionality: 768
- Formula: `sim(s1, s2) = cos(embed(s1), embed(s2))`
- **Models**:
  - `all-MiniLM-L6-v2`: Fast (384 dim)
  - `all-mpnet-base-v2`: Best accuracy (768 dim)

**Universal Sentence Encoder** (2018)
- Two variants: Transformer (high accuracy) / DAN (fast)
- Dimensionality: 512
- Trained on SNLI + MultiNLI

**Advantages**:
- Captures semantic meaning
- Handles synonyms and paraphrases
- Cross-lingual capabilities

**Disadvantages**:
- Requires model download (100MB - 500MB)
- Computationally expensive
- Domain-specific terminology may need fine-tuning

---

### 3. Graph Embeddings

#### Node2Vec (2016)

**Algorithm**:
1. Generate biased random walks through graph
2. Train Skip-gram model on walks
3. Learn node embeddings based on context

**Parameters**:
- `p`: Return parameter (control likelihood of revisiting nodes)
- `q`: In-out parameter (control BFS vs DFS exploration)
- `dimensions`: 64-128
- `walk_length`: 30-80
- `num_walks`: 10-100

**Formula**:
```
walk = biased_random_walk(graph, start_node, p, q, length)
embedding = skip_gram_train(walks)
```

**Similarity**: `sim(n1, n2) = cos(embed(n1), embed(n2))`

**Advantages**:
- Preserves graph structure
- Captures community structure
- Can be used for node classification, link prediction

#### GraphSAGE (2017)

**Algorithm**:
- Sample and aggregate neighbor information
- Learn inductive embeddings (generalize to unseen nodes)

**Aggregators**:
- Mean: Average neighbor embeddings
- Pool: Element-wise max/mean
- LSTM: Sequence-based aggregation

#### Deep Graph Infomax (DGI) (2019)

**Algorithm**:
- Maximize mutual information between global and local graph representations
- Unsupervised learning of graph embeddings

---

### 4. Structural Alignment

#### Graph Isomorphism

**Definition**: Two graphs G1, G2 are isomorphic if there exists bijection f: V1 → V2 preserving edges.

**VF2 Algorithm** (2004)
- State-of-the-art subgraph isomorphism
- Complexity: O(n! ) worst case, O(n^d) average case
- Uses feasibility rules to prune search space

**Weisfeiler-Lehman Test** (1968)
- Graph isomorphism test (not guaranteed)
- Iterative color refinement
- Complexity: O((n + m) log n)

#### Graph Edit Distance (GED)

**Definition**: Minimum edit operations (insert/delete nodes/edges) to transform G1 → G2.

**Formula**: `GED(G1, G2) = min ops to transform`

**Approximation**: A* search, beam search, Hungarian algorithm

---

### 5. Knowledge Graph Validation

**Purpose**: Use external knowledge graphs to validate mapping confidence.

**Key Resources**:
1. **ConceptNet**: 28M assertions, 304 languages
2. **WordNet**: 117k synsets, English
3. **Wikidata**: 100M+ entities, multilingual
4. **DBpedia**: 4M+ entities from Wikipedia

**Validation Process**:
1. Query KG for relationship between mapped concepts
2. Extract semantic relationship types (e.g., synonym, related_to)
3. Boost/reduce confidence based on KG evidence

---

## Knowledge Graph Resources

### ConceptNet

**Overview**: Large commonsense knowledge graph.

**Statistics** (2025):
- Assertions: 28M+
- Languages: 304
- Relationships: 34 types (e.g., `RelatedTo`, `IsA`, `PartOf`)

**API Usage**:
```python
import requests

# Query ConceptNet API
url = f"http://api.conceptnet.io/c/en/{concept}"
response = requests.get(url)
data = response.json()

# Extract related concepts
related = [edge['end']['label'] for edge in data['edges']]
```

**Relationship Types**:
- `RelatedTo`: General relatedness
- `IsA`: Taxonomic relationship
- `PartOf`: Meronymic relationship
- `HasProperty`: Attribute relationship
- `UsedFor`: Functional relationship
- `Causes`: Causal relationship

**Advantages**:
- Free and open
- Multilingual
- Commonsense reasoning
- Easy REST API

**Limitations**:
- Not domain-specific
- Some noise in crowdsourced data
- Limited reasoning capabilities

---

### WordNet

**Overview**: Lexical database of English.

**Statistics**:
- Synsets: 117,659
- Words: 207,032
- POS Tags: Noun, verb, adjective, adverb

**API Usage** (NLTK):
```python
from nltk.corpus import wordnet as wn

# Get synsets
synsets = wn.synsets('velocity')

# Path-based similarity
sim = wn.path_similarity(wn.synset('velocity.n.01'), wn.synset('speed.n.01'))

# Wu-Palmer similarity
sim_wup = wn.wup_similarity(wn.synset('velocity.n.01'), wn.synset('speed.n.01'))
```

**Similarity Metrics**:
1. **Path Similarity**: `1 / (path_length + 1)`
2. **Wu-Palmer Similarity**: `2 * depth(LCS) / (depth(s1) + depth(s2))`
3. **Lin Similarity**: Information content-based

**Advantages**:
- High-quality curated data
- Rich hierarchical structure
- Multiple similarity metrics

**Limitations**:
- English only
- Limited coverage of technical terms
- No commonsense relationships

---

### Wikidata

**Overview**: Free knowledge graph powered by Wikimedia.

**Statistics**:
- Items: 100M+
- Properties: 10K+
- Languages: 400+

**API Usage** (SPARQL):
```python
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
query = """
SELECT ?item ?itemLabel WHERE {
  ?item rdfs:label "velocity"@en.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
```

**Advantages**:
- Massive coverage
- Multilingual
- Rich metadata (identifiers, images)
- Active community

**Limitations**:
- SPARQL learning curve
- Rate-limited queries
- Some inconsistency

---

### DBpedia

**Overview**: Structured data from Wikipedia.

**Statistics**:
- Entities: 4M+
- Properties: 7K+
- Ontologies: 68+

**API Usage** (SPARQL):
```python
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?related ?label WHERE {
  <http://dbpedia.org/resource/Velocity> dbo:related ?related.
  ?related rdfs:label ?label.
  FILTER (lang(?label) = 'en')
}
"""
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
```

---

## Semantic Similarity Methods

### Hybrid Similarity Score

**Formula**:
```
sim_total = w1 * sim_lexical + w2 * sim_semantic + w3 * sim_graph + w4 * sim_kg

where:
w1 + w2 + w3 + w4 = 1
```

**Default Weights**:
- `w1` (lexical): 0.15
- `w2` (semantic): 0.40
- `w3` (graph): 0.30
- `w4` (KG validation): 0.15

---

### Confidence Scoring

**Levels**:
1. **High Confidence** (>0.8): Direct match or KG-validated synonym
2. **Medium Confidence** (0.5-0.8): Strong semantic similarity
3. **Low Confidence** (0.3-0.5): Partial match or weak evidence
4. **No Match** (<0.3): Insufficient evidence

**Aggregation**:
- Use average/maximum of all evidence
- Penalize inconsistencies
- Boost KG-validated mappings

---

## Graph Matching Algorithms

### 1. VF2 Algorithm

**Purpose**: Subgraph isomorphism detection.

**Complexity**:
- Worst case: O(n! )
- Average case: O(n^d) where d = small constant

**Implementation** (NetworkX):
```python
import networkx as nx
from networkx.algorithms.isomorphism import VF2GraphMatcher

G1 = nx.Graph()
G2 = nx.Graph()

# Build graphs...

GM = VF2GraphMatcher(G1, G2)
if GM.subgraph_is_morphic():
    mapping = GM.mapping
```

**Feasibility Rules**:
1. **Semantic rule**: Labels must match
2. **Structural rule**: Degree consistency
3. **Look-ahead**: Prune inconsistent partial mappings

---

### 2. Weisfeiler-Lehman Graph Isomorphism Test

**Purpose**: Test if graphs are isomorphic (not guaranteed).

**Algorithm**:
1. Initialize colors (e.g., based on degree)
2. Iteratively refine colors based on neighbor colors
3. Compare color histograms

**Complexity**: O((n + m) log n)

**Implementation** (graph neural networks):
```python
import torch
from torch_geometric.nn import WLConv

# 1D Weisfeiler-Lehman
conv = WLConv()
x = conv(x, edge_index)  # Refine colors
```

---

### 3. Graph Edit Distance

**Purpose**: Quantify dissimilarity between graphs.

**Exact Solution**: A* search (exponential)

**Approximation**: Beam search, Hungarian algorithm

**Formula**:
```
GED(G1, G2) = min {
  cost_insert_node * #inserted_nodes +
  cost_delete_node * #deleted_nodes +
  cost_insert_edge * #inserted_edges +
  cost_delete_edge * #deleted_edges +
  cost_substitute_node * #substituted_nodes +
  cost_substitute_edge * #substituted_edges
}
```

---

## Proposed Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Ontology Mapper (Ψ₂)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Domain A (source), Domain B (target)                 │
│  Output: Mapping + Confidence Score                          │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 1: Preprocessing                                 │ │
│  │  - Extract concepts and relations                       │ │
│  │  - Build graph representations                          │ │
│  │  - Normalize labels                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 2: Candidate Generation                          │ │
│  │  - Lexical similarity (Jaro-Winkler)                    │ │
│  │  - Initial candidate pairs                              │ │
│  │  - Filter by threshold (0.3)                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 3: Semantic Similarity                           │ │
│  │  - Sentence embeddings (SBERT)                          │ │
│  │  - Cosine similarity scoring                            │ │
│  │  - Threshold: 0.5                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 4: Graph Embedding                               │ │
│  │  - Node2Vec on domain graphs                            │ │
│  │  - Structural similarity                                │ │
│  │  - Threshold: 0.5                                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 5: Knowledge Graph Validation                    │ │
│  │  - Query ConceptNet/WordNet                             │ │
│  │  - Extract relationship types                           │ │
│  │  - Adjust confidence scores                             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Stage 6: Confidence Aggregation                        │ │
│  │  - Weighted sum of all evidence                         │ │
│  │  - Consistency checking                                 │ │
│  │  - Final mapping generation                             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                          ↓                                    │
│  Output: {                                                  │
│    mapping: {source_concept: target_concept, ...},         │
│    confidence: {pair: score, ...},                         │
│    metadata: {algorithm, timestamp, params}                │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### Component Design

#### 1. Preprocessing Module

**Responsibilities**:
- Extract concepts from domain constraints
- Build graph (nodes = concepts, edges = relations)
- Normalize labels (lowercase, remove punctuation)
- Handle synonyms and aliases

**Input**: Domain object (from I_mech)
**Output**: NetworkX graph + concept list

---

#### 2. Lexical Matcher

**Responsibilities**:
- Compute string similarity (Jaro-Winkler)
- Generate initial candidate pairs
- Filter by threshold

**Complexity**: O(n^2) where n = # concepts

**Optimization**: Use locality-sensitive hashing (LSH) for large concept sets

---

#### 3. Semantic Matcher

**Responsibilities**:
- Load sentence transformer model
- Compute embeddings for concepts
- Compute cosine similarity
- Filter by threshold

**Models**:
- Default: `all-MiniLM-L6-v2` (fast)
- Optional: `all-mpnet-base-v2` (accurate)

**Complexity**: O(n^2 * d) where d = embedding dimension

**Optimization**: Faiss index for approximate nearest neighbor

---

#### 4. Graph Embedder

**Responsibilities**:
- Run Node2Vec on domain graphs
- Compute structural similarity
- Align graph structures

**Parameters**:
- Dimensions: 64
- Walk length: 40
- Number of walks: 20
- p, q: 1.0, 1.0 (unbiased)

**Complexity**: O(n * walk_length * num_walks)

---

#### 5. Knowledge Graph Validator

**Responsibilities**:
- Query ConceptNet/WordNet APIs
- Extract relationship types
- Boost/reduce confidence based on evidence

**Caching**: Store KG responses locally to reduce API calls

**Complexity**: O(n) API calls (with caching)

---

#### 6. Confidence Aggregator

**Responsibilities**:
- Combine evidence from all sources
- Check consistency (e.g., transitive relationships)
- Generate final mapping

**Algorithm**:
```
For each candidate pair (c1, c2):
  score = w1*lexical + w2*semantic + w3*graph + w4*kg
  if score > threshold:
    mapping[c1] = c2
    confidence[c1] = score
```

**Consistency Checks**:
- Transitivity: If a→b and b→c, then a→c should have high score
- Injectivity: Each source maps to at most one target
- Symmetry: If a→b has high score, b→a should also

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)

**Deliverables**:
1. OntologyMapper class skeleton
2. Preprocessing module
3. Lexical matcher (Jaro-Winkler)
4. Basic confidence aggregation

**Dependencies**:
- `networkx` (graph operations)
- `numpy` (numerical operations)

**Complexity**: Low

---

### Phase 2: Semantic Similarity (Week 2)

**Deliverables**:
1. Sentence transformer integration
2. Semantic matcher module
3. Cosine similarity scoring
4. Model caching

**Dependencies**:
- `sentence-transformers`
- Model downloads (~100MB)

**Complexity**: Medium

---

### Phase 3: Graph Embeddings (Week 3)

**Deliverables**:
1. Node2Vec integration
2. Graph embedding module
3. Structural similarity scoring

**Dependencies**:
- `node2vec`
- `gensim` (Word2Vec trainer)

**Complexity**: Medium

---

### Phase 4: Knowledge Graph Integration (Week 4)

**Deliverables**:
1. ConceptNet API integration
2. WordNet integration
3. KG validation module
4. Response caching

**Dependencies**:
- `requests` (API calls)
- `nltk` (WordNet)
- `sqlite3` (local cache)

**Complexity**: Medium

---

### Phase 5: I_mech Integration (Week 5)

**Deliverables**:
1. Real-time mapping for Stage 2
2. Similarity scoring for I_mech
3. Integration tests

**Dependencies**:
- I_mech Stage 2 modules

**Complexity**: High

---

## Integration with I_mech

### Where Ontology Mapping Fits in I_mech

```
I_mech Pipeline:
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: FDG Extraction                                    │
│  - Extract functional dependency graphs from domains        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Isomorphism Detection (Ψ₂ Integration Point)     │
│  - Ontology mapping (THIS MODULE)                          │
│  - Graph isomorphism detection (VF2, WL)                   │
│  - Causality alignment                                      │
│  - Similarity scoring                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Solution Transfer                                 │
│  - Map solutions using ontology mappings                   │
│  - Validate transferred solutions                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Result Validation                                 │
│  - Validate isomorphic mapping                              │
│  - Generate confidence scores                               │
└─────────────────────────────────────────────────────────────┘
```

### API Interface

**Function**: `map_ontologies(source_domain: Domain, target_domain: Domain) -> MappingResult`

**Input**:
```python
source_domain: Domain {
  id: str,
  name: str,
  description: str,
  fdg: FunctionalDependencyGraph,
  ...
}

target_domain: Domain {
  ...
}
```

**Output**:
```python
MappingResult {
  concept_mapping: Dict[str, str],  # source -> target
  relation_mapping: Dict[str, str],  # source relation -> target relation
  confidence: Dict[str, float],      # pair -> score
  metadata: {
    algorithm: str,
    timestamp: str,
    parameters: Dict[str, Any],
    validation_scores: Dict[str, float]
  }
}
```

### Real-Time Mapping

**Requirement**: Stage 2 requires real-time ontology mapping for candidate domain pairs.

**Optimization**:
- Pre-compute embeddings for all domains
- Use Faiss index for fast approximate nearest neighbor
- Cache mappings between domain pairs
- Incremental updates for new domains

**Target Performance**:
- Latency: <10 seconds for domain pair mapping
- Throughput: 100+ domain pairs per minute

---

## Validation & Testing

### Unit Tests

**Test Coverage**:
1. Lexical similarity (Jaro-Winkler)
2. Semantic similarity (embeddings)
3. Graph embeddings (Node2Vec)
4. KG validation (ConceptNet, WordNet)
5. Confidence aggregation

**Test Cases**:
- Exact matches (score >0.9)
- Synonyms (score >0.7)
- Related concepts (score >0.5)
- Unrelated concepts (score <0.3)

**Example**:
```python
def test_lexical_similarity():
  mapper = OntologyMapper()
  score = mapper._lexical_similarity("velocity", "velocity_x")
  assert score > 0.8

def test_semantic_similarity():
  mapper = OntologyMapper()
  score = mapper._semantic_similarity("fast", "rapid")
  assert score > 0.7
```

---

### Integration Tests

**Test Scenarios**:
1. **Known Isomorphisms**:
   - Fluid dynamics ↔ Electricity
   - Mechanical springs ↔ Electrical circuits
   - Heat transfer ↔ Mass transfer

2. **Partial Isomorphisms**:
   - 2D kinematics ↔ 3D kinematics
   - Classical mechanics ↔ Relativistic mechanics

3. **Non-Isomorphic Domains**:
   - Fluid dynamics ↔ Quantum mechanics
   - Expected: Low confidence scores

---

### Performance Benchmarks

**Metrics**:
1. **Accuracy**: % of correct mappings (manually validated)
2. **Precision**: % of high-confidence mappings that are correct
3. **Recall**: % of all correct mappings found
4. **F1 Score**: Harmonic mean of precision and recall
5. **Latency**: Time to map domain pair
6. **Throughput**: Domain pairs per minute

**Target Performance**:
- Accuracy: >80%
- Precision: >85%
- Recall: >75%
- F1 Score: >80%
- Latency: <10 seconds
- Throughput: 100+ pairs/minute

---

### Validation on Real-World Domains

**Datasets**:
1. **Physics Domains**:
   - Mechanics, thermodynamics, electromagnetism, fluid dynamics
   - Known analogies: flow-electricity, heat-mass transfer

2. **Engineering Domains**:
   - Structural, thermal, electrical, chemical engineering
   - Cross-disciplinary transfer problems

3. **Synthetic Domains**:
   - Generated isomorphisms
   - Controlled complexity

**Validation Method**:
1. Manually create ground truth mappings
2. Run ontology mapper
3. Compare predicted vs. ground truth
4. Compute metrics (accuracy, precision, recall)

---

## Technical Specifications

### Dependencies

```python
# Core dependencies
networkx >= 3.0
numpy >= 1.24
scipy >= 1.10

# Semantic similarity
sentence-transformers >= 2.2
torch >= 2.0

# Graph embeddings
node2vec >= 0.4
gensim >= 4.3

# Knowledge graphs
requests >= 2.28
nltk >= 3.8
SPARQLWrapper >= 2.0

# Optimization (optional)
faiss-cpu >= 1.7  # or faiss-gpu for GPU acceleration
```

### File Structure

```
rese/phase2/
├── ontology_mapper.py           # Main class
├── ontology_components/          # Sub-modules
│   ├── lexical_matcher.py       # String similarity
│   ├── semantic_matcher.py      # Embeddings
│   ├── graph_embedder.py        # Node2Vec
│   ├── kg_validator.py          # KG integration
│   └── confidence.py            # Aggregation
├── ontology_cache/              # Cached data
│   ├── models/                  # Pre-trained models
│   ├── kg_cache.db              # KG responses
│   └── embeddings/              # Pre-computed embeddings
└── tests/
    ├── test_ontology_mapper.py
    ├── test_lexical_matcher.py
    ├── test_semantic_matcher.py
    ├── test_graph_embedder.py
    └── test_kg_validator.py
```

---

### Configuration

```python
# config.py

# Lexical matching
LEXICAL_THRESHOLD = 0.3
SIMILARITY_METHOD = 'jaro-winkler'  # or 'levenshtein', 'ngram'

# Semantic matching
SEMANTIC_MODEL = 'all-MiniLM-L6-v2'  # or 'all-mpnet-base-v2'
SEMANTIC_THRESHOLD = 0.5
EMBEDDING_DIM = 384  # for MiniLM-L6

# Graph embedding
GRAPH_EMBEDDING_DIM = 64
WALK_LENGTH = 40
NUM_WALKS = 20
P = 1.0  # Return parameter
Q = 1.0  # In-out parameter
GRAPH_THRESHOLD = 0.5

# Knowledge graph validation
KG_ENABLED = True
KG_CACHE_SIZE = 10000
KG_TIMEOUT = 5  # seconds

# Confidence aggregation
W_LEXICAL = 0.15
W_SEMANTIC = 0.40
W_GRAPH = 0.30
W_KG = 0.15
FINAL_THRESHOLD = 0.5

# Performance
USE_FAISS = True  # Use approximate nearest neighbor
FAISS_INDEX_TYPE = 'IVF'  # or 'HNSW'
BATCH_SIZE = 32
```

---

## References

### Academic Papers

1. **Ontology Alignment**
   - Euzenat, J., & Shvaiko, P. (2007). Ontology matching. Springer.
   - Choi, N., Song, I. Y., & Han, H. (2006). A survey on ontology mapping. SIGMOD Record.

2. **Word Embeddings**
   - Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. arXiv.
   - Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. EMNLP.
   - Bojanowski, P., et al. (2016). Enriching word vectors with subword information. TACL.

3. **Sentence Embeddings**
   - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. EMNLP.
   - Cer, D., et al. (2018). Universal sentence encoder. EMNLP.

4. **Graph Embeddings**
   - Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD.
   - Hamilton, W. L., Ying, R., & Leskovec, J. (2017). GraphSAGE. NeurIPS.
   - Velickovic, P., et al. (2019). Deep graph infomax. ICLR.

5. **Graph Matching**
   - Cordella, L. P., et al. (2004). A (sub)graph isomorphism algorithm for matching large graphs. IEEE TPAMI.
   - Weisfeiler, B., & Lehman, A. A. (1968). A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsia.

### Technical Resources

1. **ConceptNet**: https://conceptnet.io/
2. **WordNet**: https://wordnet.princeton.edu/
3. **Wikidata**: https://www.wikidata.org/
4. **DBpedia**: https://www.dbpedia.org/
5. **Sentence-Transformers**: https://www.sbert.net/
6. **Node2Vec**: https://github.com/aditya-grover/node2vec
7. **NetworkX**: https://networkx.org/

### Books

1. Euzenat, J., & Shvaiko, P. (2013). Ontology Matching. Springer.
2. Newman, M. (2018). Networks. Oxford University Press.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

---

## Next Steps

### Immediate Actions (Week 1)

1. ✅ **Complete Research Document** (this document)
2. ⏳ **Create OntologyMapper Skeleton**
3. ⏳ **Implement Lexical Matcher**
4. ⏳ **Write Unit Tests**

### Short-Term Goals (Weeks 2-4)

1. ⏳ **Implement Semantic Matcher**
2. ⏳ **Implement Graph Embedder**
3. ⏳ **Integrate Knowledge Graphs**
4. ⏳ **Integration Testing**

### Long-Term Goals (Weeks 5-8)

1. ⏳ **I_mech Stage 2 Integration**
2. ⏳ **Performance Optimization**
3. ⏳ **Real-World Validation**
4. ⏳ **Documentation**

---

## Appendix

### Example: Fluid Dynamics ↔ Electricity Mapping

**Concepts**:
- "flow rate" ↔ "current"
- "pressure" ↔ "voltage"
- "pipe resistance" ↔ "electrical resistance"
- "fluid inertia" ↔ "inductance"
- "fluid capacitance" ↔ "electrical capacitance"

**Similarity Scores**:
```
("flow rate", "current"):
  lexical: 0.0 (different words)
  semantic: 0.65 (related concepts)
  graph: 0.72 (similar structural roles)
  KG: 0.8 (ConceptNet: RelatedTo)
  final: 0.68

("pressure", "voltage"):
  lexical: 0.0
  semantic: 0.58
  graph: 0.70
  KG: 0.75 (WordNet: similar causality)
  final: 0.62
```

**Structural Alignment**:
```
Fluid Dynamics Graph:
flow rate --caused_by--> pressure --opposes--> resistance

Electricity Graph:
current --caused_by--> voltage --opposes--> resistance

Isomorphism: Perfect match
```

---

**Status**: 🔄 Research Complete
**Next**: Implement OntologyMapper

---

*Agent: G2 (Ψ₂ Specialist)*
*Date: 2025-12-31*
*Version: 1.0*
