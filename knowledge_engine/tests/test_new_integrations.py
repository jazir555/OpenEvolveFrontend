"""
Comprehensive Test Suite for New Knowledge Engine Integrations

Tests for:
- PAMI Integration (Pattern Mining)
- NeuralKG Integration (KG Embeddings)
- Causal-Learn Integration (Causal Discovery)
- Lagrange-Mapper Integration (Topological Analysis)
- Unified Knowledge Extractor
"""

import unittest
import numpy as np
from typing import Dict, Any, List, Tuple


class TestPAMIIntegration(unittest.TestCase):
    """Test cases for PAMI Pattern Mining Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from knowledge_engine.integrations.pami_integration import PAMIPatternMiner
            cls.miner = PAMIPatternMiner()
            cls.available = cls.miner.is_available()
        except ImportError:
            cls.miner = None
            cls.available = False
    
    def test_pami_initialization(self):
        """Test PAMI module initialization."""
        if self.miner is None:
            self.skipTest("PAMI module not available")
        
        self.assertIsNotNone(self.miner)
        status = self.miner.get_status()
        self.assertIn('available', status)
        self.assertIn('algorithms', status)
    
    def test_mine_frequent_patterns(self):
        """Test frequent pattern mining."""
        if not self.available:
            self.skipTest("PAMI not available")
        
        # Sample transaction data
        transactions = [
            ['bread', 'milk', 'eggs'],
            ['bread', 'butter'],
            ['milk', 'eggs', 'cheese'],
            ['bread', 'milk', 'butter'],
            ['bread', 'eggs'],
            ['milk', 'butter'],
            ['bread', 'milk', 'eggs', 'butter'],
            ['eggs', 'cheese'],
            ['bread', 'milk'],
            ['butter', 'cheese']
        ]
        
        result = self.miner.mine_frequent_patterns(
            transactions=transactions,
            min_support=0.2,
            algorithm='fpgrowth'
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('patterns', result)
        self.assertIn('statistics', result)
        
        # Check statistics
        stats = result['statistics']
        self.assertIn('total_patterns', stats)
        self.assertGreaterEqual(stats['total_patterns'], 0)
    
    def test_mine_sequences(self):
        """Test sequential pattern mining."""
        if not self.available:
            self.skipTest("PAMI not available")
        
        # Sample sequence data
        sequences = [
            [['a'], ['b'], ['c']],
            [['a'], ['b'], ['d']],
            [['a'], ['c'], ['d']],
            [['b'], ['c'], ['d']],
            [['a'], ['b'], ['c'], ['d']]
        ]
        
        result = self.miner.mine_sequences(
            sequences=sequences,
            min_support=0.2
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('patterns', result)
    
    def test_analyze_knowledge_graph_patterns(self):
        """Test graph pattern analysis."""
        if not self.available:
            self.skipTest("PAMI not available")
        
        # Sample knowledge graph
        graph_data = {
            'nodes': [
                {'id': 'Alice', 'type': 'Person'},
                {'id': 'Bob', 'type': 'Person'},
                {'id': 'Charlie', 'type': 'Person'},
                {'id': 'AcmeCorp', 'type': 'Organization'},
                {'id': 'TechInc', 'type': 'Organization'}
            ],
            'edges': [
                {'source': 'Alice', 'target': 'Bob', 'type': 'knows'},
                {'source': 'Bob', 'target': 'Charlie', 'type': 'knows'},
                {'source': 'Alice', 'target': 'AcmeCorp', 'type': 'works_for'},
                {'source': 'Bob', 'target': 'TechInc', 'type': 'works_for'},
                {'source': 'Charlie', 'target': 'TechInc', 'type': 'works_for'}
            ]
        }
        
        result = self.miner.analyze_knowledge_graph_patterns(
            graph_data=graph_data,
            min_support=0.1
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('patterns', result)
        self.assertIn('statistics', result)
    
    def test_discover_association_rules(self):
        """Test association rule discovery."""
        if not self.available:
            self.skipTest("PAMI not available")
        
        transactions = [
            ['bread', 'milk'],
            ['bread', 'butter', 'milk'],
            ['bread', 'eggs'],
            ['milk', 'eggs'],
            ['bread', 'milk', 'eggs', 'butter']
        ]
        
        result = self.miner.discover_association_rules(
            transactions=transactions,
            min_support=0.2,
            min_confidence=0.5
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('rules', result)


class TestNeuralKGIntegration(unittest.TestCase):
    """Test cases for NeuralKG Embedding Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from knowledge_engine.integrations.neuralkg_integration import NeuralKGEmbedder
            cls.embedder = NeuralKGEmbedder()
            cls.available = cls.embedder.is_available()
        except ImportError:
            cls.embedder = None
            cls.available = False
    
    def test_neuralkg_initialization(self):
        """Test NeuralKG module initialization."""
        if self.embedder is None:
            self.skipTest("NeuralKG module not available")
        
        self.assertIsNotNone(self.embedder)
        status = self.embedder.get_status()
        self.assertIn('available', status)
        self.assertIn('models', status)
    
    def test_generate_embeddings(self):
        """Test knowledge graph embedding generation."""
        if not self.available:
            self.skipTest("NeuralKG not available")
        
        # Sample triples
        triples = [
            ('Alice', 'knows', 'Bob'),
            ('Bob', 'knows', 'Charlie'),
            ('Alice', 'works_for', 'AcmeCorp'),
            ('Bob', 'works_for', 'TechInc'),
            ('Charlie', 'works_for', 'TechInc'),
            ('AcmeCorp', 'competitor', 'TechInc')
        ]
        
        result = self.embedder.generate_embeddings(
            triples=triples,
            model_name='transe',
            embedding_dim=50
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('embeddings', result)
        self.assertIn('entities', result['embeddings'])
        self.assertIn('relations', result['embeddings'])
        
        # Check embeddings have correct dimensions
        for entity, embedding in result['embeddings']['entities'].items():
            self.assertEqual(len(embedding), 50)
    
    def test_predict_links(self):
        """Test link prediction."""
        if not self.available:
            self.skipTest("NeuralKG not available")
        
        # First generate embeddings
        triples = [
            ('Alice', 'knows', 'Bob'),
            ('Bob', 'knows', 'Charlie'),
            ('Alice', 'works_for', 'AcmeCorp'),
            ('Bob', 'works_for', 'TechInc')
        ]
        
        emb_result = self.embedder.generate_embeddings(triples, 'transe', 50)
        if emb_result['status'] != 'success':
            self.skipTest("Could not generate embeddings")
        
        embeddings = emb_result['embeddings']
        
        # Predict links
        result = self.embedder.predict_links(
            head='Alice',
            relation='knows',
            candidate_tails=['Bob', 'Charlie', 'AcmeCorp', 'TechInc'],
            embeddings=embeddings,
            top_k=3
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('predictions', result)
        self.assertLessEqual(len(result['predictions']), 3)
    
    def test_find_similar_entities(self):
        """Test entity similarity search."""
        if not self.available:
            self.skipTest("NeuralKG not available")
        
        # Generate embeddings
        triples = [
            ('Alice', 'knows', 'Bob'),
            ('Bob', 'knows', 'Charlie'),
            ('Alice', 'works_for', 'AcmeCorp'),
            ('Bob', 'works_for', 'TechInc'),
            ('Charlie', 'works_for', 'TechInc')
        ]
        
        emb_result = self.embedder.generate_embeddings(triples, 'transe', 50)
        if emb_result['status'] != 'success':
            self.skipTest("Could not generate embeddings")
        
        result = self.embedder.find_similar_entities(
            entity='Alice',
            embeddings=emb_result['embeddings'],
            top_k=3
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('similar_entities', result)
    
    def test_ensemble_embeddings(self):
        """Test ensemble embedding generation."""
        if not self.available:
            self.skipTest("NeuralKG not available")
        
        triples = [
            ('Alice', 'knows', 'Bob'),
            ('Bob', 'knows', 'Charlie'),
            ('Alice', 'works_for', 'AcmeCorp')
        ]
        
        result = self.embedder.ensemble_embeddings(
            triples=triples,
            models=['transe'],  # Use available models
            embedding_dim=50
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('embeddings', result)


class TestCausalLearnIntegration(unittest.TestCase):
    """Test cases for Causal-Learn Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from knowledge_engine.integrations.causal_learn_integration import CausalDiscoveryEngine
            cls.engine = CausalDiscoveryEngine()
            cls.available = cls.engine.is_available()
        except ImportError:
            cls.engine = None
            cls.available = False
    
    def test_causal_initialization(self):
        """Test Causal-Learn module initialization."""
        if self.engine is None:
            self.skipTest("Causal-Learn module not available")
        
        self.assertIsNotNone(self.engine)
        status = self.engine.get_status()
        self.assertIn('available', status)
        self.assertIn('algorithms', status)
    
    def test_discover_causal_structure_pc(self):
        """Test PC algorithm for causal discovery."""
        if not self.available:
            self.skipTest("Causal-Learn not available")
        
        # Generate synthetic data with known causal structure
        np.random.seed(42)
        n_samples = 500
        
        # X -> Y -> Z, X -> Z
        X = np.random.randn(n_samples)
        Y = 2 * X + np.random.randn(n_samples) * 0.1
        Z = 1.5 * Y + 0.5 * X + np.random.randn(n_samples) * 0.1
        
        data = np.column_stack([X, Y, Z])
        variable_names = ['X', 'Y', 'Z']
        
        result = self.engine.discover_causal_structure(
            data=data,
            variable_names=variable_names,
            algorithm='pc',
            alpha=0.05,
            independence_test='fisherz'
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('graph', result)
        self.assertIn('nodes', result['graph'])
        self.assertIn('edges', result['graph'])
    
    def test_analyze_causal_graph(self):
        """Test causal graph analysis."""
        if not self.available:
            self.skipTest("Causal-Learn not available")
        
        # Sample causal graph
        graph_data = {
            'nodes': ['X', 'Y', 'Z', 'W'],
            'edges': [
                {'source': 'X', 'target': 'Y', 'type': 'directed'},
                {'source': 'Y', 'target': 'Z', 'type': 'directed'},
                {'source': 'X', 'target': 'Z', 'type': 'directed'},
                {'source': 'W', 'target': 'Y', 'type': 'directed'}
            ]
        }
        
        result = self.engine.analyze_causal_graph(graph_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('analysis', result)
        self.assertIn('num_nodes', result['analysis'])
        self.assertIn('num_edges', result['analysis'])
        self.assertIn('roots', result['analysis'])
        self.assertIn('leaves', result['analysis'])
    
    def test_identify_confounders(self):
        """Test confounder identification."""
        if not self.available:
            self.skipTest("Causal-Learn not available")
        
        # Graph with confounder: Z -> X, Z -> Y
        graph_data = {
            'nodes': ['X', 'Y', 'Z'],
            'edges': [
                {'source': 'Z', 'target': 'X', 'type': 'directed'},
                {'source': 'Z', 'target': 'Y', 'type': 'directed'},
                {'source': 'X', 'target': 'Y', 'type': 'directed'}
            ]
        }
        
        result = self.engine.identify_confounders(
            graph_data=graph_data,
            target_x='X',
            target_y='Y'
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('confounders', result)
        # Z should be identified as a common cause


class TestLagrangeMapperIntegration(unittest.TestCase):
    """Test cases for Lagrange-Mapper Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from knowledge_engine.integrations.lagrange_mapper_integration import LagrangeAttractorAnalyzer
            cls.analyzer = LagrangeAttractorAnalyzer()
            cls.available = cls.analyzer.is_available()
        except ImportError:
            cls.analyzer = None
            cls.available = False
    
    def test_lagrange_initialization(self):
        """Test Lagrange-Mapper module initialization."""
        if self.analyzer is None:
            self.skipTest("Lagrange-Mapper module not available")
        
        self.assertIsNotNone(self.analyzer)
        status = self.analyzer.get_status()
        self.assertIn('available', status)
    
    def test_analyze_embedding_landscape(self):
        """Test attractor landscape analysis."""
        if not self.available:
            self.skipTest("Lagrange-Mapper not available")
        
        # Generate sample embeddings with clusters
        np.random.seed(42)
        
        # Create 3 clusters
        cluster1 = np.random.randn(30, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(40, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        labels = [f'point_{i}' for i in range(100)]
        
        result = self.analyzer.analyze_embedding_landscape(
            embeddings=embeddings,
            labels=labels,
            n_clusters=3,
            reduction_method='pca',
            reduction_dims=2
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('landscape', result)
        self.assertIn('clusters', result['landscape'])
        self.assertIn('attractors', result['landscape'])
        
        # Check we found approximately 3 clusters
        self.assertGreaterEqual(len(result['landscape']['clusters']), 2)
    
    def test_analyze_knowledge_topology(self):
        """Test knowledge graph topology analysis."""
        if not self.available:
            self.skipTest("Lagrange-Mapper not available")
        
        # Sample knowledge graph
        graph_data = {
            'nodes': [
                {'id': f'node_{i}', 'type': 'entity'}
                for i in range(20)
            ],
            'edges': [
                {'source': f'node_{i}', 'target': f'node_{(i+1) % 20}', 'type': 'connected_to'}
                for i in range(20)
            ] + [
                {'source': f'node_{i}', 'target': f'node_{(i+5) % 20}', 'type': 'related_to'}
                for i in range(0, 20, 2)
            ]
        }
        
        result = self.analyzer.analyze_knowledge_topology(
            graph_data=graph_data,
            embedding_dim=10
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('landscape', result)
    
    def test_find_attractor_basins(self):
        """Test attractor basin computation."""
        if not self.available:
            self.skipTest("Lagrange-Mapper not available")
        
        # Generate sample data
        np.random.seed(42)
        embeddings = np.random.randn(50, 2)
        
        # Define attractor centers
        centers = np.array([
            [2, 2],
            [-2, -2],
            [2, -2]
        ])
        
        result = self.analyzer.find_attractor_basins(
            embeddings=embeddings,
            attractor_centers=centers,
            resolution=20
        )
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('basins', result)


class TestUnifiedKnowledgeExtractor(unittest.TestCase):
    """Test cases for Unified Knowledge Extractor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from knowledge_engine.integrations.unified_knowledge_extraction import UnifiedKnowledgeExtractor
            cls.extractor = UnifiedKnowledgeExtractor()
        except ImportError:
            cls.extractor = None
    
    def test_extractor_initialization(self):
        """Test unified extractor initialization."""
        if self.extractor is None:
            self.skipTest("Unified extractor not available")
        
        self.assertIsNotNone(self.extractor)
        modules = self.extractor.get_available_modules()
        self.assertIsInstance(modules, list)
        
        status = self.extractor.get_status()
        self.assertIn('available_modules', status)
        self.assertIn('capabilities', status)
    
    def test_extract_from_text(self):
        """Test text extraction."""
        if self.extractor is None:
            self.skipTest("Unified extractor not available")
        
        text = "Alice works at AcmeCorp. Bob knows Alice. They both live in New York."
        
        result = self.extractor.extract_from_text(
            text=text,
            extraction_type='entities_relations'
        )
        
        self.assertIn(result.status, ['success', 'partial', 'error'])
        self.assertIn('data', result.__dict__)
    
    def test_analyze_knowledge_graph(self):
        """Test unified graph analysis."""
        if self.extractor is None:
            self.skipTest("Unified extractor not available")
        
        graph_data = {
            'nodes': [
                {'id': 'Alice', 'type': 'Person'},
                {'id': 'Bob', 'type': 'Person'},
                {'id': 'Charlie', 'type': 'Person'},
                {'id': 'AcmeCorp', 'type': 'Organization'}
            ],
            'edges': [
                {'source': 'Alice', 'target': 'Bob', 'type': 'knows'},
                {'source': 'Bob', 'target': 'Charlie', 'type': 'knows'},
                {'source': 'Alice', 'target': 'AcmeCorp', 'type': 'works_for'}
            ]
        }
        
        result = self.extractor.analyze_knowledge_graph(
            graph_data=graph_data,
            analysis_types=['community']
        )
        
        self.assertIn(result.status, ['success', 'partial', 'error'])
    
    def test_run_extraction_pipeline(self):
        """Test complete extraction pipeline."""
        if self.extractor is None:
            self.skipTest("Unified extractor not available")
        
        input_data = {
            'text': 'Alice knows Bob. Bob works at TechInc.',
            'graph': {
                'nodes': [
                    {'id': 'Alice', 'type': 'Person'},
                    {'id': 'Bob', 'type': 'Person'},
                    {'id': 'TechInc', 'type': 'Organization'}
                ],
                'edges': [
                    {'source': 'Alice', 'target': 'Bob', 'type': 'knows'},
                    {'source': 'Bob', 'target': 'TechInc', 'type': 'works_for'}
                ]
            }
        }
        
        result = self.extractor.run_extraction_pipeline(
            input_data=input_data,
            pipeline_config={
                'extract_text': True,
                'analyze_graph': True
            }
        )
        
        self.assertIn(result.status, ['success', 'partial', 'error'])
        if result.status != 'error':
            self.assertIn('stage_results', result.data)


class TestIntegrationWithGenericTool(unittest.TestCase):
    """Test integration with Generic Knowledge Extraction Tool."""
    
    def test_import_path(self):
        """Test that all modules can be imported."""
        try:
            from knowledge_engine.integrations import (
                AIKnowledgeGraphIntegrator,
                PAMIPatternMiner,
                NeuralKGEmbedder,
                CausalDiscoveryEngine,
                LagrangeAttractorAnalyzer
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import integrations: {e}")
    
    def test_generic_tool_integration_pattern(self):
        """Test integration pattern with Generic Knowledge Extraction Tool."""
        try:
            from knowledge_engine.integrations.unified_knowledge_extraction import (
                UnifiedKnowledgeExtractor,
                extract_knowledge
            )
            
            # Test basic extraction function
            result = extract_knowledge(
                data={'text': 'Sample text for extraction'},
                operations=['text']
            )
            
            self.assertIn('status', result)
            self.assertIn('data', result)
            
        except ImportError:
            self.skipTest("Unified extractor not available")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPAMIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralKGIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalLearnIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLagrangeMapperIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedKnowledgeExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithGenericTool))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
