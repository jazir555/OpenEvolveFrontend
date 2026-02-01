"""
Natural Language Interface Node for BubbleLabs Integration

Provides a conversational interface for querying knowledge using natural language:
- Parse natural language questions into structured queries
- Execute queries against the knowledge graph
- Format results as natural language responses
- Support context and conversation history
- Handle ambiguous queries with clarifications
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import json
from collections import defaultdict

from .base_node import BubbleLabsNode, NodeExecutionError


class NaturalLanguageInterfaceNode(BubbleLabsNode):
    """
    Conversational interface for querying knowledge using natural language.

    Supports four operations:
    - query: Execute a natural language query against the knowledge graph
    - converse: Continue an ongoing conversation with context
    - clarify: Handle ambiguous queries by asking for clarification
    - suggest: Provide query suggestions based on context

    Features:
    - Natural language parsing with fallback pattern matching
    - Context-aware responses using conversation history
    - Structured result formatting with natural language generation
    - Source attribution and confidence scoring
    - Multi-language support framework
    """

    # Node metadata
    DISPLAY_NAME = "Natural Language Interface"
    DESCRIPTION = "Conversational interface for querying knowledge using natural language"
    ICON = "natural-language"
    CATEGORY = "interface"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        UnifiedKGIntegrationHub = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub.UnifiedKGIntegrationHub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available for NaturalLanguageInterfaceNode"
        )
        self.UnifiedKGIntegrationHub = UnifiedKGIntegrationHub

        # Safe import for NaturalLanguageQueryParser
        NaturalLanguageQueryParser = self.safe_import(
            'knowledge_engine.query.nl_parser.NaturalLanguageQueryParser',
            fallback_value=None,
            error_msg="NaturalLanguageQueryParser not available, will use fallback parsing"
        )
        self.NaturalLanguageQueryParser = NaturalLanguageQueryParser

        # Initialize instances
        self.hub = None
        self.nl_parser = None
        self._conversation_history = {}  # conversation_id -> list of exchanges

        # Initialize hub if available
        if UnifiedKGIntegrationHub:
            try:
                self.hub = UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized for NaturalLanguageInterfaceNode")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Initialize NL parser if available
        if NaturalLanguageQueryParser:
            try:
                self.nl_parser = NaturalLanguageQueryParser()
                self.logger.info("NaturalLanguageQueryParser initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize NaturalLanguageQueryParser: {e}")
                self.nl_parser = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields:
        - query: string - natural language query (for query/converse operations)
        - conversation_id: string - for context tracking (recommended)

        Optional:
        - operation: enum ["query", "converse", "clarify", "suggest"]
        - response_format: enum ["concise", "detailed", "structured"]
        - include_sources: boolean
        - max_results: integer
        - language: string
        - context_window: integer
        """
        errors = []

        # Get operation type
        operation = inputs.get('operation', self.config.get('operation', 'query'))

        valid_operations = ['query', 'converse', 'clarify', 'suggest']
        if operation not in valid_operations:
            errors.append(f"Invalid operation: {operation}. Must be one of: {', '.join(valid_operations)}")

        # Validate query for query and converse operations
        if operation in ['query', 'converse']:
            query = inputs.get('query') or self.config.get('query', '')
            if not query or not isinstance(query, str):
                errors.append(f"Operation '{operation}' requires a 'query' string")
            elif len(query.strip()) == 0:
                errors.append("Query cannot be empty or whitespace only")
            elif len(query) > 10000:
                errors.append("Query exceeds maximum length of 10,000 characters")

        # Validate numeric parameters
        if 'max_results' in inputs:
            try:
                max_results = int(inputs['max_results'])
                if max_results < 1 or max_results > 100:
                    errors.append("max_results must be between 1 and 100")
            except (ValueError, TypeError):
                errors.append("max_results must be an integer")

        if 'context_window' in inputs:
            try:
                context_window = int(inputs['context_window'])
                if context_window < 0 or context_window > 20:
                    errors.append("context_window must be between 0 and 20")
            except (ValueError, TypeError):
                errors.append("context_window must be an integer")

        # Validate response_format if provided
        if 'response_format' in inputs:
            valid_formats = ['concise', 'detailed', 'structured']
            if inputs['response_format'] not in valid_formats:
                errors.append(f"Invalid response_format. Must be one of: {', '.join(valid_formats)}")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the natural language interface operation.

        Args:
            inputs: Input data containing query and operation parameters
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing:
                - response: Natural language response string
                - structured_results: List of structured query results
                - confidence: Overall confidence score (0.0-1.0)
                - suggestions: List of follow-up query suggestions
                - conversation_id: The conversation identifier
                - metadata: Execution metadata

        Raises:
            NodeExecutionError: If execution fails
        """
        # Get parameters
        operation = inputs.get('operation', self.config.get('operation', 'query'))
        query = inputs.get('query', self.config.get('query', ''))
        conversation_id = inputs.get('conversation_id', self.config.get('conversation_id', ''))
        response_format = inputs.get('response_format', self.config.get('response_format', 'detailed'))
        include_sources = inputs.get('include_sources', self.config.get('include_sources', True))
        max_results = inputs.get('max_results', self.config.get('max_results', 5))
        language = inputs.get('language', self.config.get('language', 'en'))
        context_window = inputs.get('context_window', self.config.get('context_window', 3))

        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"

        context.update_progress(10, f"Initializing {operation} operation")
        self.logger.info(f"Natural language {operation}: {query[:100]}...")

        try:
            # Retrieve conversation history
            history = self._get_conversation_history(conversation_id, context_window)

            context.update_progress(30, "Processing natural language input")

            # Execute the appropriate operation
            if operation == 'query':
                result = self._execute_query(
                    query=query,
                    history=history,
                    response_format=response_format,
                    include_sources=include_sources,
                    max_results=max_results,
                    language=language,
                    context=context
                )
            elif operation == 'converse':
                result = self._execute_converse(
                    query=query,
                    history=history,
                    response_format=response_format,
                    include_sources=include_sources,
                    max_results=max_results,
                    language=language,
                    context=context
                )
            elif operation == 'clarify':
                result = self._execute_clarify(
                    query=query,
                    history=history,
                    language=language,
                    context=context
                )
            elif operation == 'suggest':
                result = self._execute_suggest(
                    query=query,
                    history=history,
                    max_results=max_results,
                    context=context
                )
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': ['query', 'converse', 'clarify', 'suggest']}
                )

            context.update_progress(90, "Formatting response")

            # Add conversation tracking
            result['conversation_id'] = conversation_id

            # Store exchange in history
            self._store_exchange(
                conversation_id=conversation_id,
                query=query,
                response=result.get('response', ''),
                context=context
            )

            # Add metadata
            result['metadata'] = {
                'operation': operation,
                'executed_at': datetime.now().isoformat(),
                'execution_id': self.execution_id,
                'language': language,
                'response_format': response_format,
                'conversation_length': len(history) + 1,
                'hub_available': self.hub is not None,
                'nl_parser_available': self.nl_parser is not None
            }

            context.update_progress(100, f"{operation} operation completed")
            self.logger.info(f"Natural language {operation} completed successfully")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Natural language interface failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Natural language interface failed: {str(e)}",
                details={
                    'operation': operation,
                    'query': query[:200],
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_query(
        self,
        query: str,
        history: List[Dict],
        response_format: str,
        include_sources: bool,
        max_results: int,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Execute a natural language query."""
        context.update_progress(40, "Parsing natural language query")

        # Parse the query
        parsed_query = self._parse_query(query, history)

        context.update_progress(60, "Executing against knowledge graph")

        # Execute the structured query
        results = self._execute_structured_query(parsed_query, max_results, context)

        context.update_progress(80, "Generating natural language response")

        # Generate response
        response = self._generate_response(
            query=query,
            parsed_query=parsed_query,
            results=results,
            response_format=response_format,
            include_sources=include_sources,
            language=language
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(query, results, parsed_query)

        # Calculate confidence
        confidence = self._calculate_confidence(parsed_query, results)

        return {
            'response': response,
            'structured_results': results,
            'confidence': confidence,
            'suggestions': suggestions,
            'parsed_query': parsed_query,
            'query_type': parsed_query.get('query_type', 'unknown')
        }

    def _execute_converse(
        self,
        query: str,
        history: List[Dict],
        response_format: str,
        include_sources: bool,
        max_results: int,
        language: str,
        context
    ) -> Dict[str, Any]:
        """Execute a conversational query with context awareness."""
        context.update_progress(40, "Processing conversation context")

        # Enhance query with context from history
        enhanced_query = self._enhance_query_with_context(query, history)

        context.update_progress(50, "Parsing enhanced query")

        # Parse the enhanced query
        parsed_query = self._parse_query(enhanced_query, history)

        # Add context awareness flags
        parsed_query['is_followup'] = len(history) > 0
        parsed_query['original_query'] = query

        context.update_progress(60, "Executing against knowledge graph")

        # Execute the query
        results = self._execute_structured_query(parsed_query, max_results, context)

        context.update_progress(80, "Generating contextual response")

        # Generate context-aware response
        response = self._generate_response(
            query=enhanced_query,
            parsed_query=parsed_query,
            results=results,
            response_format=response_format,
            include_sources=include_sources,
            language=language,
            history=history
        )

        # Generate contextual suggestions
        suggestions = self._generate_suggestions(enhanced_query, results, parsed_query, history)

        confidence = self._calculate_confidence(parsed_query, results)

        return {
            'response': response,
            'structured_results': results,
            'confidence': confidence,
            'suggestions': suggestions,
            'parsed_query': parsed_query,
            'query_type': parsed_query.get('query_type', 'unknown'),
            'context_used': len(history) > 0
        }

    def _execute_clarify(
        self,
        query: str,
        history: List[Dict],
        language: str,
        context
    ) -> Dict[str, Any]:
        """Handle ambiguous queries by generating clarification questions."""
        context.update_progress(40, "Analyzing query ambiguity")

        # Parse query to identify ambiguity
        parsed_query = self._parse_query(query, history)

        # Identify ambiguous aspects
        ambiguities = self._identify_ambiguities(parsed_query, query)

        context.update_progress(70, "Generating clarification options")

        # Generate clarification response
        if ambiguities:
            clarification = self._generate_clarification_response(ambiguities, language)
            response = clarification['message']
            options = clarification['options']
            is_ambiguous = True
        else:
            response = "Your query appears to be clear. Let me search for that information."
            options = []
            is_ambiguous = False

        return {
            'response': response,
            'structured_results': [],
            'confidence': 0.5 if is_ambiguous else 0.9,
            'suggestions': options,
            'parsed_query': parsed_query,
            'is_ambiguous': is_ambiguous,
            'ambiguities': ambiguities,
            'clarification_options': options
        }

    def _execute_suggest(
        self,
        query: str,
        history: List[Dict],
        max_results: int,
        context
    ) -> Dict[str, Any]:
        """Generate query suggestions based on context."""
        context.update_progress(40, "Analyzing query patterns")

        # Get suggestions based on query and history
        suggestions = self._generate_suggestions(query, [], {}, history, max_results * 2)

        context.update_progress(80, "Ranking suggestions")

        # Format response
        if suggestions:
            response = "Here are some suggested queries you might find helpful:\n\n"
            for i, suggestion in enumerate(suggestions[:max_results], 1):
                response += f"{i}. {suggestion}\n"
        else:
            response = "Try asking about specific entities, relationships, or topics in the knowledge graph."

        return {
            'response': response,
            'structured_results': [{'suggestion': s} for s in suggestions],
            'confidence': 0.8,
            'suggestions': suggestions,
            'query_type': 'suggestion'
        }

    def _parse_query(self, query: str, history: List[Dict]) -> Dict[str, Any]:
        """
        Parse natural language query into structured representation.

        Uses NaturalLanguageQueryParser if available, otherwise uses fallback parsing.
        """
        if self.nl_parser and hasattr(self.nl_parser, 'parse'):
            try:
                parsed = self.nl_parser.parse(query, context=history)
                if parsed:
                    return {
                        'query_type': parsed.get('intent', 'unknown'),
                        'entities': parsed.get('entities', []),
                        'relations': parsed.get('relations', []),
                        'filters': parsed.get('filters', {}),
                        'original_query': query,
                        'parser_used': 'nl_parser'
                    }
            except Exception as e:
                self.logger.warning(f"NL parser failed: {e}, using fallback")

        # Fallback pattern-based parsing
        return self._fallback_parse_query(query)

    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """
        Fallback query parsing using pattern matching.

        Handles common query patterns:
        - "Who works at X?" -> find employees of X
        - "What is X?" -> describe X
        - "Where is X?" -> location of X
        - "When did X happen?" -> temporal query
        - "List all X" -> enumerate X
        - "How many X?" -> count query
        """
        query_lower = query.lower().strip()

        # Pattern matching for query types
        patterns = {
            'entity_description': [
                r'^what is (.+?)\??$',
                r'^who is (.+?)\??$',
                r'^tell me about (.+?)\??$',
                r'^describe (.+?)\??$'
            ],
            'relationship_query': [
                r'^who works (?:at|for) (.+?)\??$',
                r'^who (?:created|developed|founded) (.+?)\??$',
                r'^what (?:does|did) (.+?) (?:create|develop|make)\??$',
                r'^where (?:does|is) (.+?) (?:work|located)\??$'
            ],
            'list_query': [
                r'^list all (.+)$',
                r'^show me (.+)$',
                r'^what are (?:all )?(.+)$',
                r'^find (.+)$'
            ],
            'count_query': [
                r'^how many (.+?)\??$',
                r'^count (.+)$',
                r'^number of (.+)$'
            ],
            'comparison_query': [
                r'^(?:compare|difference between) (.+?) and (.+)$',
                r'^how (?:is|does) (.+?) compare to (.+)$'
            ],
            'path_query': [
                r'^how (?:is|are) (.+?) related to (.+)$',
                r'^connection between (.+?) and (.+)$',
                r'^path from (.+?) to (.+)$'
            ]
        }

        # Match query against patterns
        for query_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.match(pattern, query_lower)
                if match:
                    return {
                        'query_type': query_type,
                        'entities': list(match.groups()),
                        'relations': [],
                        'filters': {},
                        'original_query': query,
                        'parser_used': 'fallback_pattern',
                        'matched_pattern': pattern
                    }

        # Default: unknown query type
        # Try to extract potential entities (capitalized phrases)
        entity_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
        potential_entities = re.findall(entity_pattern, query)

        return {
            'query_type': 'unknown',
            'entities': potential_entities,
            'relations': [],
            'filters': {},
            'original_query': query,
            'parser_used': 'fallback_pattern',
            'matched_pattern': None
        }

    def _execute_structured_query(
        self,
        parsed_query: Dict,
        max_results: int,
        context
    ) -> List[Dict[str, Any]]:
        """
        Execute the structured query against the knowledge graph.
        """
        query_type = parsed_query.get('query_type', 'unknown')
        entities = parsed_query.get('entities', [])

        results = []

        # Use hub if available
        if self.hub:
            try:
                if query_type == 'entity_description' and entities:
                    results = self._query_entity_description(entities[0])
                elif query_type == 'relationship_query' and len(entities) >= 1:
                    results = self._query_relationships(entities, parsed_query)
                elif query_type == 'path_query' and len(entities) >= 2:
                    results = self._query_paths(entities[0], entities[1])
                elif query_type == 'list_query' and entities:
                    results = self._query_list(entities[0], max_results)
                elif query_type == 'count_query':
                    results = self._query_count(entities[0] if entities else '')
                else:
                    # Generic entity search
                    results = self._query_entities(entities, max_results)
            except Exception as e:
                self.logger.warning(f"Hub query failed: {e}, using fallback")
                results = self._fallback_query_results(parsed_query)
        else:
            # Use fallback results
            results = self._fallback_query_results(parsed_query)

        return results[:max_results]

    def _query_entity_description(self, entity: str) -> List[Dict]:
        """Query for entity description."""
        results = []

        # Get triples about this entity
        if self.hub and hasattr(self.hub, 'get_triples'):
            triples = self.hub.get_triples(subject=entity)
            for t in triples:
                results.append({
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'confidence': getattr(t, 'confidence', 1.0),
                    'source': getattr(t, 'source', 'unknown')
                })

        # Also get as object
        if self.hub and hasattr(self.hub, 'get_triples'):
            triples = self.hub.get_triples(object=entity)
            for t in triples:
                results.append({
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'confidence': getattr(t, 'confidence', 1.0),
                    'source': getattr(t, 'source', 'unknown')
                })

        return results

    def _query_relationships(self, entities: List[str], parsed_query: Dict) -> List[Dict]:
        """Query for relationships between entities."""
        results = []

        for entity in entities:
            if self.hub and hasattr(self.hub, 'get_triples'):
                # Outgoing relationships
                triples = self.hub.get_triples(subject=entity)
                for t in triples:
                    results.append({
                        'subject': t.subject,
                        'predicate': t.predicate,
                        'object': t.object,
                        'direction': 'outgoing',
                        'confidence': getattr(t, 'confidence', 1.0),
                        'source': getattr(t, 'source', 'unknown')
                    })

                # Incoming relationships
                triples = self.hub.get_triples(object=entity)
                for t in triples:
                    results.append({
                        'subject': t.subject,
                        'predicate': t.predicate,
                        'object': t.object,
                        'direction': 'incoming',
                        'confidence': getattr(t, 'confidence', 1.0),
                        'source': getattr(t, 'source', 'unknown')
                    })

        return results

    def _query_paths(self, source: str, target: str) -> List[Dict]:
        """Query for paths between entities."""
        results = []

        if self.hub and hasattr(self.hub, 'find_paths'):
            paths = self.hub.find_paths(source, target, max_length=3)
            for i, path in enumerate(paths):
                results.append({
                    'path_id': i + 1,
                    'path': path,
                    'length': len(path),
                    'source': source,
                    'target': target
                })

        return results

    def _query_list(self, entity_type: str, max_results: int) -> List[Dict]:
        """Query for list of entities of a type."""
        results = []

        if self.hub and hasattr(self.hub, 'get_triples'):
            # Find entities of this type
            triples = self.hub.get_triples(predicate='is_a', object=entity_type)
            for t in triples[:max_results]:
                results.append({
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'type': entity_type,
                    'confidence': getattr(t, 'confidence', 1.0)
                })

        return results

    def _query_count(self, entity_type: str) -> List[Dict]:
        """Query for count of entities."""
        results = []

        if self.hub and hasattr(self.hub, 'get_triples'):
            triples = self.hub.get_triples(predicate='is_a', object=entity_type)
            results.append({
                'count': len(triples),
                'entity_type': entity_type,
                'query_type': 'count'
            })
        else:
            results.append({
                'count': 0,
                'entity_type': entity_type,
                'query_type': 'count',
                'note': 'Knowledge graph not available'
            })

        return results

    def _query_entities(self, entities: List[str], max_results: int) -> List[Dict]:
        """Generic entity search."""
        results = []

        for entity in entities:
            if self.hub and hasattr(self.hub, 'get_triples'):
                triples = self.hub.get_triples(subject=entity)
                for t in triples[:max_results]:
                    results.append({
                        'subject': t.subject,
                        'predicate': t.predicate,
                        'object': t.object,
                        'confidence': getattr(t, 'confidence', 1.0),
                        'source': getattr(t, 'source', 'unknown')
                    })

        return results

    def _fallback_query_results(self, parsed_query: Dict) -> List[Dict]:
        """
        Generate fallback results when knowledge graph is not available.
        """
        entities = parsed_query.get('entities', [])
        query_type = parsed_query.get('query_type', 'unknown')

        # Return a placeholder result indicating fallback mode
        return [{
            'fallback': True,
            'message': 'Knowledge graph not available. This is a fallback response.',
            'query_type': query_type,
            'entities_detected': entities,
            'suggestion': 'Please ensure the knowledge engine is properly configured.'
        }]

    def _generate_response(
        self,
        query: str,
        parsed_query: Dict,
        results: List[Dict],
        response_format: str,
        include_sources: bool,
        language: str,
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate natural language response from query results.
        """
        query_type = parsed_query.get('query_type', 'unknown')
        history = history or []

        # Handle fallback mode
        if results and results[0].get('fallback'):
            return results[0].get('message', 'Knowledge graph not available.')

        # Generate response based on format
        if response_format == 'concise':
            return self._generate_concise_response(query_type, results, include_sources)
        elif response_format == 'structured':
            return self._generate_structured_response(query_type, results, include_sources)
        else:  # detailed
            return self._generate_detailed_response(query_type, query, results, include_sources, history)

    def _generate_concise_response(
        self,
        query_type: str,
        results: List[Dict],
        include_sources: bool
    ) -> str:
        """Generate a concise response."""
        if not results:
            return "I couldn't find any information about that."

        if query_type == 'count_query':
            count = results[0].get('count', 0)
            entity_type = results[0].get('entity_type', 'items')
            return f"There are {count} {entity_type}."

        if query_type == 'entity_description':
            # Get key facts
            facts = []
            for r in results[:3]:
                pred = r.get('predicate', '')
                obj = r.get('object', '')
                if pred and obj:
                    facts.append(f"{pred} {obj}")
            if facts:
                return f"It is {', '.join(facts)}."

        # Default concise response
        entities = set()
        for r in results:
            if 'subject' in r:
                entities.add(r['subject'])
            if 'object' in r:
                entities.add(r['object'])

        if entities:
            return f"Found: {', '.join(list(entities)[:5])}."

        return "Information found but unable to summarize concisely."

    def _generate_detailed_response(
        self,
        query_type: str,
        query: str,
        results: List[Dict],
        include_sources: bool,
        history: List[Dict]
    ) -> str:
        """Generate a detailed response."""
        if not results:
            return "I searched the knowledge graph but couldn't find any information matching your query."

        response_parts = []

        if query_type == 'count_query':
            count = results[0].get('count', 0)
            entity_type = results[0].get('entity_type', 'items')
            response_parts.append(f"Based on the knowledge graph, there are {count} {entity_type}.")

        elif query_type == 'entity_description':
            subject = results[0].get('subject', 'the entity') if results else 'the entity'
            response_parts.append(f"Here's what I found about {subject}:")

            for r in results[:5]:
                pred = r.get('predicate', '')
                obj = r.get('object', '')
                if pred and obj:
                    response_parts.append(f"- {pred}: {obj}")

        elif query_type == 'relationship_query':
            response_parts.append("Here are the relationships I found:")

            for r in results[:5]:
                sub = r.get('subject', '')
                pred = r.get('predicate', '')
                obj = r.get('object', '')
                direction = r.get('direction', '')
                if sub and pred and obj:
                    if direction == 'outgoing':
                        response_parts.append(f"- {sub} {pred} {obj}")
                    elif direction == 'incoming':
                        response_parts.append(f"- {obj} is {pred} by {sub}")
                    else:
                        response_parts.append(f"- {sub} {pred} {obj}")

        elif query_type == 'path_query':
            if results:
                response_parts.append("Here are the paths I found:")
                for r in results[:3]:
                    path = r.get('path', [])
                    if path:
                        response_parts.append(f"- Path ({len(path)} steps): {' -> '.join(str(p) for p in path)}")
            else:
                response_parts.append("I couldn't find any direct paths between these entities.")

        else:
            # Generic response
            response_parts.append("Here's what I found in the knowledge graph:")

            for r in results[:5]:
                if 'subject' in r and 'predicate' in r and 'object' in r:
                    response_parts.append(f"- {r['subject']} {r['predicate']} {r['object']}")

        # Add sources if requested
        if include_sources:
            sources = set()
            for r in results:
                if 'source' in r and r['source']:
                    sources.add(str(r['source']))
            if sources:
                response_parts.append(f"\nSources: {', '.join(list(sources)[:3])}")

        return '\n'.join(response_parts)

    def _generate_structured_response(
        self,
        query_type: str,
        results: List[Dict],
        include_sources: bool
    ) -> str:
        """Generate a structured JSON-like response."""
        structured = {
            'query_type': query_type,
            'result_count': len(results),
            'results': results[:10]
        }

        return json.dumps(structured, indent=2, default=str)

    def _generate_suggestions(
        self,
        query: str,
        results: List[Dict],
        parsed_query: Dict,
        history: Optional[List[Dict]] = None,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Generate follow-up query suggestions.
        """
        suggestions = []
        query_type = parsed_query.get('query_type', 'unknown')
        entities = parsed_query.get('entities', [])

        # Entity-based suggestions
        if entities:
            main_entity = entities[0]
            suggestions.extend([
                f"What is {main_entity}?",
                f"Who created {main_entity}?",
                f"What is related to {main_entity}?"
            ])

        # Query type-based suggestions
        if query_type == 'entity_description' and entities:
            suggestions.append(f"Where is {entities[0]} located?")
            suggestions.append(f"When was {entities[0]} founded?")

        elif query_type == 'relationship_query':
            if entities:
                suggestions.append(f"What else did {entities[0]} create?")
                suggestions.append(f"Tell me more about {entities[0]}")

        elif query_type == 'list_query':
            suggestions.append("How many are there?")
            suggestions.append("Which is the most recent?")

        # History-based suggestions
        if history:
            last_query = history[-1].get('query', '')
            if last_query:
                suggestions.append(f"Tell me more about {last_query}")

        # Add generic suggestions if we need more
        generic_suggestions = [
            "What else can you tell me?",
            "Show me related information",
            "Give me more details"
        ]

        for suggestion in generic_suggestions:
            if len(suggestions) < max_suggestions:
                suggestions.append(suggestion)

        return suggestions[:max_suggestions]

    def _identify_ambiguities(self, parsed_query: Dict, query: str) -> List[Dict]:
        """
        Identify ambiguous aspects of the query.
        """
        ambiguities = []
        entities = parsed_query.get('entities', [])

        # Check for multiple entities that might be confused
        if len(entities) > 2:
            ambiguities.append({
                'type': 'multiple_entities',
                'message': f"Found {len(entities)} entities. Which one are you most interested in?",
                'options': entities[:5]
            })

        # Check for vague query type
        if parsed_query.get('query_type') == 'unknown':
            ambiguities.append({
                'type': 'unknown_intent',
                'message': "I'm not sure what you're looking for. Are you asking about:",
                'options': [
                    "A specific entity or person",
                    "A relationship between things",
                    "A list or count of items",
                    "A comparison"
                ]
            })

        # Check for pronouns without context
        pronoun_pattern = r'\b(it|they|them|their|this|that|these|those)\b'
        if re.search(pronoun_pattern, query.lower()) and not entities:
            ambiguities.append({
                'type': 'unclear_reference',
                'message': "You used pronouns but I couldn't determine what you're referring to. Could you clarify?",
                'options': []
            })

        return ambiguities

    def _generate_clarification_response(self, ambiguities: List[Dict], language: str) -> Dict:
        """
        Generate a clarification response with options.
        """
        if not ambiguities:
            return {
                'message': "Your query seems clear. Proceeding with search.",
                'options': []
            }

        # Build clarification message
        messages = []
        all_options = []

        for amb in ambiguities[:2]:  # Limit to top 2 ambiguities
            messages.append(amb['message'])
            all_options.extend(amb.get('options', []))

        return {
            'message': '\n\n'.join(messages),
            'options': all_options[:5]
        }

    def _enhance_query_with_context(self, query: str, history: List[Dict]) -> str:
        """
        Enhance query using conversation history for context resolution.
        """
        if not history:
            return query

        query_lower = query.lower()

        # Check for pronouns that need resolution
        pronouns = ['it', 'they', 'them', 'their', 'this', 'that', 'these', 'those']
        needs_resolution = any(p in query_lower.split() for p in pronouns)

        if needs_resolution:
            # Get entities from recent history
            recent_entities = []
            for exchange in history[-3:]:  # Look at last 3 exchanges
                prev_query = exchange.get('query', '')
                # Extract entities from previous query (simple approach)
                entity_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b'
                found = re.findall(entity_pattern, prev_query)
                recent_entities.extend(found)

            if recent_entities:
                # Replace pronouns with the most recent entity
                # This is a simple approach - more sophisticated coreference resolution
                # would be better
                most_recent = recent_entities[-1]
                for pronoun in pronouns:
                    query = re.sub(rf'\b{pronoun}\b', most_recent, query, flags=re.IGNORECASE, count=1)

        return query

    def _get_conversation_history(self, conversation_id: str, context_window: int) -> List[Dict]:
        """
        Retrieve conversation history for the given conversation ID.
        """
        if not conversation_id or conversation_id not in self._conversation_history:
            return []

        history = self._conversation_history[conversation_id]

        # Return last N exchanges based on context window
        if context_window > 0:
            return history[-context_window:]
        return []

    def _store_exchange(
        self,
        conversation_id: str,
        query: str,
        response: str,
        context
    ):
        """
        Store an exchange in conversation history.
        """
        if not conversation_id:
            return

        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = []

        exchange = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response[:500]  # Limit stored response length
        }

        self._conversation_history[conversation_id].append(exchange)

        # Store in context for workflow tracking
        context.add_artifact('nl_exchange', exchange)

    def _calculate_confidence(self, parsed_query: Dict, results: List[Dict]) -> float:
        """
        Calculate overall confidence score for the response.
        """
        confidence_scores = []

        # Parser confidence
        parser_used = parsed_query.get('parser_used', 'unknown')
        if parser_used == 'nl_parser':
            confidence_scores.append(0.9)
        elif parser_used == 'fallback_pattern':
            confidence_scores.append(0.6)
        else:
            confidence_scores.append(0.5)

        # Query type confidence
        query_type = parsed_query.get('query_type', 'unknown')
        if query_type != 'unknown':
            confidence_scores.append(0.8)
        else:
            confidence_scores.append(0.4)

        # Results confidence
        if results:
            if results[0].get('fallback'):
                confidence_scores.append(0.3)
            else:
                confidence_scores.append(0.85)
        else:
            confidence_scores.append(0.2)

        # Average confidence
        return round(sum(confidence_scores) / len(confidence_scores), 2)

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration including all operation types
        and natural language processing parameters.
        """
        return {
            "type": "object",
            "title": "Natural Language Interface Configuration",
            "description": "Configure the conversational interface for knowledge querying",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The type of natural language operation to perform",
                    "enum": ["query", "converse", "clarify", "suggest"],
                    "enumNames": [
                        "Query - Execute a natural language query",
                        "Converse - Continue a conversation with context",
                        "Clarify - Handle ambiguous queries",
                        "Suggest - Generate query suggestions"
                    ],
                    "default": "query"
                },
                "query": {
                    "type": "string",
                    "title": "Query",
                    "description": "Natural language query to execute",
                    "default": ""
                },
                "conversation_id": {
                    "type": "string",
                    "title": "Conversation ID",
                    "description": "Unique identifier for tracking conversation context (auto-generated if empty)",
                    "default": ""
                },
                "response_format": {
                    "type": "string",
                    "title": "Response Format",
                    "description": "Format for the natural language response",
                    "enum": ["concise", "detailed", "structured"],
                    "enumNames": [
                        "Concise - Brief, to-the-point responses",
                        "Detailed - Comprehensive responses with context",
                        "Structured - JSON-formatted structured output"
                    ],
                    "default": "detailed"
                },
                "include_sources": {
                    "type": "boolean",
                    "title": "Include Sources",
                    "description": "Include source attribution in responses",
                    "default": True
                },
                "max_results": {
                    "type": "integer",
                    "title": "Maximum Results",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5
                },
                "language": {
                    "type": "string",
                    "title": "Language",
                    "description": "Language code for responses (e.g., 'en', 'es', 'fr')",
                    "default": "en"
                },
                "context_window": {
                    "type": "integer",
                    "title": "Context Window",
                    "description": "Number of previous exchanges to remember for context",
                    "minimum": 0,
                    "maximum": 20,
                    "default": 3
                }
            },
            "required": ["operation"],
            "dependencies": {
                "operation": {
                    "oneOf": [
                        {
                            "properties": {
                                "operation": {"enum": ["query"]}
                            },
                            "required": ["query"],
                            "description": "Execute a natural language query against the knowledge graph"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["converse"]}
                            },
                            "required": ["query", "conversation_id"],
                            "description": "Continue a conversation with context awareness"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["clarify"]}
                            },
                            "description": "Analyze query for ambiguity and request clarification"
                        },
                        {
                            "properties": {
                                "operation": {"enum": ["suggest"]}
                            },
                            "description": "Generate query suggestions based on context"
                        }
                    ]
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if at least fallback mode is available
        """
        # Node can work in fallback mode even without hub
        return True

    def clear_conversation_history(self, conversation_id: Optional[str] = None):
        """
        Clear conversation history.

        Args:
            conversation_id: If provided, clear only that conversation's history.
                           If None, clear all conversation history.
        """
        if conversation_id:
            if conversation_id in self._conversation_history:
                del self._conversation_history[conversation_id]
        else:
            self._conversation_history.clear()
