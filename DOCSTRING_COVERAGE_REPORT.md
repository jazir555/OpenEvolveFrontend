================================================================================
DOCSTRING COVERAGE REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Files with issues: 8096
Missing module docstrings: 6488
Missing class docstrings: 8018
Missing function docstrings: 59146

PRIORITY 1: CRITICAL FILES (Core Functions)
--------------------------------------------------------------------------------

adversarial.py:
  - MISSING class docstring: MockEvaluator
  - MISSING function docstring: create_language_specific_evaluator
  - MISSING function docstring: evaluate_content
  - MISSING function docstring: create_specialized_evaluator
  - MISSING function docstring: create_comprehensive_openevolve_config
  - MISSING function docstring: run_unified_evolution
  - ... and 3 more

decomposition_engine.py:
  - MISSING function docstring: get_strategy_name
  - MISSING function docstring: get_strategy_name
  - MISSING function docstring: get_strategy_name
  - MISSING function docstring: get_strategy_name
  - MISSING function docstring: get_strategy_name
  - ... and 5 more

end_to_end_invention_planner.py:
  - MISSING function docstring: extract_from_node
  - MISSING function docstring: evaluate
  - MISSING function docstring: get_evaluation_details

maker_engine.py:
  - MISSING function docstring: render_prompt
  - MISSING function docstring: solve

mdap_engine.py:


PRIORITY 2: INTEGRATION FILES
--------------------------------------------------------------------------------

ace_hephaestus_bridge.py:
  - 12 missing function docstrings

ace_mcp_tools.py:
  - 4 missing function docstrings

ace_mcp_tools_EDGE_CASE_FIXES.py:
  - 4 missing function docstrings

ace_mcp_tools_FIXED.py:

ace_stage6_integration.py:
  - 11 missing function docstrings

ace_steer_integration.py:
  - 2 missing function docstrings

adversarial_adapter.py:

bubblelabs_evolution_integration.py:

bubblelabs_hephaestus_bridge_fixed.py:
  - 4 missing function docstrings

bubblelabs_leanaide_integration.py:

bubblelabs_mcp_tools.py:
  - 3 missing function docstrings

c2c_mcp_tools.py:
  - 1 missing function docstrings

claudiomiro_hephaestus_bridge.py:
  - 2 missing function docstrings

claudiomiro_mcp_tools.py:
  - 1 missing function docstrings

datapizza_mcp_tools.py:
  - 1 missing function docstrings

decomposition_hephaestus_bridge.py:
  - 1 missing function docstrings

decomposition_mcp_tools.py:
  - 1 missing function docstrings

evolution_adapter.py:

external_knowledge_integration.py:
  - MISSING class docstring: DuckDuckGoParser
  - 3 missing function docstrings

final_integration_test.py:
  - 1 missing function docstrings

hephaestus_client.py:
  - MISSING module docstring
  - MISSING class docstring: HephaestusClient
  - 4 missing function docstrings

hybrid_maker_integration.py:
  - MISSING class docstring: EvolutionResult
  - 2 missing function docstrings

invention_planner_integration_helpers.py:
  - MISSING class docstring: ValidatedMath
  - MISSING class docstring: ErrorSource
  - MISSING class docstring: InventionGoal

langchain_chroma_integration.py:
  - MISSING class docstring: Document
  - MISSING class docstring: RecursiveCharacterTextSplitter
  - 2 missing function docstrings

lean4_integration.py:
  - 3 missing function docstrings

leanaide_decomposition_integration.py:
  - MISSING class docstring: MathematicalDomain
  - MISSING class docstring: LeanProofStatus

leanaide_hephaestus_bridge.py:
  - 1 missing function docstrings

leanaide_mcp_tools.py:
  - 1 missing function docstrings

leanaide_sop_integration.py:

maker_integration_bridge.py:
  - 2 missing function docstrings

openevolve_integration.py:
  - MISSING class docstring: PromptConfig
  - MISSING class docstring: DatabaseConfig
  - MISSING class docstring: EvaluatorConfig
  - MISSING class docstring: EvolutionTraceConfig
  - MISSING class docstring: LLMConfigContainer
  - MISSING class docstring: OpenEvolveAPI
  - 18 missing function docstrings

openevolve_leanaide_bridge.py:

openevolve_leanaide_integration_system.py:
  - 4 missing function docstrings

openevolve_maker_integration.py:
  - 1 missing function docstrings

openevolve_mcp_tools.py:
  - 5 missing function docstrings

quick_test_integration.py:
  - 1 missing function docstrings

roma_mcp_tools.py:
  - 1 missing function docstrings

roma_mdap_maker_mcp_tools.py:
  - 1 missing function docstrings

sovereign_decomposition_hephaestus_integration.py:
  - 3 missing function docstrings

sovereign_team_integration.py:
  - MISSING module docstring

start_bubblelabs_integration.py:
  - 4 missing function docstrings

steer_hephaestus_bridge.py:
  - 2 missing function docstrings

steer_mcp_tools.py:
  - MISSING class docstring: RealityLock
  - MISSING class docstring: VerificationResult
  - MISSING class docstring: TeachingOption
  - MISSING class docstring: CustomSlopJudge
  - 2 missing function docstrings

test_ace_mcp_tools_security.py:
  - 1 missing function docstrings

test_bubblelabs_complete_integration.py:
  - MISSING class docstring: TestResult
  - MISSING class docstring: Colors
  - 8 missing function docstrings

test_integration.py:
  - 2 missing function docstrings

test_integration_openevolve.py:
  - MISSING class docstring: MockBlueTeam
  - MISSING class docstring: MockRedTeam
  - MISSING class docstring: MockEvaluatorTeam
  - MISSING class docstring: MockTeam
  - MISSING class docstring: MockModelConfig
  - 4 missing function docstrings

test_leanaide_client.py:
  - 2 missing function docstrings

test_leanaide_sop_integration.py:
  - MISSING class docstring: MockSOPWithContent
  - MISSING class docstring: MockSOPToString
  - MISSING class docstring: MockSOP
  - 5 missing function docstrings

test_n8n_integration.py:
  - MISSING module docstring
  - MISSING class docstring: TestN8NIntegration
  - 5 missing function docstrings

test_openevolve_client_enhanced.py:
  - MISSING class docstring: MockResult

test_openevolve_integration.py:
  - 2 missing function docstrings

test_openevolve_integration_verification.py:
  - MISSING class docstring: Colors

test_sidebar_integration.py:
  - MISSING module docstring

test_sovereign_integration.py:
  - MISSING module docstring
  - MISSING class docstring: TestEndToEndWorkflow
  - MISSING class docstring: TestIntegrationScenarios
  - MISSING class docstring: TestSystemIntegration

test_tripartite_integration.py:
  - 1 missing function docstrings

thorough_integration_test.py:
  - MISSING class docstring: IntegrationTester
  - 1 missing function docstrings

validate_generic_maker_integration.py:
  - MISSING class docstring: TestEvaluator
  - 2 missing function docstrings

verify_bubblelabs_integration.py:
  - 1 missing function docstrings

verify_integration.py:
  - 1 missing function docstrings

verify_mdap_maker_integration.py:

opik_integration.py:
  - 2 missing function docstrings

custom_integration_example.py:
  - MISSING class docstring: AgentResult
  - 2 missing function docstrings

run_local_adapter.py:
  - 3 missing function docstrings

test_instructor_integration.py:
  - 5 missing function docstrings

test_integration.py:
  - 1 missing function docstrings

base_agent_adapter.py:
  - MISSING module docstring

base_tool_adapter.py:
  - MISSING module docstring

test_flow_human_input_integration.py:
  - MISSING module docstring
  - 2 missing function docstrings

test_human_feedback_integration.py:
  - MISSING class docstring: ReviewFlow
  - MISSING class docstring: ReviewFlow
  - MISSING class docstring: MultiStepFlow
  - MISSING class docstring: MixedFlow
  - MISSING class docstring: StateFlow
  - MISSING class docstring: HistoryFlow
  - MISSING class docstring: AsyncFlow
  - MISSING class docstring: ReviewState
  - MISSING class docstring: StructuredFlow
  - MISSING class docstring: MetadataFlow
  - MISSING class docstring: EventFlow
  - MISSING class docstring: FallbackFlow
  - MISSING class docstring: WhitespaceFlow
  - MISSING class docstring: NoRoutingFlow
  - 24 missing function docstrings

test_streaming_integration.py:
  - MISSING class docstring: ResearchFlow
  - MISSING class docstring: SimpleFlow
  - MISSING class docstring: AsyncResearchFlow
  - 3 missing function docstrings

test_base_agent_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: ConcreteAgentAdapter
  - MISSING class docstring: DummyOutput
  - MISSING class docstring: ConcreteAgentAdapterWithoutRequiredMethods
  - 8 missing function docstrings

test_base_tool_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: ConcreteToolAdapter
  - MISSING class docstring: ConcreteToolAdapterWithoutRequiredMethods
  - 12 missing function docstrings

test_flow_crew_span_integration.py:
  - MISSING class docstring: SampleFlow
  - MISSING class docstring: SampleTestFlowNotSet
  - MISSING class docstring: SampleMultiCrewFlow
  - MISSING class docstring: AsyncTestFlow

enterprise_adapter.py:
  - MISSING module docstring

lancedb_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: LanceDBAdapter
  - 3 missing function docstrings

rag_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: RAGAdapter
  - 2 missing function docstrings

zapier_adapter.py:
  - MISSING module docstring
  - 1 missing function docstrings

mcp_adapter_test.py:
  - MISSING module docstring
  - 15 missing function docstrings

base_agent_adapter.py:
  - MISSING module docstring

base_tool_adapter.py:
  - MISSING module docstring

test_flow_human_input_integration.py:
  - MISSING module docstring
  - 2 missing function docstrings

test_human_feedback_integration.py:
  - MISSING class docstring: ReviewFlow
  - MISSING class docstring: ReviewFlow
  - MISSING class docstring: MultiStepFlow
  - MISSING class docstring: MixedFlow
  - MISSING class docstring: StateFlow
  - MISSING class docstring: HistoryFlow
  - MISSING class docstring: AsyncFlow
  - MISSING class docstring: ReviewState
  - MISSING class docstring: StructuredFlow
  - MISSING class docstring: MetadataFlow
  - MISSING class docstring: EventFlow
  - MISSING class docstring: FallbackFlow
  - MISSING class docstring: WhitespaceFlow
  - MISSING class docstring: NoRoutingFlow
  - 24 missing function docstrings

test_streaming_integration.py:
  - MISSING class docstring: ResearchFlow
  - MISSING class docstring: SimpleFlow
  - MISSING class docstring: AsyncResearchFlow
  - 3 missing function docstrings

test_base_agent_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: ConcreteAgentAdapter
  - MISSING class docstring: DummyOutput
  - MISSING class docstring: ConcreteAgentAdapterWithoutRequiredMethods
  - 8 missing function docstrings

test_base_tool_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: ConcreteToolAdapter
  - MISSING class docstring: ConcreteToolAdapterWithoutRequiredMethods
  - 12 missing function docstrings

test_flow_crew_span_integration.py:
  - MISSING class docstring: SampleFlow
  - MISSING class docstring: SampleTestFlowNotSet
  - MISSING class docstring: SampleMultiCrewFlow
  - MISSING class docstring: AsyncTestFlow

enterprise_adapter.py:
  - MISSING module docstring

lancedb_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: LanceDBAdapter
  - 3 missing function docstrings

rag_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: RAGAdapter
  - 2 missing function docstrings

zapier_adapter.py:
  - MISSING module docstring
  - 1 missing function docstrings

mcp_adapter_test.py:
  - MISSING module docstring
  - 15 missing function docstrings

anthropic_client.py:
  - MISSING module docstring

memory_adapter.py:
  - MISSING module docstring

test_anthropic_memory_adapter.py:
  - MISSING module docstring
  - 2 missing function docstrings

azure_openai_client.py:
  - MISSING module docstring
  - MISSING class docstring: AzureOpenAIClient

bedrock_client.py:
  - MISSING module docstring

memory_adapter.py:
  - MISSING module docstring

test_bedrock_memory_adapter.py:
  - MISSING module docstring

google_client.py:
  - MISSING module docstring

memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: GoogleMemoryAdapter

test_memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: Dummy
  - 5 missing function docstrings

memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: MistralMemoryAdapter

mistral_client.py:
  - MISSING module docstring
  - 1 missing function docstrings

test_mistral_client.py:
  - MISSING module docstring
  - 1 missing function docstrings

memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: OpenAIMemoryAdapter

openai_client.py:
  - MISSING module docstring

test_base_client.py:
  - MISSING module docstring
  - 1 missing function docstrings

test_memory_adapter.py:
  - MISSING module docstring
  - 11 missing function docstrings

memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: OpenAILikeMemoryAdapter

openai_completion_client.py:
  - MISSING module docstring

memory_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: WatsonXMemoryAdapter

watsonx_client.py:
  - MISSING module docstring

client_manager.py:
  - MISSING module docstring
  - MISSING class docstring: ClientManager

mock_client.py:
  - MISSING module docstring
  - MISSING class docstring: FakeMemoryAdapter

test_client_factory.py:
  - MISSING module docstring
  - 1 missing function docstrings

client.py:
  - MISSING module docstring
  - MISSING class docstring: InferenceClientModule
  - MISSING class docstring: StructuredResponseInferenceClientModule
  - MISSING class docstring: StreamInferenceClientModule
  - 3 missing function docstrings

test_mock_client.py:
  - MISSING module docstring
  - MISSING class docstring: TestModel
  - 8 missing function docstrings

memory_adapter.py:
  - MISSING module docstring
  - 1 missing function docstrings

mcp_client.py:
  - MISSING module docstring
  - 4 missing function docstrings

adapter.py:
  - MISSING module docstring
  - 4 missing function docstrings

client.py:
  - MISSING module docstring
  - MISSING class docstring: MCPClient

networkx_adapter.py:
  - MISSING module docstring
  - MISSING class docstring: NetworkxAdapter

bge_reranker_client.py:
  - MISSING class docstring: BGERerankerClient
  - 1 missing function docstrings

openai_reranker_client.py:
  - MISSING class docstring: OpenAIRerankerClient
  - 1 missing function docstrings

client.py:
  - MISSING class docstring: EmbedderConfig
  - MISSING class docstring: EmbedderClient
  - 2 missing function docstrings

client.py:
  - MISSING class docstring: LLMClient
  - 2 missing function docstrings

groq_client.py:
  - MISSING class docstring: GroqClient

openai_generic_client.py:
  - 1 missing function docstrings

test_bge_reranker_client_int.py:
  - 4 missing function docstrings

test_anthropic_client.py:
  - MISSING class docstring: MockRateLimitError
  - MISSING class docstring: MockAPIError

test_azure_openai_client.py:
  - MISSING module docstring
  - MISSING class docstring: DummyResponses
  - MISSING class docstring: DummyChatCompletions
  - MISSING class docstring: DummyChat
  - MISSING class docstring: DummyAzureClient
  - MISSING class docstring: DummyResponseModel
  - 4 missing function docstrings

test_client.py:
  - 1 missing function docstrings

client.py:
  - 1 missing function docstrings

test_diagnostic_integration.py:
  - 2 missing function docstrings

test_monitoring_integration.py:
  - 1 missing function docstrings

verifiable_client.py:
  - MISSING module docstring

client_compatibility_features_test.py:
  - MISSING module docstring
  - 4 missing function docstrings

client_compatibility_produce_consume_test.py:
  - MISSING module docstring
  - 3 missing function docstrings

adapter.py:
  - MISSING module docstring
  - MISSING class docstring: CacheControlAdapter
  - 1 missing function docstrings

adapters.py:
  - 1 missing function docstrings

_adapters.py:
  - MISSING module docstring
  - 18 missing function docstrings

adapters.py:
  - 1 missing function docstrings

_adapters.py:
  - MISSING module docstring
  - MISSING class docstring: Message
  - 1 missing function docstrings

_adapters.py:
  - MISSING module docstring
  - 18 missing function docstrings

authlib_tornado_integration.py:
  - MISSING module docstring
  - MISSING class docstring: TornadoIntegration

curl_httpclient.py:
  - MISSING class docstring: CurlAsyncHTTPClient
  - MISSING class docstring: CurlError
  - 6 missing function docstrings

httpclient.py:
  - 12 missing function docstrings

simple_httpclient.py:
  - MISSING module docstring
  - MISSING class docstring: _HTTPConnection
  - 8 missing function docstrings

tcpclient.py:
  - 11 missing function docstrings

curl_httpclient_test.py:
  - MISSING module docstring
  - MISSING class docstring: CurlHTTPClientCommonTestCase
  - MISSING class docstring: DigestAuthHandler
  - MISSING class docstring: CustomReasonHandler
  - MISSING class docstring: CustomFailReasonHandler
  - MISSING class docstring: CurlHTTPClientTestCase
  - 12 missing function docstrings

httpclient_test.py:
  - MISSING module docstring
  - MISSING class docstring: HelloWorldHandler
  - MISSING class docstring: PostHandler
  - MISSING class docstring: PutHandler
  - MISSING class docstring: RedirectHandler
  - MISSING class docstring: RedirectWithoutLocationHandler
  - MISSING class docstring: ChunkHandler
  - MISSING class docstring: AuthHandler
  - MISSING class docstring: CountdownHandler
  - MISSING class docstring: EchoPostHandler
  - MISSING class docstring: UserAgentHandler
  - MISSING class docstring: ContentLength304Handler
  - MISSING class docstring: PatchHandler
  - MISSING class docstring: AllMethodsHandler
  - MISSING class docstring: SetHeaderHandler
  - MISSING class docstring: InvalidGzipHandler
  - MISSING class docstring: HeaderEncodingHandler
  - MISSING class docstring: HTTPClientCommonTestCase
  - MISSING class docstring: RequestProxyTest
  - MISSING class docstring: HTTPResponseTestCase
  - MISSING class docstring: SyncHTTPClientTest
  - MISSING class docstring: SyncHTTPClientSubprocessTest
  - MISSING class docstring: HTTPRequestTestCase
  - MISSING class docstring: HTTPErrorTestCase
  - 88 missing function docstrings

simple_httpclient_test.py:
  - MISSING module docstring
  - MISSING class docstring: SimpleHTTPClientCommonTestCase
  - MISSING class docstring: TriggerHandler
  - MISSING class docstring: ContentLengthHandler
  - MISSING class docstring: HeadHandler
  - MISSING class docstring: OptionsHandler
  - MISSING class docstring: NoContentHandler
  - MISSING class docstring: SeeOtherPostHandler
  - MISSING class docstring: SeeOtherGetHandler
  - MISSING class docstring: HostEchoHandler
  - MISSING class docstring: NoContentLengthHandler
  - MISSING class docstring: EchoPostHandler
  - MISSING class docstring: RespondInPrepareHandler
  - MISSING class docstring: SimpleHTTPClientTestMixin
  - MISSING class docstring: TimeoutResolver
  - MISSING class docstring: SimpleHTTPClientTestCase
  - MISSING class docstring: SimpleHTTPSClientTestCase
  - MISSING class docstring: CreateAsyncHTTPClientTestCase
  - MISSING class docstring: HTTP100ContinueTestCase
  - MISSING class docstring: HTTP204NoContentTestCase
  - MISSING class docstring: HostnameMappingTestCase
  - MISSING class docstring: ResolveTimeoutTestCase
  - MISSING class docstring: BadResolver
  - MISSING class docstring: MaxHeaderSizeTest
  - MISSING class docstring: SmallHeaders
  - MISSING class docstring: LargeHeaders
  - MISSING class docstring: MaxBodySizeTest
  - MISSING class docstring: SmallBody
  - MISSING class docstring: LargeBody
  - MISSING class docstring: MaxBufferSizeTest
  - MISSING class docstring: LargeBody
  - MISSING class docstring: ChunkedWithContentLengthTest
  - MISSING class docstring: ChunkedWithContentLength
  - 106 missing function docstrings

tcpclient_test.py:
  - MISSING module docstring
  - MISSING class docstring: TestTCPServer
  - MISSING class docstring: TCPClientTest
  - MISSING class docstring: TimeoutResolver
  - MISSING class docstring: TestConnectorSplit
  - MISSING class docstring: ConnectorTest
  - MISSING class docstring: FakeStream
  - 44 missing function docstrings

phi2_integration.py:
  - 1 missing function docstrings

stage3_integration.py:
  - MISSING class docstring: SimpleState
  - 5 missing function docstrings

test_stage3_integration.py:
  - 12 missing function docstrings

client_session.py:
  - MISSING module docstring
  - MISSING class docstring: DatasetInfo
  - MISSING class docstring: AuthManagerProtocol
  - 7 missing function docstrings

test_client_session.py:
  - MISSING module docstring
  - 13 missing function docstrings

test_policy_integration.py:
  - 2 missing function docstrings

client.py:
  - MISSING module docstring

test_chat_client.py:
  - MISSING module docstring
  - 6 missing function docstrings

example07_custom_integration.py:
  - 1 missing function docstrings

lean_continuous_bridge.py:
  - MISSING module docstring
  - MISSING class docstring: CASResult
  - MISSING class docstring: LeanAideClient
  - MISSING class docstring: ProofVerifier
  - MISSING class docstring: VerifiedResult
  - MISSING class docstring: VerifiedODE
  - MISSING class docstring: ParsedExpression
  - 5 missing function docstrings

test_e2b_integration.py:
  - 1 missing function docstrings

test_logging_integration.py:
  - 5 missing function docstrings

integration.py:
  - 1 missing function docstrings

test_integration.py:

test_lean4_integration.py:
  - MISSING module docstring
  - MISSING class docstring: MockResponse
  - MISSING class docstring: TestLean4Integration
  - 8 missing function docstrings

test_pygraphistry_integration.py:
  - MISSING module docstring
  - MISSING class docstring: TestPygraphistryIntegration
  - 1 missing function docstrings

test_phi2_integration.py:
  - 2 missing function docstrings

test_all_stage_integrations.py:
  - 8 missing function docstrings

test_phase1_integration.py:
  - 6 missing function docstrings

test_stage3_integration.py:
  - 5 missing function docstrings


PRIORITY 3: OTHER FILES
--------------------------------------------------------------------------------

ace_analytics.py:
  - 1 missing function docstrings

ace_security_utils.py:
  - 4 missing function docstrings

ace_workflow_knowledge_extractor.py:
  - 3 missing function docstrings

advanced_cache.py:
  - 2 missing function docstrings

advanced_features.py:
  - 1 missing function docstrings

advanced_system_unit_tests.py:
  - 1 missing function docstrings

advanced_unit_tests_comprehensive.py:
  - 2 missing function docstrings

adversarial_mdap_mcts.py:
  - 3 missing class docstrings

adversarial_unified.py:
  - 5 missing class docstrings
  - 5 missing function docstrings

analyze_imports.py:
  - 1 missing function docstrings

api_endpoints.py:
  - 1 missing function docstrings

api_key_manager.py:
  - 2 missing function docstrings

api_server.py:
  - 5 missing class docstrings
  - 2 missing function docstrings

apply_ace_security_fixes.py:
  - 1 missing function docstrings

apply_code_quality_fixes.py:
  - 1 missing function docstrings

async_executor.py:
  - 7 missing function docstrings

auth_system.py:
  - 1 missing function docstrings

auto_approval.py:
  - 2 missing function docstrings

benchmark_configuration_performance.py:
  - 1 missing class docstrings
  - 3 missing function docstrings

benchmark_performance.py:
  - 1 missing class docstrings
  - 8 missing function docstrings

... and 7897 more files