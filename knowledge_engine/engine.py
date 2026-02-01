import asyncio
import json
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import yaml
import openai

from .indexer import CodeIndexer
from . import document_loader
from .core import KnowledgeState, EntityKnowledgeGraph
from .bedrock_kb import BedrockKnowledgeBaseClient
from .eks_kb import EKSKnowledgeBaseHandler
from .elasticsearch_search import ElasticsearchSearchEngine

# LLM client initialization - use fallback if not available
try:
    from llm_utils import initialize_llm_client
except ImportError:
    # Try adding parent directory to path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    try:
        from llm_utils import initialize_llm_client
    except ImportError:
        initialize_llm_client = None


class KnowledgeEngine:
    """
    A facade for the knowledge engine, providing a simplified interface
    to the underlying indexing and retrieval logic from the CodeIndexer.
    """

    def __init__(
        self,
        indexer_config_path: str = "knowledge_engine/indexer_config.yaml",
        api_secrets_path: str = "mcp_agent.secrets.yaml",
    ):
        """
        Initializes the KnowledgeEngine.

        Args:
            indexer_config_path: Path to the indexer's configuration file.
            api_secrets_path: Path to the API secrets configuration file.
        """
        self.indexer_config_path = indexer_config_path
        self.api_secrets_path = api_secrets_path
        print(
            f"🚀 Initializing KnowledgeEngine with config: {indexer_config_path} and secrets: {api_secrets_path}"
        )
        self.knowledge_state: KnowledgeState = KnowledgeState(query="initial_query") # Initialize with a dummy query
        self.entity_graph: EntityKnowledgeGraph = EntityKnowledgeGraph()
        
        self.llm_config = self._load_llm_config(indexer_config_path)
        self.llm_client: Optional[openai.AsyncOpenAI] = None
        self.llm_client_type: Optional[str] = None
        self._setup_logger() # Setup logger before LLM init
        asyncio.run(self._initialize_llm_client()) # Initialize LLM client asynchronously

        self.bedrock_client: Optional[BedrockKnowledgeBaseClient] = None
        self.eks_handler: Optional[EKSKnowledgeBaseHandler] = None
        self.elasticsearch_client: Optional[ElasticsearchSearchEngine] = None

        # These would typically be configured via a config file or environment variables
        # For demonstration, we instantiate them directly.
        try:
            self.bedrock_client = BedrockKnowledgeBaseClient()
        except Exception as e:
            print(f"⚠️ Could not initialize BedrockKnowledgeBaseClient: {e}. AWS credentials might be missing or invalid.")
        
        try:
            self.eks_handler = EKSKnowledgeBaseHandler()
        except Exception as e:
            print(f"⚠️ Could not initialize EKSKnowledgeBaseHandler: {e}. AWS credentials might be missing or invalid.")

        try:
            # Placeholder for Elasticsearch hosts and API key
            # In a real application, retrieve these securely from config/env
            es_hosts = ["http://localhost:9200"] 
            es_api_key = os.environ.get("ELASTICSEARCH_API_KEY", "your_elasticsearch_api_key")
            self.elasticsearch_client = ElasticsearchSearchEngine(hosts=es_hosts, api_key=es_api_key)
        except Exception as e:
            print(f"⚠️ Could not initialize ElasticsearchSearchEngine: {e}. Elasticsearch might not be running or config is invalid.")

    def _setup_logger(self):
        """Setup a basic logger for the KnowledgeEngine."""
        self.logger = logging.getLogger("KnowledgeEngine")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _load_llm_config(self, config_path: str) -> Dict[str, Any]:
        """Loads LLM configuration from the indexer config file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            # Assuming 'llm' section directly contains provider configs
            llm_section = config.get("llm", {})
            
            # Extract default models from the loaded config
            default_models = {
                "anthropic": llm_section.get("anthropic_default_model", "claude-sonnet-4-20250514"),
                "openai": llm_section.get("openai_default_model", "o3-mini"),
                "google": llm_section.get("google_default_model", "gemini-2.0-flash"),
            }
            return {"default_models": default_models, **llm_section} # Merge default_models into llm_section
        except FileNotFoundError:
            print(f"❌ LLM config file not found at {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"❌ Error parsing LLM config file {config_path}: {e}")
            return {}

    async def _initialize_llm_client(self):
        """Initializes the LLM client using the centralized llm_utils.initialize_llm_client."""
        if not self.llm_config:
            self.logger.warning("LLM configuration not loaded. Cannot initialize LLM client.")
            return

        try:
            # Load API secrets
            api_config = {}
            if Path(self.api_secrets_path).exists():
                with open(self.api_secrets_path, "r", encoding="utf-8") as f:
                    api_config = yaml.safe_load(f) or {}

            self.llm_client, self.llm_client_type = await initialize_llm_client(
                api_config=api_config,
                default_models=self.llm_config.get("default_models", {}),
                logger=self.logger,
                verbose_output=self.llm_config.get("verbose_output", False)
            )
            self.logger.info(f"LLM client initialized: {self.llm_client_type}")
        except ValueError as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
            self.llm_client_type = None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during LLM client initialization: {e}")
            self.llm_client = None
            self.llm_client_type = None

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Centralized LLM call method using the initialized client."""
        if not self.llm_client:
            self.logger.error("LLM client not initialized. Cannot make LLM call.")
            return "ERROR: LLM client not initialized."

        try:
            model = self.llm_config.get(f"{self.llm_client_type}_default_model", self.llm_config.get("model"))
            if not model:
                model = self.llm_config.get("default_models", {}).get(self.llm_client_type)

            if self.llm_client_type == "anthropic":
                response = await self.llm_client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                return content
            elif self.llm_client_type == "openai":
                response = await self.llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            else:
                return f"ERROR: Unsupported LLM client type: {self.llm_client_type}"
        except Exception as e:
            self.logger.error(f"Error during LLM call: {e}")
            return f"ERROR: LLM call failed - {e}"

    async def add_document(self, path_or_url: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Adds a document to the knowledge base by downloading and/or converting it to text.

        Args:
            path_or_url: The path to a local file or a URL of a document.
            output_dir: Optional directory to store downloaded or converted files.

        Returns:
            The extracted text content of the document, or None if processing fails.
        """
        print(f"📄 Processing document: {path_or_url}")
        temp_dir = None
        if output_dir:
            processing_dir = Path(output_dir)
            processing_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp())
            processing_dir = temp_dir

        try:
            # Check if it's a URL
            urls = document_loader.URLExtractor.extract_urls(path_or_url)
            if urls:
                file_url = urls[0]
                filename = document_loader.URLExtractor.infer_filename_from_url(file_url)
                file_path = processing_dir / filename
                print(f"📥 Downloading {file_url} to {file_path}...")
                download_result = await document_loader.download_file(file_url, str(file_path))
                if not download_result.get("success"):
                    print(f"❌ Download failed: {download_result.get('error')}")
                    return None
            else:
                file_path = Path(path_or_url)
                if not file_path.exists():
                    print(f"❌ File not found: {file_path}")
                    return None

            # Convert to markdown text
            ext = file_path.suffix.lower()
            text_content = None

            if ext == ".pdf":
                converter = document_loader.SimplePdfConverter()
                result = converter.convert_pdf_to_markdown(str(file_path))
                if result.get("success"):
                    text_content = result.get("markdown_content")
            elif ext in document_loader.PDFConverter.OFFICE_FORMATS:
                pdf_converter = document_loader.PDFConverter()
                pdf_path = pdf_converter.convert_office_to_pdf(file_path, str(processing_dir))
                converter = document_loader.SimplePdfConverter()
                result = converter.convert_pdf_to_markdown(str(pdf_path))
                if result.get("success"):
                    text_content = result.get("markdown_content")
            elif ext in ['.txt', '.md']:
                with open(file_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            else:
                 print(f"Unsupported file type: {ext}")
                 return None

            if text_content:
                print("✅ Document processed successfully.")
            else:
                print("❌ Document processing failed to extract text.")

            return text_content

        except Exception as e:
            print(f"❌ An error occurred during document processing: {e}")
            return None
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir)

    async def generate_knowledge(self, context: str, query: str) -> str:
        """
        Generates new knowledge based on the provided context and query using an LLM.
        """
        self.logger.info(f"🧠 Generating knowledge for query: {query}")
        return await self._call_llm(
            prompt=f"Context: {context}\n\nQuery: {query}\n\nGenerate knowledge based on the context to answer the query.",
            system_prompt=self.llm_config.get("system_prompt", "You are a helpful assistant."),
            max_tokens=self.llm_config.get("max_tokens", 1000),
            temperature=self.llm_config.get("temperature", 0.3),
        )

    async def compress_knowledge(self, knowledge_text: str) -> str:
        """
        Compresses given knowledge text using an LLM.
        """
        self.logger.info(f"📦 Compressing knowledge...")
        return await self._call_llm(
            prompt=f"Compress the following knowledge:\n\n{knowledge_text}",
            system_prompt="You are a helpful assistant that summarizes text concisely.",
            max_tokens=self.llm_config.get("compress_max_tokens", 500),
            temperature=self.llm_config.get("compress_temperature", 0.3),
        )

    async def query_bedrock_knowledge_base(self, knowledge_base_id: str, query: str) -> Dict[str, Any]:
        """
        Queries an Amazon Bedrock Knowledge Base.
        Requires AWS Bedrock client configuration.
        """
        if not self.bedrock_client:
            print("❌ BedrockKnowledgeBaseClient not initialized.")
            return {"error": "BedrockKnowledgeBaseClient not initialized"}
        print(f"☁️ Querying Amazon Bedrock Knowledge Base '{knowledge_base_id}' for: {query}")
        response = await self.bedrock_client.query_knowledge_base(knowledge_base_id, query)
        return response

    async def query_eks_knowledge_base(self, query: str) -> Dict[str, Any]:
        """
        Queries an EKS troubleshooting knowledge base.
        Requires EKS hosted MCP service integration.
        """
        if not self.eks_handler:
            print("❌ EKSKnowledgeBaseHandler not initialized.")
            return {"error": "EKSKnowledgeBaseHandler not initialized"}
        print(f"☸️ Querying EKS Knowledge Base for: {query}")
        response = await self.eks_handler.query_eks_knowledge_base(query)
        return response

    async def query_elasticsearch(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Queries an Elasticsearch instance for information.
        Requires Elasticsearch client configuration.
        """
        if not self.elasticsearch_client:
            print("❌ ElasticsearchSearchEngine not initialized.")
            return {"error": "ElasticsearchSearchEngine not initialized"}
        print(f"🔍 Querying Elasticsearch index '{index}' for: {query}")
        response = await self.elasticsearch_client.search(index, query)
        return response

    async def index_project(
        self,
        project_path: str,
        target_structure: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Analyzes and indexes a single project directory.

        This method uses the CodeIndexer to analyze a given project path against
        a target structure and saves the resulting index to the specified
        output directory.

        Args:
            project_path: The root path of the project to index.
            target_structure: A string describing the target structure for analysis.
            output_dir: The directory where the index file(s) will be saved.

        Returns:
            A dictionary containing the path to the created index file, or an empty dictionary on failure.
        """
        print(f"🔧 Starting indexing for project at: {project_path}")
        try:
            project_dir = Path(project_path).resolve()
            
            # The CodeIndexer's process_repository method works on a single directory.
            # We instantiate it with the necessary configs.
            indexer = CodeIndexer(
                code_base_path=str(project_dir.parent), # The base path for resolving relative file paths in the index.
                target_structure=target_structure,
                output_dir=output_dir,
                config_path=self.api_secrets_path,
                indexer_config_path=self.indexer_config_path,
            )

            # Call process_repository directly on the specific project directory
            repo_index = await indexer.process_repository(project_dir)
            
            # Manually save the index file, replicating logic from the original build_all_indexes
            output_filename = indexer.index_filename_pattern.format(
                repo_name=repo_index.repo_name
            )
            
            # Note: The original indexer created a subdir with the repo name. 
            # We will save directly into the specified output_dir for simplicity.
            output_file = Path(output_dir) / output_filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Use asdict from dataclasses to convert the RepoIndex object to a dict
            from dataclasses import asdict
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asdict(repo_index), f, indent=2, ensure_ascii=False)

            print(f"✅ Indexing completed. Index saved to: {output_file}")
            return {repo_index.repo_name: str(output_file)}

        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def load_index(self, index_path: str) -> Dict[str, Any]:
        """
        Loads a previously generated knowledge index from a JSON file.

        Args:
            index_path: The path to the JSON index file.

        Returns:
            A dictionary containing the loaded index data.
        """
        print(f"📂 Loading index from: {index_path}")
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: Index file not found at {index_path}")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Error: Could not decode JSON from {index_path}")
            return {}

    def query_index_by_keyword(
        self,
        index_data: Dict[str, Any],
        keyword: str
    ) -> list:
        """
        Performs a simple keyword search within the loaded index data.

        This search is case-insensitive and looks for the keyword in file paths,
        summaries, key concepts, and main functions.

        Args:
            index_data: The loaded index data (from load_index).
            keyword: The keyword to search for.

        Returns:
            A list of matching file summaries.
        """
        if not index_data:
            print("⚠️ Cannot query empty index.")
            return []

        print(f"🔍 Searching for keyword '{keyword}' in index...")
        keyword = keyword.lower()
        matches = []
        
        file_summaries = index_data.get("file_summaries", [])
        for summary in file_summaries:
            if (
                keyword in summary.get("file_path", "").lower()
                or keyword in summary.get("summary", "").lower()
                or keyword in " ".join(summary.get("key_concepts", [])).lower()
                or keyword in " ".join(summary.get("main_functions", [])).lower()
            ):
                matches.append(summary)
        
        print(f"🔎 Found {len(matches)} match(es) for '{keyword}'.")
        return matches

    # --- DeepCode Workflows Integration ---

    async def run_code_implementation_workflow(
        self,
        plan_file_path: str,
        target_directory: str,
        pure_code_mode: bool = True,
        enable_read_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Runs the DeepCode Code Implementation Workflow.

        Args:
            plan_file_path: Path to the implementation plan file (e.g., initial_plan.txt).
            target_directory: Directory where the code will be implemented.
            pure_code_mode: If True, focuses on pure code generation.
            enable_read_tools: If True, enables tools for reading existing code.

        Returns:
            A dictionary containing the workflow result.
        """
        self.logger.info(f"🚀 Starting Code Implementation Workflow with plan: {plan_file_path}")
        from workflows.code_implementation_workflow import CodeImplementationWorkflow
        
        try:
            workflow = CodeImplementationWorkflow(
                config_path=self.api_secrets_path # Use the engine's API secrets path
            )
            result = await workflow.run_workflow(
                plan_file_path=plan_file_path,
                target_directory=target_directory,
                pure_code_mode=pure_code_mode,
                enable_read_tools=enable_read_tools
            )
            if result["status"] == "success":
                self.logger.info("✅ Code Implementation Workflow completed successfully.")
            else:
                self.logger.error(f"❌ Code Implementation Workflow failed: {result.get('message', 'Unknown error.')}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error running Code Implementation Workflow: {e}")
            return {"status": "error", "message": str(e)}

    async def run_multi_agent_research_pipeline(
        self,
        input_source: str,
        enable_indexing: bool = True,
        progress_callback: Optional[Any] = None # Streamlit needs progress_callback
    ) -> Dict[str, Any]:
        """
        Runs the DeepCode Multi-Agent Research Pipeline.

        Args:
            input_source: URL to a research paper or a local file path.
            enable_indexing: If True, enables advanced intelligence analysis (indexing, GitHub acquisition).
            progress_callback: Optional callback function for progress updates.

        Returns:
            A dictionary containing the pipeline summary.
        """
        self.logger.info(f"🔬 Starting Multi-Agent Research Pipeline for: {input_source}")
        from workflows.agent_orchestration_engine import execute_multi_agent_research_pipeline
        from unittest.mock import MagicMock # For mocking logger if not provided

        logger_for_pipeline = self.logger # Use the engine's logger
        if progress_callback:
            # If a specific progress callback is provided, we might need a dummy logger
            # or ensure the progress_callback handles logging adequately.
            # For now, let's keep the engine's logger and pass the callback.
            try:
                progress_callback({"stage": "start", "message": "Research pipeline starting"})
            except (TypeError, ValueError, RuntimeError):
                self.logger.debug("Progress callback failed during research pipeline start.")

        try:
            pipeline_summary = await execute_multi_agent_research_pipeline(
                input_source=input_source,
                logger=logger_for_pipeline,
                progress_callback=progress_callback,
                enable_indexing=enable_indexing
            )
            self.logger.info("✅ Multi-Agent Research Pipeline completed successfully.")
            return {"status": "success", "summary": pipeline_summary}
        except Exception as e:
            self.logger.error(f"❌ Error running Multi-Agent Research Pipeline: {e}")
            return {"status": "error", "message": str(e)}

    async def run_chat_based_planning_pipeline(
        self,
        user_input: str,
        enable_indexing: bool = True,
        progress_callback: Optional[Any] = None # Streamlit needs progress_callback
    ) -> Dict[str, Any]:
        """
        Runs the DeepCode Chat-Based Planning & Implementation Pipeline.

        Args:
            user_input: User's coding requirements (chat input).
            enable_indexing: If True, enables advanced intelligence analysis (indexing, GitHub acquisition).
            progress_callback: Optional callback function for progress updates.

        Returns:
            A dictionary containing the pipeline summary.
        """
        self.logger.info(f"💬 Starting Chat-Based Planning Pipeline for input: {user_input[:50]}...")
        from workflows.agent_orchestration_engine import execute_chat_based_planning_pipeline
        from unittest.mock import MagicMock # For mocking logger if not provided

        logger_for_pipeline = self.logger # Use the engine's logger
        if progress_callback:
            # Similar to research pipeline, keep engine's logger and pass callback.
            try:
                progress_callback({"stage": "start", "message": "Chat-based planning pipeline starting"})
            except (TypeError, ValueError, RuntimeError):
                self.logger.debug("Progress callback failed during chat pipeline start.")

        try:
            pipeline_summary = await execute_chat_based_planning_pipeline(
                user_input=user_input,
                logger=logger_for_pipeline,
                progress_callback=progress_callback,
                enable_indexing=enable_indexing
            )
            self.logger.info("✅ Chat-Based Planning Pipeline completed successfully.")
            return {"status": "success", "summary": pipeline_summary}
        except Exception as e:
            self.logger.error(f"❌ Error running Chat-Based Planning Pipeline: {e}")
            return {"status": "error", "message": str(e)}


# Example usage
async def main():
    # NOTE: This requires API keys to be set up in `mcp_agent.secrets.yaml`
    # and a project to be available at the specified path.
    
    # Create a dummy secrets file if it doesn't exist
    if not Path("mcp_agent.secrets.yaml").exists():
        print("⚠️ mcp_agent.secrets.yaml not found. Creating a dummy file.")
        print("   Please fill it with your actual API keys for the indexer to work.")
        with open("mcp_agent.secrets.yaml", "w") as f:
            f.write(
"""
# Please fill in your API keys
# openai:
#   api_key: "sk-..."
#   base_url: "https://api.openai.com/v1" # Optional
#
# anthropic:
#   api_key: "sk-ant-..."
#
# google:
#   api_key: "AIza..."
"""
            )

    engine = KnowledgeEngine()
    
    # --- Document Loading Example ---
    doc_url = "https://arxiv.org/pdf/1706.03762" # A famous paper
    document_text = await engine.add_document(doc_url, output_dir="temp_docs")
    if document_text:
        print("\n--- Extracted Document Text (first 500 chars) ---")
        print(document_text[:500] + "...")

    # --- DeepCode Workflow Examples ---
    # To run these, you'd need actual plan files or user input.
    # For demonstration, these are commented out.
    # if Path("dummy_plan.txt").exists():
    #     print("\n--- Running Code Implementation Workflow ---")
    #     impl_result = await engine.run_code_implementation_workflow(
    #         plan_file_path="dummy_plan.txt",
    #         target_directory="generated_code_output"
    #     )
    #     print(f"Code Implementation Result: {impl_result}")

    # research_input = "https://arxiv.org/abs/2307.09288" # Example research paper
    # print(f"\n--- Running Multi-Agent Research Pipeline for {research_input} ---")
    # research_result = await engine.run_multi_agent_research_pipeline(input_source=research_input)
    # print(f"Research Pipeline Result: {research_result}")

    # chat_query = "Create a Python function to calculate the Fibonacci sequence up to N."
    # print(f"\n--- Running Chat-Based Planning Pipeline for: {chat_query[:30]}... ---")
    # chat_result = await engine.run_chat_based_planning_pipeline(user_input=chat_query)
    # print(f"Chat Pipeline Result: {chat_result}")

    # --- Indexing Example ---
    # Define a dummy project to index. We'll use the 'knowledge_engine' dir itself.
    project_to_index = "." 
    output_directory = "knowledge_index"
    
    # A target structure for analysis
    target = """
    This is a knowledge engine. We want to find concepts related to:
    - Indexing files
    - Querying for information
    - Interacting with LLMs
    """
    
    # Run the indexing
    # This will create an index file in the 'knowledge_index' directory
    await engine.index_project(project_to_index, target, output_directory)

    # --- Querying Example ---
    # Assuming the indexing created a file. Let's find it.
    index_dir = Path(output_directory)
    # The indexer creates a sub-directory with the project name. The project name for '.' is the current dir name.
    project_name = Path(project_to_index).resolve().name
    
    # The default index file name is '{repo_name}_index.json'
    # We need to find the output dir inside the project path
    index_files = list(index_dir.glob(f"**/{project_name}_index.json"))

    if index_files:
        index_file_path = index_files[0]
        print(f"\n--- Found index file: {index_file_path} ---")
        
        # Load the index
        knowledge_data = engine.load_index(str(index_file_path))
        
        # Perform a query
        if knowledge_data:
            search_term = "LLM"
            results = engine.query_index_by_keyword(knowledge_data, search_term)
            
            if results:
                print("\n--- Query Results ---")
                for result in results:
                    print(f"  File: {result['file_path']}")
                    print(f"  Summary: {result['summary']}")
                    print("-" * 20)
    else:
        print("\nCould not find index file to run query example.")


if __name__ == "__main__":
    asyncio.run(main())
