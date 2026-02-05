import asyncio
import os
from pathlib import Path
import shutil

# Make sure the script can find the knowledge_engine module
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from knowledge_engine.engine import KnowledgeEngine

async def run_test():
    """
    An end-to-end test for the KnowledgeEngine.
    It indexes a directory, loads the created index, and queries it.
    """
    print("--- Knowledge Engine Test ---")

    # --- Setup ---
    # Use the 'knowledge_engine' directory as the project to index
    project_to_index = "knowledge_engine"
    output_directory = "test_index_output"
    
    # Clean up previous test runs
    if Path(output_directory).exists():
        print(f"🧹 Removing old test directory: {output_directory}")
        shutil.rmtree(output_directory)

    # A target structure for analysis. This guides the LLM in finding relationships.
    target_structure_description = """
    We are looking for a system that can:
    1. Analyze and understand code.
    2. Use Large Language Models (LLMs) to perform analysis.
    3. Be configured via YAML files.
    4. Store its findings in a structured JSON format.
    Key concepts: 'indexer', 'llm', 'config', 'json'
    """

    # Create a dummy secrets file if it doesn't exist to avoid errors.
    # The indexer's mock mode should prevent it from needing real keys.
    secrets_file = "mcp_agent.secrets.yaml"
    if not Path(secrets_file).exists():
        print(f"[WARN] {secrets_file} not found. Creating a dummy file.")
        with open(secrets_file, "w") as f:
            f.write("# Dummy file for testing. Fill with real keys for live mode.\n")

    # --- Step 1: Instantiate the Engine ---
    # We will run this in mock mode to avoid needing real API keys for the test.
    # To do this, we'll need to modify the config file it uses.
    
    # First, read the original config
    with open("knowledge_engine/indexer_config.yaml", "r") as f:
        test_config_content = f.read()

    # Modify it to enable mock mode
    test_config_content = test_config_content.replace(
        "mock_llm_responses: false", "mock_llm_responses: true"
    )
    
    # Write to a temporary test config file
    test_config_path = "knowledge_engine/test_indexer_config.yaml"
    with open(test_config_path, "w") as f:
        f.write(test_config_content)
        
    print("🚀 Instantiating KnowledgeEngine in mock mode.")
    engine = KnowledgeEngine(
        indexer_config_path=test_config_path,
        api_secrets_path=secrets_file
    )

    # --- Step 2: Index the project ---
    print(f"\n🧠 Indexing project: '{project_to_index}'")
    output_files = await engine.index_project(
        project_path=project_to_index,
        target_structure=target_structure_description,
        output_dir=output_directory
    )

    if not output_files:
        print("[FAIL] Test Failed: Indexing returned no output files.")
        return

    # --- Step 3: Load the created index ---
    repo_name = Path(project_to_index).name
    # The default index file name is "{repo_name}_index.json"
    expected_index_file = Path(output_directory) / f"{repo_name}_index.json"
    
    print(f"\n📂 Checking for index file at: {expected_index_file}")
    
    if not expected_index_file.exists():
        print(f"[FAIL] Test Failed: Expected index file was not created.")
        # Let's see what was created
        print("Contents of output directory:")
        for path in Path(output_directory).rglob('*'):
            print(path)
        return

    knowledge_data = engine.load_index(str(expected_index_file))

    if not knowledge_data:
        print("[FAIL] Test Failed: Could not load data from the created index file.")
        return
        
    print("[OK] Index loaded successfully.")

    # --- Step 4: Query the index ---
    print("\n🔍 Performing queries...")
    
    # Query 1: A general term
    query1 = "LLM"
    results1 = engine.query_index_by_keyword(knowledge_data, query1)
    assert len(results1) > 0, f"Query for '{query1}' should have returned results."
    print(f"[OK] Query 1 for '{query1}' returned {len(results1)} results as expected.")

    # Query 2: A more specific term
    query2 = "CodeIndexer"
    results2 = engine.query_index_by_keyword(knowledge_data, query2)
    assert len(results2) > 0, f"Query for '{query2}' should have returned results."
    print(f"[OK] Query 2 for '{query2}' returned {len(results2)} results as expected.")
    
    # --- Teardown ---
    print("\n🧹 Cleaning up test files...")
    os.remove(test_config_path)
    shutil.rmtree(output_directory)
    print("[OK] Test finished successfully!")


if __name__ == "__main__":
    asyncio.run(run_test())
