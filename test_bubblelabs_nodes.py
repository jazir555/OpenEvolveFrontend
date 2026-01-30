import asyncio
import logging
from bubblelabs_plugin_system import get_plugin_registry
from openevolve_bubblelabs_plugin import register_openevolve_plugin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting BubbleLabs Node Test")
    
    # 1. Get registry and ensure plugin is registered
    registry = get_plugin_registry()
    
    # 2. Load the plugin
    logger.info("Loading OpenEvolve plugin...")
    plugin = await registry.load_plugin("openevolve")
    
    if not plugin:
        logger.error("Failed to load plugin")
        return
        
    logger.info(f"Plugin loaded: {plugin.get_metadata().name}")
    
    # 3. Start plugin
    await registry.start_plugin("openevolve")
    
    # 4. List supported nodes
    if hasattr(plugin, "list_supported_nodes"):
        nodes = plugin.list_supported_nodes()
        logger.info(f"Supported nodes: {nodes}")
        
        # 5. Instantiate each node
        for node_type in nodes:
            try:
                node = plugin.get_node(node_type)
                if node:
                    logger.info(f"Successfully instantiated node: {node.get_display_name()} ({node_type})")
                    # Check if internal component loaded (if applicable)
                    if hasattr(node, "decomposer") and node.decomposer is None:
                        logger.warning(f"  - Internal component missing for {node_type}")
                else:
                    logger.error(f"Failed to instantiate node: {node_type}")
            except (RuntimeError, TypeError, AttributeError, ImportError) as e:
                logger.error(f"Error instantiating {node_type}: {e}")
    else:
        logger.error("Plugin does not have list_supported_nodes method")

    # 6. Cleanup
    await registry.unload_plugin("openevolve")
    logger.info("Test complete")

if __name__ == "__main__":
    asyncio.run(main())
