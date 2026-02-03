# MCP Migration Report

Generated: 2026-02-02T14:22:57.972907

## Summary
- Total MCP files found: 57
- Total tools identified: 5
- Deprecated tools: 0
- Conflicts detected: 0

## Files to Migrate
- mcp_client.py
- user_api_key_auth_mcp.py
- mcp_server_manager.py
- mcp_management_endpoints.py
- mcp_tools.py
- litellm_proxy_mcp_handler.py
- mcp.py
- mcp_server_manager.py
- response_mcp_call_arguments_delta_event.py
- response_mcp_call_arguments_done_event.py
- response_mcp_call_completed_event.py
- response_mcp_call_failed_event.py
- response_mcp_call_in_progress_event.py
- response_mcp_list_tools_completed_event.py
- response_mcp_list_tools_failed_event.py
- response_mcp_list_tools_in_progress_event.py
- tool_choice_mcp.py
- tool_choice_mcp_param.py
- bubblelab_crewai_mcp_server.py
- bubblelab_mcp_client.py
- bubblelabs_mcp_tools.py
- bubblelabs_mcp_tools_security_patch.py
- mcp_events.py
- mcp_native_tool.py
- mcp_tool_wrapper.py
- mcp_adapter.py
- mcp_events.py
- mcp_native_tool.py
- mcp_tool_wrapper.py
- mcp_adapter.py
- mcp_client.py
- datapizza_mcp_tools.py
- decomposition_mcp_tools.py
- mcp.py
- mcp_server.py
- graphiti_mcp_server.py
- guardrails_mcp_tools.py
- mcp_demo.py
- mcp_bridge.py
- mcp_server.py
- math_mcp_tools.py
- mcp_gateway_integration.py
- mcp_server.py
- leanaide_continuous_mcp.py
- leanaide_mcp_tools.py
- lmql_mcp_tools.py
- unified_mcp_gateway.py
- kggen_mcp_wrapper.py
- migrate_to_unified_mcp.py
- mcp_server.py
- mcp_gateway_integration.py
- mcp_server.py
- openevolve_mcp_tools.py
- roma_mdap_maker_mcp_tools.py
- steer_mcp_tools.py
- unified_mcp_server.py
- z3_mcp_tools.py

## Recommendations
- Consider consolidating 57 MCP files into unified_mcp_server.py

## Migration Steps
1. Review this report
2. Create backup: `python migrate_to_unified_mcp.py --backup`
3. Resolve any conflicts
4. Run unified server tests
5. Update client configurations
6. Archive old files

## Rollback
If issues occur, restore from backup using the restore.py script.