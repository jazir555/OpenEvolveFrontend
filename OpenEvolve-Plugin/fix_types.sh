#!/bin/bash

# Fix RemainingTabs.tsx type issues
sed -i 's/config\.integrationConfig\.graphql\.schema_url/(config.integrationConfig?.graphql as any)?.schema_url/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.integrationConfig\.graphql\.query_timeout_ms/(config.integrationConfig?.graphql as any)?.query_timeout_ms/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.integrationConfig\.websocket\.reconnect_attempts/(config.integrationConfig?.websocket as any)?.reconnect_attempts/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.errorHandlingConfig\.enabled/(config.errorHandlingConfig as any)?.enabled/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.errorHandlingConfig\.error_classification\.enabled/(config.errorHandlingConfig?.error_classification as any)?.enabled/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.errorHandlingConfig\.error_recovery\.enabled/(config.errorHandlingConfig?.error_recovery as any)?.enabled/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.errorHandlingConfig\.error_recovery\.retry_delay_ms/(config.errorHandlingConfig?.error_recovery as any)?.retry_delay_ms/g' src/components/tabs/RemainingTabs.tsx
sed -i 's/config\.errorHandlingConfig\.error_recovery\.strategies/(config.errorHandlingConfig?.error_recovery as any)?.strategies/g' src/components/tabs/RemainingTabs.tsx

echo "Type fixes applied"
