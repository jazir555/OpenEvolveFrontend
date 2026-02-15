#!/bin/bash
# Master script to generate all missing probe.sh scripts for core projects

set -e

FRONTEND_DIR="C:\Users\mmeadow\Documents\OpenEvolve\Frontend"
CORE_PROJECTS_DIR="${FRONTEND_DIR}/core-projects"
ADAPTERS_DIR="${FRONTEND_DIR}/glue/adapters"

# Array of projects needing probes
PROJECTS=(
  "adaptive_mdap"
  "agentic-context-engine"
  "agentjson"
  "ai-knowledge-graph"
  "arbor"
  "causal-learn"
  "cav-nlp"
  "claudiomiro"
  "cognitive-hydraulics"
  "crewAI"
  "datapizza"
  "DeepKE"
  "deep-research-agent"
  "detllm"
  "drift"
  "dspy"
  "dspy-helm"
  "DTS"
  "Formal-Reasoning-Mode"
  "foundry"
  "Generic-Knowledge-Extraction-Tool"
  "guardrails"
  "Iterative-Contextual-Refinements"
  "jsonformer"
  "kg-gen"
  "lagrange-mapper"
  "Lean4-LLM-Ai-Agent-Mooc"
  "LoongFlow"
  "Matryoshka"
  "mrs-core"
  "NeuralKG"
  "neuromancer"
  "OneKE"
  "outlines"
  "PAMI"
  "pygraphistry"
  "rlm"
  "ROMA"
  "slither"
  "steer"
  "uqsa"
  "valkey"
)

echo "Starting probe generation for ${#PROJECTS[@]} projects..."

for PROJECT in "${PROJECTS[@]}"; do
  ADAPTER_DIR="${ADAPTERS_DIR}/${PROJECT}-adapter"
  PROBES_DIR="${ADAPTER_DIR}/probes"
  PROBE_SCRIPT="${PROBES_DIR}/check_api.sh"

  echo "Creating probe for ${PROJECT}..."

  # Create directory structure
  mkdir -p "${PROBES_DIR}"

  # Create the probe script
  cat > "${PROBE_SCRIPT}" << EOF
#!/bin/bash
# Probe for ${PROJECT}
# Verifies the internal API is accessible
# Law of Runtime Truth: This probe must successfully execute before implementing the adapter

CONTAINER_NAME="${PROJECT}-core"
API_ENDPOINT="http://localhost:8080/health"  # Default endpoint - adjust per project

echo "Probing ${CONTAINER_NAME}..."

# Try to curl the API endpoint
if curl -f -s "${API_ENDPOINT}" > /dev/null 2>&1; then
    echo "✓ ${CONTAINER_NAME} API is accessible"
    exit 0
else
    echo "✗ ${CONTAINER_NAME} API is NOT accessible"
    echo "  Expected endpoint: ${API_ENDPOINT}"
    echo "  Please verify:"
    echo "  1. Container is running: docker ps | grep ${CONTAINER_NAME}"
    echo "  2. Port is correct: Check core-projects/${PROJECT}/README.md"
    echo "  3. Update API_ENDPOINT in this script if different"
    exit 1
fi
EOF

  # Make executable
  chmod +x "${PROBE_SCRIPT}"

  echo "  ✓ Created: ${PROBE_SCRIPT}"
done

echo ""
echo "Probe generation complete!"
echo "Created ${#PROJECTS[@]} probe scripts."
echo ""
echo "NEXT STEPS:"
echo "1. For each project, read core-projects/{project}/README.md to discover actual API endpoints"
echo "2. Update the API_ENDPOINT variable in each probe script"
echo "3. Run each probe to verify the container is accessible"
echo "4. If probe fails, the project has no API and needs a different integration strategy"
