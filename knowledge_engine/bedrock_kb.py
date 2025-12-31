import boto3
from typing import Dict, Any

class BedrockKnowledgeBaseClient:
    """
    A client for interacting with Amazon Bedrock Knowledge Bases.
    """
    def __init__(self, region_name: str = 'us-east-1'):
        self.client = boto3.client('bedrock-agent-runtime', region_name=region_name)

    async def query_knowledge_base(self, knowledge_base_id: str, query_text: str) -> Dict[str, Any]:
        """
        Queries a specific Amazon Bedrock Knowledge Base.

        Args:
            knowledge_base_id: The ID of the Bedrock Knowledge Base.
            query_text: The query string.

        Returns:
            A dictionary containing the response from the Bedrock Knowledge Base.
        """
        print(f"Bedrock KB Client: Querying KB '{knowledge_base_id}' with '{query_text}'")
        response = self.client.retrieve_and_generate(
            input={'text': query_text},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0' # Example model ARN
                }
            }
        )
        return response
