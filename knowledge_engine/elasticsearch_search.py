from elasticsearch import Elasticsearch
from typing import Dict, Any, List, Optional

class ElasticsearchSearchEngine:
    """
    A client for interacting with an Elasticsearch instance for searching.
    """
    def __init__(self, hosts: List[str], api_key: str):
        self.es = Elasticsearch(hosts=hosts, api_key=api_key)

    async def search(self, index: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs a search query against a specified Elasticsearch index.

        Args:
            index: The name of the Elasticsearch index to search.
            query: The Elasticsearch query body.

        Returns:
            A dictionary containing the search results.
        """
        print(f"Elasticsearch Client: Searching index '{index}' with query: {query}")
        response = await self.es.search(index=index, body=query)
        return response

    async def index_document(self, index: str, document: Dict[str, Any], id: Optional[str] = None) -> Dict[str, Any]:
        """
        Indexes a document into a specified Elasticsearch index.

        Args:
            index: The name of the Elasticsearch index.
            document: The document to index.
            id: Optional. The ID of the document. If not provided, Elasticsearch generates one.

        Returns:
            A dictionary containing the indexing response.
        """
        print(f"Elasticsearch Client: Indexing document into '{index}'")
        response = await self.es.index(index=index, document=document, id=id)
        return response
