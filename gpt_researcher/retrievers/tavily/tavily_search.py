# Tavily API Retriever

# libraries
import os
from typing import Literal, Sequence
import requests
import json


class TavilySearch:
    """
    Tavily API Retriever
    """

    def __init__(self, query, headers=None, topic="general", query_domains=None):
        """
        Initializes the TavilySearch object.

        Args:
            query (str): The search query string.
            headers (dict, optional): Additional headers to include in the request. Defaults to None.
            topic (str, optional): The topic for the search. Defaults to "general".
            query_domains (list, optional): List of domains to include in the search. Defaults to None.
        """
        self.query = query
        self.headers = headers or {}
        self.topic = topic
        self.base_url = "https://api.tavily.com/search"
        self.api_keys = self.get_api_keys()
        self.api_key_index = 0
        self.headers = {
            "Content-Type": "application/json",
        }
        self.query_domains = query_domains or None

    def get_api_keys(self):
        """
        Gets the Tavily API keys (supports multiple keys separated by comma)
        Returns:
            list: List of API keys for round-robin polling
        """
        # First check headers for api key(s)
        api_keys_str = self.headers.get("tavily_api_key", "")
        
        # If not in headers, check environment variable
        if not api_keys_str:
            api_keys_str = os.environ.get("TAVILY_API_KEY", "")
        
        if not api_keys_str:
            print(
                "Tavily API key not found, set to blank. If you need a retriver, please set the TAVILY_API_KEY environment variable."
            )
            return [""]
        
        # Split by comma and strip whitespace
        api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
        
        if not api_keys:
            return [""]
        
        print(f"Loaded {len(api_keys)} Tavily API key(s)")
        return api_keys
    
    def get_next_api_key(self):
        """
        Get the next API key using round-robin polling
        Returns:
            str: The next API key in rotation
        """
        if not self.api_keys:
            return ""
        
        api_key = self.api_keys[self.api_key_index]
        self.api_key_index = (self.api_key_index + 1) % len(self.api_keys)
        return api_key


    def _search(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: str = "general",
        days: int = 2,
        max_results: int = 10,
        include_domains: Sequence[str] = None,
        exclude_domains: Sequence[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        use_cache: bool = True,
    ) -> dict:
        """
        Internal search method to send the request to the API.
        Uses round-robin polling for multiple API keys.
        """
        # Try each API key in rotation until one succeeds or all fail
        errors = []
        for attempt in range(len(self.api_keys)):
            api_key = self.get_next_api_key()
            
            if not api_key:
                continue
                
            data = {
                "query": query,
                "search_depth": search_depth,
                "topic": topic,
                "days": days,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "max_results": max_results,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_images": include_images,
                "api_key": api_key,
                "use_cache": use_cache,
            }

            try:
                response = requests.post(
                    self.base_url, data=json.dumps(data), headers=self.headers, timeout=100
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    # Store error for this API key
                    errors.append(f"API key {attempt + 1}: HTTP {response.status_code}")
                    if response.status_code == 401 or response.status_code == 403:
                        # Invalid API key, try next one
                        print(f"API key {attempt + 1} failed with status {response.status_code}, trying next key...")
                        continue
                    else:
                        # Other error, might be worth retrying with same key
                        response.raise_for_status()
            except requests.exceptions.RequestException as e:
                errors.append(f"API key {attempt + 1}: {str(e)}")
                print(f"Request failed with API key {attempt + 1}: {e}")
                continue
        
        # If all API keys failed, raise an error with all attempts
        error_msg = "All API keys failed. Errors: " + "; ".join(errors)
        raise Exception(error_msg)

    def search(self, max_results=10):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self._search(
                self.query,
                search_depth="basic",
                max_results=max_results,
                topic=self.topic,
                include_domains=self.query_domains,
            )
            sources = results.get("results", [])
            if not sources:
                raise Exception("No results found with Tavily API search.")
            # Return the results
            search_response = [
                {"href": obj["url"], "body": obj["content"]} for obj in sources
            ]
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []
        return search_response
