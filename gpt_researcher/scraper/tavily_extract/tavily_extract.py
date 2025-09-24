from bs4 import BeautifulSoup
import os
from ..utils import get_relevant_images, extract_title

class TavilyExtract:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session
        from tavily import TavilyClient
        self.api_keys = self.get_api_keys()
        self.api_key_index = 0
        self.tavily_clients = [TavilyClient(api_key=key) for key in self.api_keys if key]

    def get_api_keys(self) -> list:
        """
        Gets the Tavily API keys (supports multiple keys separated by comma)
        Returns:
        List of API keys for round-robin polling
        """
        try:
            api_keys_str = os.environ.get("TAVILY_API_KEY", "")
            if not api_keys_str:
                raise KeyError()
            
            # Split by comma and strip whitespace
            api_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]
            
            if not api_keys:
                raise KeyError()
            
            print(f"Loaded {len(api_keys)} Tavily API key(s) for extraction")
            return api_keys
        except KeyError:
            raise Exception(
                "Tavily API key not found. Please set the TAVILY_API_KEY environment variable.")
    
    def get_next_client(self):
        """
        Get the next Tavily client using round-robin polling
        Returns:
            TavilyClient: The next client in rotation
        """
        if not self.tavily_clients:
            raise Exception("No valid Tavily clients available")
        
        client = self.tavily_clients[self.api_key_index]
        self.api_key_index = (self.api_key_index + 1) % len(self.tavily_clients)
        return client

    def scrape(self) -> tuple:
        """
        This function extracts content from a specified link using the Tavily Python SDK, the title and
        images from the link are extracted using the functions from `gpt_researcher/scraper/utils.py`.
        Uses round-robin polling for multiple API keys.

        Returns:
          The `scrape` method returns a tuple containing the extracted content, a list of image URLs, and
        the title of the webpage specified by the `self.link` attribute. It uses the Tavily Python SDK to
        extract and clean content from the webpage. If any exception occurs during the process, an error
        message is printed and an empty result is returned.
        """
        
        errors = []
        # Try each client in rotation until one succeeds or all fail
        for attempt in range(len(self.tavily_clients)):
            try:
                client = self.get_next_client()
                response = client.extract(urls=self.link)
                
                if response['failed_results']:
                    errors.append(f"Client {attempt + 1}: Extraction failed")
                    print(f"Extraction failed with client {attempt + 1}, trying next...")
                    continue

                # Parse the HTML content of the response to create a BeautifulSoup object for the utility functions
                response_bs = self.session.get(self.link, timeout=4)
                soup = BeautifulSoup(
                    response_bs.content, "lxml", from_encoding=response_bs.encoding
                )

                # Since only a single link is provided to tavily_client, the results will contain only one entry.
                content = response['results'][0]['raw_content']

                # Get relevant images using the utility function
                image_urls = get_relevant_images(soup, self.link)

                # Extract the title using the utility function
                title = extract_title(soup)

                return content, image_urls, title

            except Exception as e:
                errors.append(f"Client {attempt + 1}: {str(e)}")
                print(f"Error with client {attempt + 1}: {e}")
                continue
        
        # If all clients failed, return empty result
        error_msg = "All Tavily clients failed. Errors: " + "; ".join(errors)
        print(f"Error! : {error_msg}")
        return "", [], ""