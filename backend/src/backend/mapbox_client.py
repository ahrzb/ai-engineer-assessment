from __future__ import annotations

import os
from typing import Optional
from backend.similarity import address_similarity
import httpx

from dotenv import load_dotenv

load_dotenv()

class MapboxClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.getenv("MAPBOX_ACCESS_TOKEN")
        self.client = httpx.Client()

        if not self.token:
            raise Exception("MAPBOX_ACCESS_TOKEN must be set")

    def geocode_best_match(self, query: str) -> Optional[str]:
        url = "https://api.mapbox.com/search/geocode/v6/forward"

        # TODO: implement function to find the best match and return it here
        request = self.client.get(url, params={"access_token": self.token, "q": query})

        addresses = [response["properties"]["full_address"] for response in request.json()["features"]]
        if not addresses:
            return None

        addresses.sort(key=lambda x: address_similarity(query, x))

        return addresses[-1]
