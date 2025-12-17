from __future__ import annotations

import os
from typing import Optional
import httpx

from dotenv import load_dotenv

from backend.similarity import AddressSimilarity, BaselineAddressSimilarity

load_dotenv()

class MapboxClient:
    def __init__(
        self,
        token: str | None = None,
        similarity: AddressSimilarity | None = None,
    ) -> None:
        self.token = token or os.getenv("MAPBOX_ACCESS_TOKEN")
        self.client = httpx.Client()
        self.similarity = similarity or BaselineAddressSimilarity()

        if not self.token:
            raise Exception("MAPBOX_ACCESS_TOKEN must be set")

    def geocode_best_match(self, query: str) -> Optional[str]:
        url = "https://api.mapbox.com/search/geocode/v6/forward"

        # TODO: implement function to find the best match and return it here
        request = self.client.get(url, params={"access_token": self.token, "q": query})

        addresses = [response["properties"]["full_address"] for response in request.json()["features"]]
        if not addresses:
            return None

        return max(addresses, key=lambda x: float(self.similarity.score(query, x)))
