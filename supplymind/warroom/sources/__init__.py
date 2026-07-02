"""Realtime event sources."""
from . import newsapi, gdelt, usgs, marinetraffic, fred_brent

SOURCES = {
    "newsapi": newsapi,
    "gdelt": gdelt,
    "usgs": usgs,
    "marinetraffic": marinetraffic,
    "fred_brent": fred_brent,
}

__all__ = ["SOURCES", "newsapi", "gdelt", "usgs", "marinetraffic", "fred_brent"]
