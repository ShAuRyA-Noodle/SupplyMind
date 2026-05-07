"""sources_v2 — Pass-6 expansion fleet (live realtime data sources).

Active modules (no auth required, all live-tested 2026-04-25):

  who_don              — WHO Disease Outbreak News (JSON API)
  gdelt_conflict       — GDELT 2.0 conflict-tone filter (UCDP substitute)
  gdelt_humanitarian   — GDELT 2.0 humanitarian filter (ReliefWeb substitute)
  noaa_ndbc            — NOAA realtime ocean buoys (chokepoint coverage)
  noaa_tides           — NOAA tides + currents at major US ports
  nasa_eonet           — NASA Earth Observatory natural events tracker

Auth-required (kept for future use; need API keys we don't have yet):

  ucdp                 — Uppsala Conflict Data Program (needs x-ucdp-access-token)
  reliefweb            — UN OCHA ReliefWeb v2 (needs registered "appname")

Schema for one event (uniform across all modules):

  {
    "source":          "who_don" | "gdelt_conflict" | ... ,
    "event_id":        unique within source,
    "title":           short string,
    "description":     longer string,
    "occurred_at_utc": ISO8601 string,
    "lat":             optional float,
    "lon":             optional float,
    "severity_proxy":  optional float in [0,1],
    "raw_url":         direct URL to source record (always REAL),
    "fetched_at_utc":  ISO8601 string,
    "inference_type":  "live_<source>",
  }
"""
__all__ = [
    "who_don",
    "gdelt_conflict", "gdelt_humanitarian",
    "noaa_ndbc", "noaa_tides", "nasa_eonet",
    "ucdp", "reliefweb",
]
