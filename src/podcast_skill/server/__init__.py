"""Hosted answer service (build plan phase 3): the `ask` pipeline behind a small HTTP API.

`app.py` is the FastAPI app, `store.py` keeps the daily counters, follow-up history and answer
cache (Postgres on Render, in memory for local runs and tests).
"""
