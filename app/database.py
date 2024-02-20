import os
from supabase.client import create_client, Client, ClientOptions
import logging

if not os.getenv("SHOW_HTTPX_INFO_LOGS", "True") == "True":
    logging.getLogger("httpx").setLevel(logging.WARNING)

url: str = os.environ.get("SUPABASE_URL", "not-provided")
key: str = os.environ.get("SUPABASE_KEY", "not-provided")
supabase_instance: Client = create_client(
    url,
    key,
    options=ClientOptions(
        postgrest_client_timeout=10, storage_client_timeout=10, auto_refresh_token=False
    ),
)
