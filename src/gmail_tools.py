from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailClient:
    def __init__(self, client_secret_file: str, token_file: str):
        self.client_secret_file = client_secret_file
        self.token_file = token_file
        self._service = None

    def _authenticate(self) -> None:
        creds = None
        token_path = Path(self.token_file)

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, SCOPES)
                creds = flow.run_local_server(port=0)

            with token_path.open("w", encoding="utf-8") as handle:
                handle.write(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)

    @property
    def service(self):
        if self._service is None:
            self._authenticate()
        return self._service

    @staticmethod
    def _decode_base64url(data: str) -> str:
        padding = "=" * (-len(data) % 4)
        decoded = base64.urlsafe_b64decode(data + padding)
        return decoded.decode("utf-8", errors="replace")

    @staticmethod
    def _header_value(headers: list[dict[str, str]], key: str) -> str:
        for h in headers:
            if h.get("name", "").lower() == key.lower():
                return h.get("value", "")
        return ""

    def _extract_plain_text(self, payload: dict[str, Any]) -> str:
        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        data = body.get("data")

        if mime_type == "text/plain" and data:
            return self._decode_base64url(data)

        for part in payload.get("parts", []) or []:
            text = self._extract_plain_text(part)
            if text.strip():
                return text

        if data:
            return self._decode_base64url(data)

        return ""

    def list_message_ids(self, query: str, max_results: int = 10) -> list[str]:
        resp = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        return [m["id"] for m in resp.get("messages", [])]

    def get_message_details(self, msg_id: str) -> dict[str, Any]:
        message = (
            self.service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        payload = message.get("payload", {})
        headers = payload.get("headers", [])

        return {
            "id": msg_id,
            "thread_id": message.get("threadId", ""),
            "subject": self._header_value(headers, "Subject"),
            "from": self._header_value(headers, "From"),
            "to": self._header_value(headers, "To"),
            "cc": self._header_value(headers, "Cc"),
            "date": self._header_value(headers, "Date"),
            # "snippet": message.get("snippet", ""),
            "body_text": self._extract_plain_text(payload),
        }

    def get_message_metadata(self, msg_id: str) -> dict[str, Any]:
        """Fetch only headers — no body download."""
        message = (
            self.service.users()
            .messages()
            .get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["Subject", "From","To","Cc", "Date"],
            )
            .execute()
        )

        headers = message.get("payload", {}).get("headers", [])

        return {
            "id": msg_id,
            "thread_id": message.get("threadId", ""),
            "subject": self._header_value(headers, "Subject"),
            "from": self._header_value(headers, "From"),
            "to": self._header_value(headers, "To"),
            "cc": self._header_value(headers, "Cc"),
            "date": self._header_value(headers, "Date"),
        }

    def fetch_recent_messages(
        self, query: str, max_results: int, metadata_only: bool = False
    ) -> list[dict[str, Any]]:
        message_ids = self.list_message_ids(query=query, max_results=max_results)
        fetch = self.get_message_metadata if metadata_only else self.get_message_details
        # tqdm on fetching 
        messages =[]
        print(f"Fetching {len(message_ids)} Emails using query: {query} max_results {max_results}...")  
        for msg_id in tqdm(message_ids):
            messages.append(fetch(msg_id))

        return messages

