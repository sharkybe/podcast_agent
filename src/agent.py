from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from openai import OpenAI

from agent_memory import MemoryStore, build_model_messages, update_profile_memory
from gmail_tools import GmailClient


class EmailAgent:
    def __init__(self, config: dict[str, Any] | None = None, config_path: str = "config.yaml"):
        self.cfg = config or self._load_config(config_path)

        if not self.cfg["openai_api_key"]:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to config.yaml")

        self.client = OpenAI(api_key=self.cfg["openai_api_key"])

        client_secret = Path(self.cfg["gmail_oauth_client_secret_file"])
        token_file = Path(self.cfg["gmail_token_file"])
        self.gmail_client = GmailClient(client_secret_file=str(client_secret), token_file=str(token_file))

        self.show_token_estimate = self.cfg.get("show_token_estimate", False)

    @staticmethod
    def _load_config(config_path: str = "config.yaml") -> dict[str, Any]:
        config_file = Path(config_path)
        if not config_file.exists():
            raise RuntimeError(
                f"Config file not found: {config_file}. "
                "Copy config.example.yaml to config.yaml and fill in your values."
            )

        with config_file.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        return raw

    @staticmethod
    def _trim_text(text: str, max_chars: int = 2500) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _estimate_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.cfg["openai_model"])
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _build_prompt(self, messages: list[dict[str, Any]], user_goal: str) -> str:
        compact_messages = []
        for message in messages:
            compact_messages.append(
                {
                    "subject": message.get("subject", ""),
                    "from": message.get("from", ""),
                    "date": message.get("date", ""),
                    "body_text": self._trim_text(message.get("body_text", "")),
                }
            )

        return (
            "You are an email assistant. Read the provided emails and produce a useful summary.\n"
            "Goals from user:\n"
            f"{user_goal}\n\n"
            "Required output format:\n"
            "1) High-priority emails (bulleted)\n"
            "2) Follow-ups needed today\n"
            "3) Waiting-for items\n"
            "4) Short digest\n\n"
            "Emails JSON:\n"
            f"{json.dumps(compact_messages, ensure_ascii=True)}"
        )

    def reset_memory(self, session_id: str) -> None:
        store = MemoryStore(Path(self.cfg["agent_memory_dir"]), session_id)
        store.reset()

    def perform_task(
        self,
        user_goal: str,
        session_id: str = "default",
    ) -> str:
        client_secret = Path(self.cfg["gmail_oauth_client_secret_file"])
        token_file = Path(self.cfg["gmail_token_file"])

        if not client_secret.exists():
            raise RuntimeError(
                f"Missing OAuth client secret file: {client_secret}. "
                "Create Desktop OAuth credentials in Google Cloud and download JSON here."
            )

        messages = self.gmail_client.fetch_recent_messages(
            query=self.cfg["gmail_query"],
            max_results=self.cfg["gmail_max_results"],
            metadata_only=True
        )

        if not messages:
            return "No emails found for the selected query."

        store = MemoryStore(Path(self.cfg["agent_memory_dir"]), session_id)
        history = store.load_history()
        profile = store.load_profile()
        email_context = self._build_prompt(messages, user_goal)
        model_input = build_model_messages(
            email_context=email_context,
            history=history,
            profile=profile,
            history_turns=self.cfg["agent_history_turns"],
            memory_items=self.cfg["agent_memory_items"],
        )

        if self.show_token_estimate:
            email_context_tokens = self._estimate_tokens(email_context)
            total_input_tokens = self._estimate_tokens(
                json.dumps(model_input, ensure_ascii=True)
            )
            print(f"[Token estimate] email_context: {email_context_tokens}")
            print(f"[Token estimate] total input payload: {total_input_tokens}")

        response = self.client.responses.create(
            model=self.cfg["openai_model"],
            input=model_input,
        )

        assistant_reply = response.output_text
        store.append_turn(user_goal, assistant_reply)
        update_profile_memory(
            client=self.client,
            model=self.cfg["openai_model"],
            store=store,
            user_message=user_goal,
            assistant_message=assistant_reply,
            memory_items=self.cfg["agent_memory_items"],
        )

        return assistant_reply

    def run_interactive(self, session_id: str) -> None:
        print("Interactive email assistant. Type 'exit' to stop.")
        while True:
            user_goal = input("You> ").strip()
            if not user_goal:
                continue
            if user_goal.lower() in {"exit", "quit"}:
                break

            result = self.perform_task(
                user_goal,
                session_id=session_id,
            )
            print(f"\nAssistant> {result}\n")


def main() -> None:

    parser = argparse.ArgumentParser(description="External Gmail-reading agent")
    parser.add_argument(
        "--goal",
        type=str,
        default="Highlight important emails and what I should do next.",
        help="What kind of summary/action plan you want.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="default",
        help="Conversation memory namespace to load and update.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the agent running and remember conversation turns in a local session.",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete saved conversation history and long-term memory before running.",
    )
    args = parser.parse_args()
    
    agent = EmailAgent()

    if args.reset_memory:
        agent.reset_memory(args.session_id)

    if args.interactive:
        agent.run_interactive(
            args.session_id,
        )
        return

    result = agent.perform_task(
        args.goal,
        session_id=args.session_id,
    )
    print(result)


if __name__ == "__main__":
    main()
