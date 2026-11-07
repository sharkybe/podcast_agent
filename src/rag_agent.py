from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import tiktoken
import yaml
from openai import OpenAI


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rag_config_path() -> Path:
    config_override = os.environ.get("RAG_CONFIG_PATH") or os.environ.get("PODCAST_CONFIG_PATH")
    if config_override:
        override_path = Path(config_override)
        if override_path.is_absolute():
            return override_path
        return _project_root() / override_path
    return _project_root() / "config_podcast.yaml"


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise RuntimeError(f"Config must be a mapping: {config_path}")

    return config


def _config_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name) or {}
    if not isinstance(section, dict):
        raise RuntimeError(f"Config section '{section_name}' must be a mapping")
    return section


def _resolve_with_project_root(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def _extract_first_timestamp(text: str) -> str | None:
    match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if match:
        return match.group(1)
    return None


@dataclass
class ChunkRecord:
    chunk_id: int
    text: str
    episode_number: str
    episode_name: str
    link: str
    transcript_path: str
    chunk_index: int
    token_start: int
    token_end: int
    timestamp_hint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "episode_number": self.episode_number,
            "episode_name": self.episode_name,
            "link": self.link,
            "transcript_path": self.transcript_path,
            "chunk_index": self.chunk_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "timestamp_hint": self.timestamp_hint,
        }


class TranscriptChunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int, encoding_name: str = "cl100k_base"):
        if chunk_size_tokens < 32:
            raise ValueError("rag.chunk_size_tokens must be >= 32")
        if chunk_overlap_tokens < 0:
            raise ValueError("rag.chunk_overlap_tokens must be >= 0")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("rag.chunk_overlap_tokens must be smaller than rag.chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_transcript(
        self,
        chunk_start_id: int,
        transcript_text: str,
        episode_number: str,
        episode_name: str,
        link: str,
        transcript_path: str,
    ) -> list[ChunkRecord]:
        text = transcript_text.strip()
        if not text:
            return []

        tokens = self.encoding.encode(text)
        if not tokens:
            return []

        step = self.chunk_size_tokens - self.chunk_overlap_tokens
        chunks: list[ChunkRecord] = []
        idx = 0
        chunk_id = chunk_start_id

        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size_tokens, len(tokens))
            token_slice = tokens[start:end]
            chunk_text = self.encoding.decode(token_slice).strip()
            if not chunk_text:
                continue

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    episode_number=episode_number,
                    episode_name=episode_name,
                    link=link,
                    transcript_path=transcript_path,
                    chunk_index=idx,
                    token_start=start,
                    token_end=end,
                    timestamp_hint=_extract_first_timestamp(chunk_text),
                )
            )
            idx += 1
            chunk_id += 1

            if end >= len(tokens):
                break

        return chunks


class EmbeddingStore:
    def __init__(self, rag_config: dict[str, Any], openai_config: dict[str, Any]):
        self.rag_config = rag_config
        self.openai_config = openai_config

        self.embedding_provider = str(rag_config.get("embedding_provider") or "openai").strip().lower()
        self.openai_embedding_model = str(rag_config.get("openai_embedding_model") or "text-embedding-3-small")
        self.local_embedding_model = str(
            rag_config.get("local_embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []

        self._openai_client: OpenAI | None = None
        self._local_embedding_model = None

    def _embedding_dimension(self) -> int:
        if self.embedding_provider == "openai":
            probe = self._embed_texts_openai(["dimension probe"])
            return int(probe.shape[1])

        probe = self._embed_texts_local(["dimension probe"])
        return int(probe.shape[1])

    def _ensure_index(self) -> None:
        if self.index is not None:
            return
        dimension = self._embedding_dimension()
        self.index = faiss.IndexFlatIP(dimension)

    def _ensure_openai_client(self) -> OpenAI:
        if self._openai_client is not None:
            return self._openai_client

        api_key = self.openai_config.get("api_key")
        if not api_key:
            raise RuntimeError("Config field 'openai.api_key' is required for OpenAI embeddings")

        self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _ensure_local_embedding_model(self):
        if self._local_embedding_model is not None:
            return self._local_embedding_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require sentence-transformers. Install it with 'pip install sentence-transformers'."
            ) from exc

        self._local_embedding_model = SentenceTransformer(self.local_embedding_model)
        return self._local_embedding_model

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _embed_texts_openai(self, texts: list[str]) -> np.ndarray:
        client = self._ensure_openai_client()
        response = client.embeddings.create(
            model=self.openai_embedding_model,
            input=texts,
        )
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return self._normalize(vectors)

    def _embed_texts_local(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_local_embedding_model()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self.embedding_provider == "openai":
            return self._embed_texts_openai(texts)

        if self.embedding_provider == "local":
            return self._embed_texts_local(texts)

        raise RuntimeError(
            "Config field 'rag.embedding_provider' must be either 'openai' or 'local'"
        )

    def build_index(self, chunks: list[ChunkRecord], batch_size: int = 64) -> None:
        self._ensure_index()
        self.metadata = []
        if self.index is None:
            raise RuntimeError("Failed to initialize FAISS index")

        if self.index.ntotal > 0:
            self.index.reset()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_texts(texts)
            self.index.add(embeddings)
            for chunk in batch:
                self.metadata.append(chunk.to_dict())

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Run the index action first.")

        query_embedding = self.embed_texts([query])
        distances, indices = self.index.search(query_embedding, top_k)

        results: list[tuple[float, dict[str, Any]]] = []
        for score, index_pos in zip(distances[0], indices[0]):
            if index_pos < 0:
                continue
            if index_pos >= len(self.metadata):
                continue
            results.append((float(score), self.metadata[index_pos]))

        return results

    def save(self, faiss_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("No index to save. Run indexing first.")

        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(faiss_path))
        payload = {
            "embedding_provider": self.embedding_provider,
            "openai_embedding_model": self.openai_embedding_model,
            "local_embedding_model": self.local_embedding_model,
            "chunks": self.metadata,
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def load(self, faiss_path: Path, metadata_path: Path) -> None:
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.index = faiss.read_index(str(faiss_path))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadata = payload.get("chunks") or []


class Retriever:
    def __init__(self, store: EmbeddingStore, top_k: int):
        if top_k < 1:
            raise ValueError("rag.top_k_retrieval must be >= 1")
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        results = self.store.search(query, self.top_k)
        enriched: list[dict[str, Any]] = []
        for score, item in results:
            copy_item = dict(item)
            copy_item["score"] = score
            enriched.append(copy_item)
        return enriched


class PodcastRAG:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.openai_config = _config_section(config, "openai")
        self.paths_config = _config_section(config, "paths")
        self.rag_config = _config_section(config, "rag")

        if not self.rag_config:
            raise RuntimeError("Config section 'rag' is required")

        chunk_size = int(self.rag_config.get("chunk_size_tokens") or 800)
        chunk_overlap = int(self.rag_config.get("chunk_overlap_tokens") or 200)
        encoding_name = str(self.rag_config.get("token_encoding") or "cl100k_base")
        self.chunker = TranscriptChunker(chunk_size, chunk_overlap, encoding_name=encoding_name)

        self.store = EmbeddingStore(self.rag_config, self.openai_config)
        top_k = int(self.rag_config.get("top_k_retrieval") or 5)
        self.retriever = Retriever(self.store, top_k=top_k)

        faiss_path_cfg = str(self.rag_config.get("faiss_index_path") or "rag_index/podcast_index.faiss")
        metadata_path_cfg = str(self.rag_config.get("metadata_path") or "rag_index/podcast_metadata.json")
        self.faiss_path = _resolve_with_project_root(faiss_path_cfg)
        self.metadata_path = _resolve_with_project_root(metadata_path_cfg)
        if self.faiss_path is None or self.metadata_path is None:
            raise RuntimeError("rag.faiss_index_path and rag.metadata_path are required")

        self.qa_model = str(self.rag_config.get("qa_model") or self.openai_config.get("model") or "gpt-4o-mini")
        self.qa_temperature = float(self.rag_config.get("qa_temperature") or 0.2)

    def _load_summary_rows(self) -> list[dict[str, str]]:
        summary_csv = _resolve_with_project_root(self.paths_config.get("summary_csv"))
        if summary_csv is None or not summary_csv.exists():
            raise FileNotFoundError(f"Episodes CSV not found: {summary_csv}")

        rows: list[dict[str, str]] = []
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                rows.append({
                    "episode_number": (raw_row.get("episode_number") or "").strip(),
                    "episode_name": (raw_row.get("episode_name") or "").strip(),
                    "link": (raw_row.get("link") or "").strip(),
                    "transcript_path": (raw_row.get("transcript_path") or "").strip(),
                })
        return rows

    def _collect_chunks(self, limit: int | None = None) -> list[ChunkRecord]:
        rows = self._load_summary_rows()
        chunks: list[ChunkRecord] = []
        chunk_id = 0

        for row in rows:
            transcript_rel = row.get("transcript_path")
            transcript_path = _resolve_with_project_root(transcript_rel)
            if transcript_path is None or not transcript_path.exists():
                continue

            transcript_text = transcript_path.read_text(encoding="utf-8").strip()
            if not transcript_text:
                continue

            new_chunks = self.chunker.chunk_transcript(
                chunk_start_id=chunk_id,
                transcript_text=transcript_text,
                episode_number=row.get("episode_number") or "",
                episode_name=row.get("episode_name") or transcript_path.stem,
                link=row.get("link") or "",
                transcript_path=str(transcript_path),
            )
            chunks.extend(new_chunks)
            chunk_id += len(new_chunks)

            if limit is not None and len(chunks) >= limit:
                return chunks[:limit]

        return chunks

    def index_transcripts(self, limit: int | None = None) -> None:
        chunks = self._collect_chunks(limit=limit)
        if not chunks:
            raise RuntimeError(
                "No transcript chunks found. Make sure paths.summary_csv exists and rows contain valid transcript_path files."
            )

        self.store.build_index(chunks)
        self.store.save(self.faiss_path, self.metadata_path)
        print(f"Indexed {len(chunks)} chunks.")
        print(f"FAISS index: {self.faiss_path}")
        print(f"Metadata: {self.metadata_path}")

    def load_index(self) -> None:
        self.store.load(self.faiss_path, self.metadata_path)

    def _build_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in retrieved_chunks:
            episode_number = item.get("episode_number") or "unknown"
            episode_name = item.get("episode_name") or "Unknown episode"
            link = item.get("link") or ""
            chunk_index = item.get("chunk_index")
            timestamp_hint = item.get("timestamp_hint")
            score = item.get("score")

            header = (
                f"Episode {episode_number} | {episode_name} | chunk {chunk_index} | "
                f"timestamp {timestamp_hint or 'n/a'} | score {score:.3f}"
            )
            if link:
                header += f" | {link}"

            lines.append(header)
            lines.append(item.get("text") or "")
            lines.append("---")

        return "\n".join(lines)

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        self.load_index()
        retrieved = self.retriever.retrieve(question)
        if not retrieved:
            return "No relevant chunks were found in the index."

        context_block = self._build_context(retrieved)

        api_key = self.openai_config.get("api_key")
        if not api_key:
            raise RuntimeError("Config field 'openai.api_key' is required for QA")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.qa_model,
            temperature=self.qa_temperature,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user using only the retrieved podcast excerpts. "
                        "If evidence is missing, say you do not know. "
                        "Cite supporting episodes by number and title."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nRetrieved excerpts:\n{context_block}",
                },
            ],
        )

        answer = (response.output_text or "").strip()
        citations = []
        seen = set()
        for item in retrieved:
            key = (item.get("episode_number"), item.get("episode_name"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                f"- Episode {item.get('episode_number')}: {item.get('episode_name')} ({item.get('link') or 'no link'})"
            )

        if citations:
            answer += "\n\nSources:\n" + "\n".join(citations)

        return answer

    def interactive(self) -> None:
        self.load_index()
        print("Interactive RAG mode. Type 'exit' to stop.")
        while True:
            question = input("You> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            try:
                answer = self.ask(question)
            except Exception as exc:
                print(f"ERROR: {exc}")
                continue
            print(f"\nAssistant> {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Podcast transcript RAG")
    parser.add_argument(
        "--action",
        choices=["index", "query", "interactive"],
        required=True,
        help="index: build FAISS index, query: ask one question, interactive: chat loop",
    )
    parser.add_argument("--question", type=str, default="", help="Question for --action query")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of chunks to index")

    args = parser.parse_args()

    config = _load_config(_rag_config_path())
    rag = PodcastRAG(config)

    if args.action == "index":
        rag.index_transcripts(limit=args.limit)
        return

    if args.action == "query":
        if not args.question.strip():
            raise ValueError("--question is required when --action query")
        answer = rag.ask(args.question)
        print(answer)
        return

    rag.interactive()


if __name__ == "__main__":
    main()
