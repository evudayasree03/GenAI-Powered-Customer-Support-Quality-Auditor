"""
SamiX Knowledge Intelligence & RAG Engine

This module implements the Retrieval-Augmented Generation (RAG) pipeline 
that powers the platform's factual integrity and compliance audits.

Architecture:
1. Vector Store: Milvus Lite (local) for high-performance semantic search.
2. Embeddings: HuggingFace 'all-MiniLM-L6-v2' for local, CPU-optimized vectors.
3. Retrieval: Maximal Marginal Relevance (MMR) to ensure diverse context.
4. Fallback: Pure keyword-overlap search for environments without vector support.

Data Collections:
- 'policies': Internal company SOPs and support guidelines.
- 'compliance': Regulatory frameworks (GDPR, PCI-DSS, ISO).
- 'product_kb': Technical manuals and product specifications.
"""
from __future__ import annotations

import io
import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import streamlit as st


# RAG Configuration 

CHUNK_SIZE     = 800      # Increased for BGE-small's 512 token limit (~800-1000 chars)
CHUNK_OVERLAP  = 100
TOP_K          = 10      # Increased for reranking (pre-rerank pool)
RERANK_K       = 4       # Final Top-K returned after reranking
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MILVUS_DB      = "milvus_lite.db"
META_PATH      = "data/kb/kb_meta.json"
KB_DIR         = "data/kb"

COLLECTIONS    = ["policies", "product_kb", "compliance"]

# Seed Knowledge (Built-in) 
# These are default industry standards loaded into every new instance.
GENERALISED_KB: list[dict] = [
    {
        "name":       "Customer Support Best Practices (ITIL v4)",
        "collection": "policies",
        "chunks":     120,
        "content": textwrap.dedent("""\
            Incident management: acknowledge within 60 seconds.
            Always confirm customer identity before accessing account data.
            Closing protocol: confirm resolution before ending every call.
            Escalation path: agent -> senior agent -> supervisor -> manager.
            SLA for billing disputes: resolve within 2 business days.
            Empathy language: mirror customer emotion, then redirect to solution.
        """),
    },
    {
        "name":       "BPO Compliance Framework (ISO 9001-2015)",
        "collection": "compliance",
        "chunks":     98,
        "content": textwrap.dedent("""\
            All agents must follow approved scripts for regulated topics.
            Financial information must not be disclosed without identity verification.
            Call recordings are mandatory for quality assurance.
            Non-compliance must be reported within 24 hours.
            Corrective actions must be documented and reviewed quarterly.
        """),
    },
    {
        "name":       "De-escalation Techniques",
        "collection": "policies",
        "chunks":     65,
        "content": textwrap.dedent("""\
            Step 1: Let the customer finish speaking without interruption.
            Step 2: Acknowledge the frustration explicitly: 'I completely understand'.
            Step 3: Apologise for the inconvenience before moving to resolution.
            Step 4: Offer a concrete next step with a specific timeline.
            Step 5: Confirm the customer is satisfied before closing.
            Avoid: 'calm down', 'that is our policy', 'there is nothing I can do'.
        """),
    },
    {
        "name":       "GDPR Customer Data Handling",
        "collection": "compliance",
        "chunks":     88,
        "content": textwrap.dedent("""\
            Never read back full card numbers or passwords on a call.
            Only collect data necessary for the stated purpose.
            Customer has the right to data deletion within 30 days of request.
            Any data breach must be reported to the DPO within 72 hours.
            Call recordings may not be retained beyond 12 months without consent.
        """),
    },
    {
        "name":       "Empathy Language Patterns (50 phrases)",
        "collection": "policies",
        "chunks":     50,
        "content": textwrap.dedent("""\
            'I completely understand how frustrating that must be.'
            'I sincerely apologise that this has happened.'
            'Thank you for bringing this to our attention.'
            'I appreciate your patience while I look into this.'
            'I can absolutely see why you feel that way.'
            'Let me personally make sure this is resolved for you.'
            'I am going to take ownership of this issue right now.'
        """),
    },
    {
        "name":       "GenAI QA Auditor Standard Rubric",
        "collection": "product_kb",
        "chunks":     72,
        "content": textwrap.dedent("""\
            Empathy (20%): acknowledgment, emotional mirroring, apology quality.
            Professionalism (15%): language, tone, script adherence.
            Compliance (25%): policy accuracy, regulatory adherence, script compliance.
            Resolution (20%): issue resolved, root cause addressed, no false close.
            Communication (5%): clarity, pacing, active listening signals.
            Integrity (15%): factual accuracy, no hallucinations, correct policy citation.
            Phase bonus (+/-5): improving arc = +5, declining arc = -5.
            Auto-fail triggers: rude language, data breach, impossible promise.
        """),
    },
]


# Data Structures 

@dataclass
class KBFile:
    """ Metadata record for a document uploaded to the knowledge base. """
    filename:   str
    collection: str
    chunks:     int  = 0
    size_bytes: int  = 0
    indexed:    bool = False

    @property
    def size_label(self) -> str:
        """ Returns a human-readable file size (KB or MB). """
        kb = self.size_bytes / 1024
        return f"{kb:.1f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"


@dataclass
class RAGResult:
    """ A single snippet of retrieved knowledge and its associated metadata. """
    text:       str
    source:     str
    collection: str
    score:      float     # Semantic similarity score (higher is better).
    page:       int = 0

    def to_citation(self) -> str:
        """ Formats the result as a standard academic/legal citation. """
        return f"{self.source} (conf {self.score:.2f})"


# Knowledge Base Manager 

class KBManager:
    """
    The brain of the SamiX RAG system.
    
    Handles the ingestion of PDFs and text, maintains the Milvus vector stores, 
    and provides a high-level API for semantic querying and policy auditing.
    """

    def __init__(self) -> None:
        """ Initializes storage, loads metadata, and warms up the embedding model. """
        os.makedirs(KB_DIR, exist_ok=True)
        self._files: list[KBFile]      = []
        self._embeddings               = None
        self._reranker                 = None
        self._stores: dict[str, object] = {}   # Mapping of collection -> Milvus VectorStore
        self._load_meta()
        self._init_embeddings()
        self._init_reranker()
        self._reload_existing_stores()
        self._load_generalised_kb()
        self._autoload_dropped_files()

    @staticmethod
    def _safe_source_name(source: str) -> str:
        """ Normalizes a source name for chunk-backup filenames on Windows. """
        return source.replace(":", "-").replace("/", "_").replace("\\", "_")

    def _fallback_path(self, source: str) -> str:
        """ Returns the chunk-backup path for a given source. """
        return os.path.join(KB_DIR, f"{self._safe_source_name(source)}.chunks.txt")

    def _source_collection_map(self) -> dict[str, str]:
        """ Maps persisted chunk sources back to their owning collection. """
        mapping = {
            self._safe_source_name(f.filename): f.collection
            for f in self._files
        }
        mapping.update({
            self._safe_source_name(item["name"]): item["collection"]
            for item in GENERALISED_KB
        })
        return mapping

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """ Normalizes text for keyword retrieval and numeric policy checks. """
        return re.findall(r"[a-z0-9]+", text.lower())

    def _autoload_dropped_files(self) -> None:
        """ 
        Scans the KB directory for files added outside the UI.
        Ensures the system is always in sync with the physical disk.
        """
        known = {f.filename for f in self._files}
        for fname in os.listdir(KB_DIR):
            if fname.endswith(".chunks.txt") or fname == "kb_meta.json" or fname.startswith("."):
                continue
            if fname.endswith(".txt") or fname.endswith(".pdf"):
                if fname not in known:
                    path = os.path.join(KB_DIR, fname)
                    try:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        
                        text   = self._extract_text(data, fname)
                        chunks = self._chunk_text(text)
                        
                        self._index_text_chunks(chunks, source=fname, collection="policies")
                        
                        kbf = KBFile(
                            filename=fname,
                            collection="policies",
                            chunks=len(chunks),
                            size_bytes=len(data),
                            indexed=self._embeddings is not None,
                        )
                        self._files.append(kbf)
                        self._save_meta()
                    except Exception as exc:
                        st.warning(f"Failed to auto-load dropped file {fname}: {exc}")

    # Initialization (Private) 

    def _init_embeddings(self) -> None:
        """ Loads the local HuggingFace embedding model. """
        try:
            # Suppress noisy architectural warnings (e.g., position_ids UNEXPECTED)
            import logging
            from transformers import logging as transformers_logging
            transformers_logging.set_verbosity_error()
            
            # Check for HF_TOKEN to enable higher rate limits if provided
            hf_token = os.getenv("HF_TOKEN", "")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            st.warning(f"Embedding model unavailable ({exc}). Keyword fallback active.")
            self._embeddings = None

    def _init_reranker(self) -> None:
        """
        [Stage 2 - Retrieval] Loads the Cross-Encoder reranker.
        Uses sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2).
        """
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        except Exception as exc:
            import streamlit as _st
            _st.warning(f"Reranker unavailable ({exc}). Using RRF only.")
            self._reranker = None

    def _load_generalised_kb(self) -> None:
        """ Ensures the seed KB exists for both vector and keyword retrieval. """
        for item in GENERALISED_KB:
            if os.path.exists(self._fallback_path(item["name"])):
                continue
            self._index_text(
                text=item["content"],
                source=item["name"],
                collection=item["collection"],
            )

    def _reload_existing_stores(self) -> None:
        """ Re-connects to Milvus collections that already exist on disk. """
        if not self._embeddings:
            return
        for col in COLLECTIONS:
            store = self._try_connect_store(col)
            if store:
                self._stores[col] = store

    def _try_connect_store(self, collection: str) -> Optional[object]:
        """ Attempts a connection to a specific Milvus collection. """
        try:
            from langchain_milvus import Milvus
            # Milvus Lite on Windows prefers forward slashes or simple relative paths
            uri = MILVUS_DB.replace("\\", "/")
            
            store = Milvus(
                embedding_function=self._embeddings,
                connection_args={"uri": uri},
                collection_name=collection,
                drop_old=False,
            )
            return store
        except Exception:
            return None

    def _load_meta(self) -> None:
        """ Reads the JSON metadata file for known KB files. """
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH) as fh:
                    raw = json.load(fh)
                self._files = [KBFile(**r) for r in raw]
            except Exception:
                self._files = []

    def _save_meta(self) -> None:
        """ Persists the current state of known files to disk. """
        with open(META_PATH, "w") as fh:
            json.dump([asdict(f) for f in self._files], fh, indent=2)

    # State Accessors 

    @property
    def is_vector_enabled(self) -> bool:
        """ Returns True if semantic search is currently operational. """
        return self._embeddings is not None

    @property
    def files(self) -> list[KBFile]:
        return self._files

    @property
    def generalised_kb(self) -> list[dict]:
        return GENERALISED_KB

    @property
    def total_chunks(self) -> int:
        """ Returns the total number of knowledge vectors available for retrieval. """
        return (
            sum(f.chunks for f in self._files)
            + sum(g["chunks"] for g in GENERALISED_KB)
        )

    # Ingestion (Public) 

    def add_file(
        self,
        file_bytes: bytes,
        filename:   str,
        collection: str = "policies",
    ) -> KBFile:
        """
        Adds a new document to the knowledge base.
        Saves to disk and indexes into the appropriate vector collection.
        """
        dest = os.path.join(KB_DIR, filename)
        with open(dest, "wb") as fh:
            fh.write(file_bytes)

        text   = self._extract_text(file_bytes, filename)
        chunks = self._chunk_text(text)
        self._index_text_chunks(chunks, source=filename, collection=collection)

        kbf = KBFile(
            filename=filename,
            collection=collection,
            chunks=len(chunks),
            size_bytes=len(file_bytes),
            indexed=True,
        )
        # Update or add the file record.
        self._files = [f for f in self._files if f.filename != filename]
        self._files.append(kbf)
        self._save_meta()
        return kbf

    def remove_file(self, filename: str) -> None:
        """ Removes a file from metadata and physical storage. """
        self._files = [f for f in self._files if f.filename != filename]
        for ext in ("", ".chunks.txt"):
            p = os.path.join(KB_DIR, filename + ext)
            if os.path.exists(p):
                os.remove(p)
        self._save_meta()

    # Retrieval & Auditing (Public) 

    async def query(
        self,
        question:   str,
        top_k:      int  = TOP_K,
        collection: Optional[str] = None,
    ) -> list[RAGResult]:
        """
        Executes a semantic search query for relevant knowledge asynchronously.
        Defaults to searching across all known collections.
        """
        return await asyncio.to_thread(self._sync_query, question, top_k, collection)

    def _sync_query(
        self,
        question:   str,
        top_k:      int,
        collection: Optional[str],
    ) -> list[RAGResult]:
        """ Hybrid Query: Combines Vector Search (Milvus) and Keyword Search (BM25). """
        cols = [collection] if collection else COLLECTIONS
        all_vector_results: list[RAGResult] = []
        all_keyword_results: list[RAGResult] = []
        
        for col in cols:
            # 1. Vector Search
            store = self._stores.get(col)
            if store:
                all_vector_results.extend(self._milvus_query(store, question, col, top_k * 2))
            
            # 2. Keyword Search (BM25)
            all_keyword_results.extend(self._bm25_query(question, col, top_k * 2))

        # 3. RRF — merge vector + keyword into a broad candidate set
        candidates = self._fuse_results(all_vector_results, all_keyword_results, top_k * 2)

        # 4. Neural Reranking (Cross-Encoder) — precision pass over candidates
        return self._rerank_results(question, candidates, top_k)

    def _rerank_results(
        self,
        query: str,
        results: list[RAGResult],
        top_k: int,
    ) -> list[RAGResult]:
        """
        [Stage 2 — Retrieval] Cross-Encoder reranking.
        Each (query, passage) pair is scored by a bi-encoder-independent model
        that attends to both texts jointly, yielding far more accurate scores.
        Falls back to RRF order if the reranker is unavailable.
        """
        if not self._reranker or not results:
            return results[:top_k]
        try:
            import numpy as np
            pairs = [[query, r.text] for r in results]
            raw_scores = self._reranker.predict(pairs)
            for res, sc in zip(results, raw_scores):
                # Convert logit to probability-like score in [0, 1]
                res.score = float(1.0 / (1.0 + np.exp(-float(sc))))
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        except Exception as exc:
            import streamlit as _st
            _st.warning(f"Reranking error: {exc}")
            return results[:top_k]

    def _fuse_results(
        self, 
        vector_res: list[RAGResult], 
        keyword_res: list[RAGResult], 
        top_k: int,
        k: int = 60
    ) -> list[RAGResult]:
        """ 
        Standard Reciprocal Rank Fusion (RRF) to merge two ranked lists. 
        Formula: score = sum(1 / (k + rank))
        """
        scores: dict[str, float] = {}
        metadata: dict[str, RAGResult] = {}

        # Rank vector results
        for rank, res in enumerate(vector_res, 1):
            scores[res.text] = scores.get(res.text, 0) + 1.0 / (k + rank)
            metadata[res.text] = res

        # Rank keyword results
        for rank, res in enumerate(keyword_res, 1):
            scores[res.text] = scores.get(res.text, 0) + 1.0 / (k + rank)
            if res.text not in metadata:
                metadata[res.text] = res

        # Sort by fused score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for text, score in fused[:top_k]:
            res = metadata[text]
            # Normalize score for UI (heuristic)
            res.score = min(0.99, round(score * k, 3)) 
            final_results.append(res)
            
        return final_results

    async def get_live_suggestions(self, text: str) -> list[str]:
        """
        Retrieves real-time suggestions from the KB based on live transcript turns.
        """
        results = await self.query(text, top_k=2)
        return [r.text for r in results if r.score > 0.4]

    async def audit_chain(
        self,
        agent_statement: str,
        context_question: str,
    ) -> dict:
        """
        Determines the factual accuracy of an agent's statement asynchronously.
        """
        # Retrieve relevant policies from the Knowledge Base
        search_query = " ".join(
            part.strip() for part in (context_question, agent_statement) if part and part.strip()
        )
        chunks = await self.query(search_query)
        
        if not chunks:
            return {
                "answer":        "No relevant policy found in Knowledge Base.",
                "citations":     [],
                "groundedness":  0.0,
                "policy_breach": False,
                "top_source":    "Unknown",
                "top_score":     0.0,
            }

        groundedness = round(
            sum(r.score for r in chunks) / len(chunks), 3
        ) if chunks else 0.0
        citations = [r.to_citation() for r in chunks] if chunks else []
        context_text = "\n".join([r.text for r in chunks]) if chunks else ""

        # Factual integrity check logic.
        if not chunks or not chunks[0]:
            return {
                "answer":        "No relevant policy found in Knowledge Base.",
                "citations":     [],
                "groundedness":  0.0,
                "policy_breach": False,
                "top_source":    "Unknown",
                "top_score":     0.0,
            }
        
        top = chunks[0].text.lower() if chunks[0].text else ""
        stmt_lower = agent_statement.lower() if agent_statement else ""

        # Heuristic: Numeric discrepancy check (highly accurate for SLA/Price leaks).
        import re
        nums_policy = set(re.findall(r"\d+", top)) if top else set()
        nums_agent  = set(re.findall(r"\d+", stmt_lower)) if stmt_lower else set()
        policy_breach = bool(
            nums_agent and nums_policy and
            not nums_agent.intersection(nums_policy) and
            len(nums_policy) > 0
        )

        return {
            "answer":        context_text[:800],
            "citations":     citations,
            "groundedness":  groundedness,
            "policy_breach": policy_breach,
            "top_source":    chunks[0].source,
            "top_score":     chunks[0].score,
        }

    # Indexing Implementation (Private) 

    def _index_text(self, text: str, source: str, collection: str) -> None:
        """ Chunks and indexes a raw string of text. """
        chunks = self._chunk_text(text)
        self._index_text_chunks(chunks, source=source, collection=collection)

    def _index_text_chunks(
        self,
        chunks:     list[str],
        source:     str,
        collection: str,
    ) -> None:
        """ 
        Writes chunks to the vector store and a local plain-text fallback.
        """
        if not chunks:
            return

        # Write text-based backup for keyword search fallback.
        fallback_path = self._fallback_path(source)
        with open(fallback_path, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(c + "\n---CHUNK---\n")

        if not self._embeddings:
            return

        try:
            from langchain_milvus import Milvus
            from langchain_core.documents import Document

            docs = [
                Document(
                    page_content=c,
                    metadata={"source": source, "collection": collection},
                )
                for c in chunks
            ]

            # Milvus Lite on Windows prefers forward slashes or simple relative paths
            uri = MILVUS_DB.replace("\\", "/")
            
            # Upsert into existing or new collection.
            if collection in self._stores and self._stores[collection] is not None:
                self._stores[collection].add_documents(docs)
            else:
                store = Milvus.from_documents(
                    docs,
                    self._embeddings,
                    connection_args={"uri": uri},
                    collection_name=collection,
                    drop_old=False,
                )
                self._stores[collection] = store
        except Exception as exc:
            st.warning(f"Milvus indexing error ({exc}). Keyword fallback used.")

    # Retrieval Implementation (Private) 

    def _query_collection(
        self,
        question:   str,
        collection: str,
        top_k:      int,
    ) -> list[RAGResult]:
        """ Tiered query: Vector search if available, else Keyword overlap. """
        store = self._stores.get(collection)
        if store is not None:
            return self._milvus_query(store, question, collection, top_k)
        return self._keyword_query(question, collection, top_k)

    @staticmethod
    def _milvus_query(
        store:      object,
        question:   str,
        collection: str,
        top_k:      int,
    ) -> list[RAGResult]:
        """ Performs semantic search via Milvus. """
        try:
            # MMR (Maximal Marginal Relevance) maximizes information gain while minimizing redundancy.
            docs = store.max_marginal_relevance_search(
                question, k=top_k, fetch_k=top_k * 3, lambda_mult=0.6
            )
            # Fallback to pure similarity if MMR returns nothing.
            if not docs:
                docs_scores = store.similarity_search_with_score(question, k=top_k)
                return [
                    RAGResult(
                        text=d.page_content,
                        source=d.metadata.get("source", "KB"),
                        collection=collection,
                        score=round(float(1 - s), 3),
                    )
                    for d, s in docs_scores
                ]
            return [
                RAGResult(
                    text=d.page_content,
                    source=d.metadata.get("source", "KB"),
                    collection=collection,
                    score=0.85, # MMR doesn't provide easy raw scores at this level.
                )
                for d in docs
            ]
        except Exception:
            return []

    def _bm25_query(
        self,
        question:   str,
        collection: str,
        top_k:      int,
    ) -> list[RAGResult]:
        """ 
        Modern keyword search using BM25 algorithm.
        Dynamically builds the index from local chunk backups.
        """
        try:
            from rank_bm25 import BM25Okapi
            source_map = self._source_collection_map()
            
            # 1. Collect all chunks for this collection
            corpus_chunks: list[tuple[str, str]] = [] # (text, source)
            for fname in os.listdir(KB_DIR):
                if not fname.endswith(".chunks.txt"):
                    continue
                source = fname.replace(".chunks.txt", "")
                if source_map.get(source) != collection:
                    continue
                try:
                    with open(os.path.join(KB_DIR, fname), encoding="utf-8") as fh:
                        content = fh.read()
                    for chunk in content.split("---CHUNK---\n"):
                        chunk = chunk.strip()
                        if chunk:
                            corpus_chunks.append((chunk, source))
                except Exception:
                    continue

            if not corpus_chunks:
                return []

            # 2. Tokenize and index
            tokenized_corpus = [self._tokenize(c[0]) for c in corpus_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 3. Query
            tokenized_query = self._tokenize(question)
            doc_scores = bm25.get_scores(tokenized_query)
            
            # 4. Format results
            indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            if not indices:
                return []
            if doc_scores[indices[0]] <= 0:
                return self._keyword_query(question, collection, top_k)

            results = []
            max_score = doc_scores[indices[0]]
            
            for i in indices[:top_k]:
                if doc_scores[i] <= 0: break
                text, source = corpus_chunks[i]
                results.append(RAGResult(
                    text=text,
                    source=source,
                    collection=collection,
                    score=round(doc_scores[i] / max_score, 3)
                ))
            return results
        except Exception as exc:
            st.warning(f"BM25 Error: {exc}")
            return self._keyword_query(question, collection, top_k)

    def _keyword_query(
        self,
        question:   str,
        collection: str,
        top_k:      int,
    ) -> list[RAGResult]:
        """ 
        Legacy keyword overlap search. 
        Used when vector DB is unavailable or for simple exact-match lookups.
        """
        keywords = set(self._tokenize(question))
        results: list[tuple[int, str, str]] = []
        source_map = self._source_collection_map()

        for fname in os.listdir(KB_DIR):
            if not fname.endswith(".chunks.txt"):
                continue
            source = fname.replace(".chunks.txt", "")
            if source_map.get(source) != collection:
                continue
            try:
                with open(os.path.join(KB_DIR, fname), encoding="utf-8") as fh:
                    raw = fh.read()
                for chunk in raw.split("---CHUNK---\n"):
                    chunk = chunk.strip()
                    if not chunk: continue
                    overlap = len(keywords & set(self._tokenize(chunk)))
                    if overlap > 0:
                        results.append((overlap, chunk, source))
            except Exception:
                continue

        results.sort(key=lambda x: x[0], reverse=True)
        mx = results[0][0] if results else 1
        return [
            RAGResult(
                text=text,
                source=source,
                collection=collection,
                score=round(overlap / mx, 3),
            )
            for overlap, text, source in results[:top_k]
        ]

    # Parsing & Chunking (Private) 

    @staticmethod
    def _extract_text(data: bytes, filename: str) -> str:
        """ Extracts raw text from PDF or TXT bytes. """
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                pass
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """ Splits text into smaller, overlapping chunks for optimal retrieval. """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            # Improved separators for better semantic boundaries
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
                add_start_index=True,
            )
            return [c for c in splitter.split_text(text) if c.strip()]
        except Exception:
            # Plan B: sentence-aware chunking if possible
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            curr = ""
            for s in sentences:
                if len(curr) + len(s) < CHUNK_SIZE:
                    curr += " " + s
                else:
                    chunks.append(curr.strip())
                    curr = s
            if curr: chunks.append(curr.strip())
            return [c for c in chunks if c]
