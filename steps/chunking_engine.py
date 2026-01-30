import nltk
import hashlib
import re
from typing import Dict, List, Any
from transformers import AutoTokenizer
from zenml import step
import mlflow
nltk.download("punkt_tab")

class RecursiveChunker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_tokens: int = 500,
        overlap: int = 80,
        min_chunk_size: int = 420,
        respect_sentence_boundaries: bool = True,
        respect_paragraph_boundaries: bool = True,
        preserve_lists: bool = True
    ):
        mlflow.log_param("model_name_used_in_chunking", model_name)
        mlflow.log_param("max_tokens", max_tokens)
        mlflow.log_param("overlap", overlap)
        mlflow.log_param("min_chunk_size", min_chunk_size)
        mlflow.log_param("respect_sentence_boundaries", respect_sentence_boundaries)
        mlflow.log_param("respect_paragraph_boundaries", respect_paragraph_boundaries)
        mlflow.log_param("preserve_lists", preserve_lists)
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        self.preserve_lists = preserve_lists

    # ---------------------------------------------
    # Tokenization helpers
    # ---------------------------------------------
    def token_len(self, text: str, cache: Dict[str,int] = None) -> int:
        """Get token count for text (with optional caching)"""
        if cache is not None and text in cache:
            return cache[text]
        count = len(self.tokenizer.encode(text, truncation=False, add_special_tokens=False))
        if cache is not None:
            cache[text] = count
        return count

    def split_sentences(self, text: str) -> List[str]:
        try:
            sentences = nltk.sent_tokenize(text)
            cleaned = [s.strip() for s in sentences if s.strip() and len(s.strip())>10]
            return cleaned
        except:
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    # ---------------------------------------------
    # Preserve lists/tables
    # ---------------------------------------------
    def _preserve_lists(self, text: str) -> str:
        lines = text.split('\n')
        preserved = []
        current_list = []
        in_list = False

        list_pattern = re.compile(r'^(\s*[-*•+]|\d+\.)\s+')
        table_pattern = re.compile(r'\|')

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if list_pattern.match(line_stripped):
                if current_list and not in_list:
                    preserved.append("\n".join(current_list))
                    current_list = []
                in_list = True
                current_list.append(line)
            elif in_list and line_stripped == '':
                current_list.append(line)
            elif in_list and not list_pattern.match(line_stripped):
                preserved.append("\n".join(current_list))
                current_list = [line]
                in_list = False
            else:
                current_list.append(line)

        if current_list:
            preserved.append("\n".join(current_list))

        # Tables
        final_lines = []
        i = 0
        while i < len(preserved):
            line = preserved[i]
            if table_pattern.search(line):
                table_rows = [line]
                i += 1
                while i < len(preserved) and table_pattern.search(preserved[i]):
                    table_rows.append(preserved[i])
                    i += 1
                final_lines.append("\n".join(table_rows))
            else:
                final_lines.append(line)
                i += 1

        return "\n".join(final_lines)

    # ---------------------------------------------
    # Paragraph -> chunks
    # ---------------------------------------------
    def _split_paragraph_into_chunks(self, paragraph: str, metadata: Dict[str,Any], token_cache=None) -> List[Dict[str,Any]]:
        chunks = []

        if self.token_len(paragraph, token_cache) <= self.max_tokens:
            return [{"chunk_id": self._make_chunk_id(paragraph, metadata),
                     "text": paragraph, "metadata": metadata}]

        if self.respect_sentence_boundaries:
            sentences = self.split_sentences(paragraph)
        else:
            sentences = paragraph.split()  # fallback word split

        current_chunk, current_tokens = [], 0

        for sent in sentences:
            sent_tokens = self.token_len(sent, token_cache)
            if (current_tokens + sent_tokens > self.max_tokens) and (current_tokens >= self.min_chunk_size):
                chunk_text = " ".join(current_chunk)
                chunks.append({"chunk_id": self._make_chunk_id(chunk_text, metadata),
                               "text": chunk_text, "metadata": metadata})
                current_chunk, current_tokens = [], 0
            current_chunk.append(sent)
            current_tokens += sent_tokens

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({"chunk_id": self._make_chunk_id(chunk_text, metadata),
                           "text": chunk_text, "metadata": metadata})

        return chunks

    # ---------------------------------------------
    # Merge small chunks (backward + forward)
    # ---------------------------------------------
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], token_cache=None) -> List[Dict[str, Any]]:
        if len(chunks) <= 1: return chunks
        merged = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            current_tokens = self.token_len(current["text"], token_cache)
            if current_tokens < self.min_chunk_size:
                if i+1 < len(chunks):
                    next_chunk = chunks[i+1]
                    combined_text = current["text"] + " " + next_chunk["text"]
                    if self.token_len(combined_text, token_cache) <= self.max_tokens * 1.5:
                        merged.append({"chunk_id": self._make_chunk_id(combined_text, current["metadata"]),
                                       "text": combined_text, "metadata": current["metadata"]})
                        i += 2
                        continue
                if merged:
                    merged[-1]["text"] += " " + current["text"]
                else:
                    merged.append(current)
            else:
                merged.append(current)
            i += 1
        return merged

    # ---------------------------------------------
    # Add smart overlap
    # ---------------------------------------------
    def _add_smart_overlap(self, chunks: List[Dict[str,Any]], token_cache=None) -> List[Dict[str,Any]]:
        overlapped = []
        for i in range(len(chunks)):
            curr_text = chunks[i]["text"]
            curr_meta = chunks[i]["metadata"]
            overlap_text = ""

            if i>0:
                prev_text = chunks[i-1]["text"]
                back_tokens = self.token_len(prev_text, token_cache)
                overlap_tokens = min(self.overlap, back_tokens)
                overlap_text += self.tokenizer.decode(self.tokenizer.encode(prev_text, add_special_tokens=False)[-overlap_tokens:], skip_special_tokens=True)

            if i<len(chunks)-1:
                next_text = chunks[i+1]["text"]
                next_tokens = self.token_len(next_text, token_cache)
                overlap_tokens = min(self.overlap, next_tokens)
                overlap_text += " " + self.tokenizer.decode(self.tokenizer.encode(next_text, add_special_tokens=False)[:overlap_tokens], skip_special_tokens=True)

            final_text = overlap_text + " " + curr_text if overlap_text else curr_text
            overlapped.append({"chunk_id": self._make_chunk_id(final_text, curr_meta),
                               "text": final_text, "metadata": curr_meta})
        return overlapped

    # ---------------------------------------------
    # Main chunking
    # ---------------------------------------------
    def chunk_text(self, docs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        all_chunks = []
        token_cache = {}
        for doc in docs:
            text = doc.get("text","")
            meta = doc.get("metadata",{})
            if not text.strip(): continue

            if self.preserve_lists:
                text = self._preserve_lists(text)

            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()] if self.respect_paragraph_boundaries else [text]
            doc_chunks = []
            for para in paragraphs:
                para_chunks = self._split_paragraph_into_chunks(para, meta, token_cache)
                doc_chunks.extend(para_chunks)

            doc_chunks = self._merge_small_chunks(doc_chunks, token_cache)
            if self.overlap>0:
                doc_chunks = self._add_smart_overlap(doc_chunks, token_cache)

            all_chunks.extend(doc_chunks)
        return all_chunks

    # ---------------------------------------------
    # Chunk ID
    # ---------------------------------------------
    def _make_chunk_id(self, text:str, metadata:Dict[str,Any])->str:
        source = metadata.get("source","")
        base = f"{text[:100]}_{source}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()


@step(enable_cache=True)
def chunk_documents(docs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """Chunk documents using RecursiveChunker."""
    chunker = RecursiveChunker()
    return chunker.chunk_text(docs)

