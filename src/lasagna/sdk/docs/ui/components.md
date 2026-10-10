# Lasagna — UI Components Reference

Copy-paste-ready TypeScript types, API client, React hooks, and components for building frontends on top of Lasagna. Every example is self-contained and written for standalone use.

**Prerequisites:** React 18+, TypeScript. No other required dependencies (icons use emoji; add `lucide-react` for better icons). Add [`marked`](https://github.com/markedjs/marked) if your LLM responses contain markdown formatting.

---

## Table of Contents

- [Part 1: Foundation (Framework-Agnostic)](#part-1-foundation)
  - [1.1 TypeScript Types](#11-typescript-types)
  - [1.2 API Client](#12-api-client)
  - [1.3 Citation Parser](#13-citation-parser)
  - [1.4 Chat History Persistence](#14-chat-history-persistence)
- [Part 2: React Hooks](#part-2-react-hooks)
  - [2.1 useKnowledge — Documents & Upload](#21-useknowledge)
  - [2.2 useCitations — Citation State](#22-usecitations)
  - [2.3 useDocumentPicker — "/" Command](#23-usedocumentpicker)
  - [2.4 useRAGConfig — Config & Profiles](#24-useragconfig)
  - [2.5 useChatHistory — Per-Message Persistence](#25-usechathistory)
- [Part 3: React Components](#part-3-react-components)
  - [3.1 DocumentUploadPanel](#31-documentuploadpanel)
  - [3.2 SearchResults](#32-searchresults)
  - [3.3 CitationBadge + Preview](#33-citationbadge--preview)
  - [3.4 MessageWithCitations](#34-messagewithcitations)
  - [3.5 SourcesSidebar](#35-sourcessidebar)
  - [3.6 DocumentPicker](#36-documentpicker)
  - [3.7 ProfileSwitcher](#37-profileswitcher)
  - [3.8 RAGConfigPanel](#38-ragconfigpanel)
- [Part 4: Full Integration Example](#part-4-full-integration-example)

---

## Part 1: Foundation

### 1.1 TypeScript Types

All types mirror the REST API contracts from `api/rest/models.py`. Use these as the single source of truth for your frontend.

```typescript
// types/knowledge.ts

// ── Document Status ──────────────────────────────────────────────────
export enum DocumentStatus {
  UPLOADING   = "uploading",
  PROCESSING  = "processing",
  INDEXING    = "indexing",
  READY       = "ready",
  ERROR       = "error",
}

// ── Constants ────────────────────────────────────────────────────────
export const SUPPORTED_EXTENSIONS = [
  ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".pdf", ".docx",
] as const;

export const MAX_FILE_SIZE_MB = 50;

// ── REST API Response Types ──────────────────────────────────────────
// These match the Pydantic models in api/rest/models.py exactly.

export interface DocumentResponse {
  document_id: string;
  session_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  chunks_count: number;
  upload_date: string;
  error_message?: string;
}

export interface ChunkResponse {
  chunk_id: string;
  content: string;
  document_id: string;
  chunk_index: number;
  page_number?: number;
  metadata: Record<string, unknown>;
}

export interface SearchResultResponse {
  chunk: ChunkResponse;
  score: number;
  document_id: string;
  filename: string;
}

export interface RoutingInfo {
  tier: string;
  strategy: string;
  confidence: number;
  rationale: string;
  query_type?: string;
  estimated_complexity?: string;
}

export interface SearchResponse {
  results: SearchResultResponse[];
  query: string;
  total_results: number;
  routing?: RoutingInfo;
}

export interface CitationResponse {
  citation_id: string;
  document_id: string;
  document_name: string;
  chunk_id: string;
  chunk_index: number;
  page_number?: number;
  score: number;
  content: string;
  highlight?: string;
  highlights?: string[];
  highlight_ranges?: number[][];
}

export interface SearchWithCitationsResponse {
  results: SearchResultResponse[];
  citations: CitationResponse[];
  context: string;
  query: string;
  total_results: number;
  routing?: RoutingInfo;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: string;
  chunks_count: number;
}

export interface DeleteResponse {
  success: boolean;
  document_id: string;
  message: string;
}

export interface SessionResponse {
  token: string;
  session_id: string;
  user_id: string;
  expires_at: string;
}

export interface ConfigResponse {
  // Core
  vector_backend: string;
  embedding_model: string;
  embedding_dim: number;
  chunking_strategy: string;
  // Feature toggles
  enable_hybrid_search: boolean;
  enable_reranking: boolean;
  enable_mmr: boolean;
  enable_hyde: boolean;
  enable_hierarchical: boolean;
  enable_context_reordering: boolean;
  enable_contextual_retrieval: boolean;
  enable_rag_fusion: boolean;
  enable_crag: boolean;
  enable_query_classification: boolean;
  enable_query_routing: boolean;
  // Search parameters
  default_top_k: number;
  rerank_top_k: number;
  similarity_threshold: number;
  hybrid_alpha: number;
  mmr_lambda: number;
  // Routing
  routing_strategy: string;
  routing_default_tier: string;
  routing_confidence_threshold: number;
  // Agentic / Quality
  use_agentic_rag: boolean;
  use_structured_graph: boolean;
  agentic_pipeline: string;
  enable_hallucination_check: boolean;
  enable_answer_quality_check: boolean;
  max_quality_check_attempts: number;
  coverage_scorer_mode: string;
  final_search_mode: string;
  // Chunking
  chunk_size: number;
  chunk_overlap: number;
  semantic_threshold: number;
  min_chunk_size: number;
  max_chunk_size: number;
  parent_chunk_size: number;
  child_chunk_size: number;
  auto_merge_threshold: number;
  preserve_tables: boolean;
  preserve_headers: boolean;
  // OCR & Docling
  enable_ocr: boolean;
  ocr_engine: string;
  docling_enabled: boolean;
  docling_mode: string;
  docling_ocr_enabled: boolean;
  docling_table_structure: boolean;
  docling_max_tokens: number;
  // Infrastructure
  enable_prometheus: boolean;
  enable_tracing: boolean;
}

export interface ProfileInfo {
  name: string;
  description: string;
}

export interface ProfileListResponse {
  profiles: ProfileInfo[];
  active?: string;
}

// ── Frontend Display Types ───────────────────────────────────────────
// Convenience types for UI state (camelCase, with derived fields).

export interface KnowledgeDocument {
  id: string;
  filename: string;
  fileType: string;
  sizeBytes: number;
  status: DocumentStatus;
  chunkCount: number;
  uploadedAt: Date;
  errorMessage?: string;
}

/** Convert REST response → frontend document */
export function toKnowledgeDocument(doc: DocumentResponse): KnowledgeDocument {
  return {
    id: doc.document_id,
    filename: doc.filename,
    fileType: doc.file_type,
    sizeBytes: doc.file_size,
    status: doc.status as DocumentStatus,
    chunkCount: doc.chunks_count,
    uploadedAt: new Date(doc.upload_date),
    errorMessage: doc.error_message,
  };
}
```

---

### 1.2 API Client

Framework-agnostic `fetch` wrapper for all REST endpoints. Works in any JavaScript runtime.

```typescript
// services/knowledgeApi.ts

import type {
  DocumentResponse, SearchResponse, SearchWithCitationsResponse,
  UploadResponse, DeleteResponse, SessionResponse, ConfigResponse,
  ProfileListResponse, CitationResponse, ChunkResponse,
} from "../types/knowledge";

export interface SearchOptions {
  top_k?: number;
  min_score?: number;
  document_ids?: string[];
  use_hyde?: boolean;
  use_reranking?: boolean;
  routing_tier?: "fast" | "standard" | "comprehensive";
}

export class KnowledgeAPI {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  // ── Session ──────────────────────────────────────────────────────
  async createSession(sessionId?: string, userId?: string): Promise<SessionResponse> {
    const res = await this.post("/api/v1/sessions", {
      session_id: sessionId,
      user_id: userId,
    });
    const data: SessionResponse = await res.json();
    this.token = data.token;
    return data;
  }

  // ── Documents ────────────────────────────────────────────────────
  async uploadDocument(file: File, documentId?: string): Promise<UploadResponse> {
    const form = new FormData();
    form.append("file", file);
    if (documentId) form.append("document_id", documentId);
    const res = await this.fetch("/api/v1/documents", { method: "POST", body: form });
    return res.json();
  }

  async listDocuments(): Promise<{ documents: DocumentResponse[]; total: number }> {
    const res = await this.fetch("/api/v1/documents");
    return res.json();
  }

  async getDocument(id: string): Promise<DocumentResponse> {
    const res = await this.fetch(`/api/v1/documents/${id}`);
    return res.json();
  }

  async getDocumentStatus(id: string): Promise<{ document_id: string; status: string; chunks_count: number; error_message?: string }> {
    const res = await this.fetch(`/api/v1/documents/${id}/status`);
    return res.json();
  }

  async getDocumentChunks(id: string, offset = 0, limit = 50): Promise<ChunkResponse[]> {
    const res = await this.fetch(`/api/v1/documents/${id}/chunks?offset=${offset}&limit=${limit}`);
    return res.json();
  }

  async deleteDocument(id: string): Promise<DeleteResponse> {
    const res = await this.fetch(`/api/v1/documents/${id}`, { method: "DELETE" });
    return res.json();
  }

  async reindexDocument(id: string): Promise<UploadResponse> {
    const res = await this.fetch(`/api/v1/documents/${id}/reindex`, { method: "POST" });
    return res.json();
  }

  // ── Search ───────────────────────────────────────────────────────
  async search(query: string, options?: SearchOptions): Promise<SearchResponse> {
    const res = await this.post("/api/v1/search", { query, ...options });
    return res.json();
  }

  async searchWithCitations(query: string, options?: SearchOptions): Promise<SearchWithCitationsResponse> {
    const res = await this.post("/api/v1/search/citations", { query, ...options });
    return res.json();
  }

  async searchAgentic(query: string, options?: SearchOptions): Promise<SearchWithCitationsResponse> {
    const res = await this.post("/api/v1/search/agentic", { query, ...options });
    return res.json();
  }

  /** Get pre-formatted context string for LLM injection */
  async getContext(query: string, options?: SearchOptions): Promise<string> {
    const result = await this.searchWithCitations(query, options);
    return result.context;
  }

  // ── Citations ────────────────────────────────────────────────────
  async listCitations(): Promise<CitationResponse[]> {
    const res = await this.fetch("/api/v1/citations");
    const data: { citations: CitationResponse[]; session_id: string; total: number } = await res.json();
    return data.citations;
  }

  async getCitation(id: string): Promise<CitationResponse> {
    const res = await this.fetch(`/api/v1/citations/${id}`);
    return res.json();
  }

  // ── Config ───────────────────────────────────────────────────────
  async getConfig(): Promise<ConfigResponse> {
    const res = await this.fetch("/api/v1/config");
    return res.json();
  }

  async patchConfig(updates: Partial<ConfigResponse>): Promise<ConfigResponse> {
    const res = await this.fetch("/api/v1/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
    return res.json();
  }

  async listProfiles(): Promise<ProfileListResponse> {
    const res = await this.fetch("/api/v1/config/profiles");
    return res.json();
  }

  async applyProfile(name: string): Promise<ConfigResponse> {
    const res = await this.fetch(`/api/v1/config/profiles/${name}`, { method: "PUT" });
    return res.json();
  }

  // ── Internals ────────────────────────────────────────────────────
  private async fetch(path: string, init?: RequestInit): Promise<Response> {
    const headers: Record<string, string> = { ...(init?.headers as Record<string, string>) };
    if (this.token) headers["X-Session-Token"] = this.token;

    const res = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new KnowledgeAPIError(res.status, body.detail || res.statusText, path);
    }
    return res;
  }

  private async post(path: string, body: unknown): Promise<Response> {
    return this.fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }
}

export class KnowledgeAPIError extends Error {
  constructor(public status: number, message: string, public path: string) {
    super(`[${status}] ${message} (${path})`);
    this.name = "KnowledgeAPIError";
  }
}
```

**Usage:**

```typescript
const api = new KnowledgeAPI("http://localhost:8000");
await api.createSession(); // auto-generates session_id and user_id

// Upload
const result = await api.uploadDocument(fileInput.files[0]);

// Search with citations
const { results, citations, context } = await api.searchWithCitations("quarterly revenue", {
  top_k: 5,
  routing_tier: "standard",
});

// Get context for LLM injection
const llmContext = await api.getContext("key findings");
const prompt = `Based on this context:\n${llmContext}\n\nAnswer: What were the key findings?`;
```

---

### 1.3 Citation Parser

Parse `[1]`, `[2]` markers from LLM responses. Groups consecutive citations like `[1][2]` into `[1,2]`. Framework-agnostic — works with any rendering library.

```typescript
// utils/parseCitations.ts

export interface TextSegment {
  type: "text" | "citation" | "invalid-citation";
  content: string;
  /** 0-based indices into citations array */
  citationIndices?: number[];
  /** 1-based indices that were out of range */
  invalidIndices?: number[];
}

/**
 * Parse inline citation markers [1], [2] from text.
 * Groups consecutive citations separated only by whitespace/commas.
 *
 * Example: "Revenue grew [1] sharply [2][3]"
 *   → [{type:"text", content:"Revenue grew "},
 *      {type:"citation", content:"[1]", citationIndices:[0]},
 *      {type:"text", content:" sharply "},
 *      {type:"citation", content:"[2,3]", citationIndices:[1,2]}]
 */
export function parseInlineCitations(text: string): TextSegment[] {
  const segments: TextSegment[] = [];
  const citationPattern = /\[(\d+)\]/g;

  // Find all citation positions
  const citations: { index: number; endIndex: number; number: number }[] = [];
  let match;
  while ((match = citationPattern.exec(text)) !== null) {
    citations.push({
      index: match.index,
      endIndex: match.index + match[0].length,
      number: parseInt(match[1], 10),
    });
  }

  if (citations.length === 0) {
    return [{ type: "text", content: text }];
  }

  // Group consecutive citations
  const groups: { startIndex: number; endIndex: number; numbers: number[] }[] = [];
  let currentGroup: typeof groups[0] | null = null;

  for (const citation of citations) {
    if (!currentGroup) {
      currentGroup = { startIndex: citation.index, endIndex: citation.endIndex, numbers: [citation.number] };
    } else {
      const between = text.substring(currentGroup.endIndex, citation.index);
      if (/^[\s,]*$/.test(between)) {
        currentGroup.endIndex = citation.endIndex;
        currentGroup.numbers.push(citation.number);
      } else {
        groups.push(currentGroup);
        currentGroup = { startIndex: citation.index, endIndex: citation.endIndex, numbers: [citation.number] };
      }
    }
  }
  if (currentGroup) groups.push(currentGroup);

  // Build segments
  let lastIndex = 0;
  for (const group of groups) {
    if (group.startIndex > lastIndex) {
      segments.push({ type: "text", content: text.substring(lastIndex, group.startIndex) });
    }
    segments.push({
      type: "citation",
      content: `[${group.numbers.join(",")}]`,
      citationIndices: group.numbers.map((n) => n - 1),
    });
    lastIndex = group.endIndex;
  }
  if (lastIndex < text.length) {
    segments.push({ type: "text", content: text.substring(lastIndex) });
  }

  return segments;
}

/** Quick check if text contains any citation markers */
export function hasCitations(text: string): boolean {
  return /\[\d+\]/.test(text);
}

/** Validate all citation indices are within range */
export function validateCitationRange(
  text: string,
  maxValidIndex: number
): { valid: boolean; outOfRange: number[] } {
  const outOfRange: number[] = [];
  const matches = text.matchAll(/\[(\d+)\]/g);
  for (const m of matches) {
    const n = parseInt(m[1], 10);
    if (n < 1 || n > maxValidIndex) outOfRange.push(n);
  }
  return { valid: outOfRange.length === 0, outOfRange: [...new Set(outOfRange)].sort((a, b) => a - b) };
}
```

---

### 1.4 Chat History Persistence

Persist each message (answer HTML + citations + chunks) to `sessionStorage` so citations survive page refreshes. Each message owns its own citation set — essential for multi-turn conversations.

> **Scope:** `sessionStorage` clears when the browser tab closes. The `sessionId` itself lives in `localStorage` (persistent), so the same session ID comes back on reopen but the chat history is empty. To survive browser restarts, swap every `sessionStorage` call below for `localStorage`.

```typescript
// utils/chatHistory.ts

import type { CitationResponse } from "../types/knowledge";

const STORAGE_PREFIX = "lasagna_chat_history_";
export const MAX_MESSAGES = 50;

export interface PersistedMessage {
  id: string;
  userQuery: string;
  /** Answer pre-rendered as HTML. Call marked.parse(answerMarkdown) before storing. */
  answerHtml: string;
  citations: CitationResponse[];
  chunks: Array<{ chunk_id: string; content: string; chunk_index?: number }>;
  timestamp: number;
}

function storageKey(sessionId: string): string {
  return `${STORAGE_PREFIX}${sessionId}`;
}

/** Append a completed message. On QuotaExceededError, logs a warning and skips persistence
 *  (the message is still visible in the current session via React state, but won't survive refresh). */
export function saveMessage(sessionId: string, message: PersistedMessage): void {
  try {
    const key = storageKey(sessionId);
    const existing = getMessages(sessionId);
    existing.push(message);
    sessionStorage.setItem(key, JSON.stringify(existing.slice(-MAX_MESSAGES)));
  } catch (e) {
    console.warn("[chatHistory] Failed to save:", e);
  }
}

/** Load all messages for the session, newest last. */
export function getMessages(sessionId: string): PersistedMessage[] {
  try {
    const raw = sessionStorage.getItem(storageKey(sessionId));
    return raw ? (JSON.parse(raw) as PersistedMessage[]) : [];
  } catch {
    return [];
  }
}

/** Wipe history (e.g., on conversation restart). */
export function clearMessages(sessionId: string): void {
  sessionStorage.removeItem(storageKey(sessionId));
}
```

**Usage:**

```typescript
import { marked } from "marked";
import { saveMessage, getMessages, clearMessages } from "./utils/chatHistory";

// After a completed LLM turn (convert markdown once, reuse for display and persistence):
// marked.parse() is the explicit synchronous API — avoids the string|Promise<string> ambiguity of marked()
const answerHtml = marked.parse(llmMarkdownReply) as string;
saveMessage(sessionId, {
  id: `msg_${Date.now()}`,
  userQuery: "What were the key findings?",
  answerHtml,
  citations: result.citations,
  // result.results[i].chunk — SearchWithCitationsResponse has no top-level chunks field
  chunks: result.results.map((r) => r.chunk),
  timestamp: Date.now(),
});

// On page mount — restore previous conversation:
const previous = getMessages(sessionId);

// On restart — wipe the slate:
clearMessages(sessionId);
```

---

## Part 2: React Hooks

### 2.1 useKnowledge

Manages document state with optimistic UI updates. Documents appear immediately as "uploading" before the server confirms.

```tsx
// hooks/useKnowledge.ts

import { useReducer, useCallback, useEffect } from "react";
import {
  KnowledgeDocument, DocumentStatus, toKnowledgeDocument,
} from "../types/knowledge";
import { KnowledgeAPI } from "../services/knowledgeApi";

interface KnowledgeState {
  documents: KnowledgeDocument[];
  isLoading: boolean;
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
}

type Action =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_DOCUMENTS"; payload: KnowledgeDocument[] }
  | { type: "ADD_PENDING"; payload: { filename: string; size: number }[] }
  | { type: "UPDATE_STATUS"; payload: { filename: string; status: DocumentStatus; doc?: KnowledgeDocument } }
  | { type: "REMOVE"; payload: string }
  | { type: "SET_UPLOADING"; payload: boolean }
  | { type: "SET_PROGRESS"; payload: number }
  | { type: "SET_ERROR"; payload: string | null };

function reducer(state: KnowledgeState, action: Action): KnowledgeState {
  switch (action.type) {
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_DOCUMENTS":
      return { ...state, documents: action.payload, isLoading: false };
    case "ADD_PENDING":
      return {
        ...state,
        documents: [
          ...state.documents,
          ...action.payload.map((p) => ({
            id: `pending_${p.filename}_${Date.now()}`,
            filename: p.filename,
            fileType: p.filename.split(".").pop() || "",
            sizeBytes: p.size,
            status: DocumentStatus.UPLOADING,
            chunkCount: 0,
            uploadedAt: new Date(),
          })),
        ],
      };
    case "UPDATE_STATUS":
      return {
        ...state,
        documents: state.documents.map((d) =>
          d.filename === action.payload.filename
            ? action.payload.doc ?? { ...d, status: action.payload.status }
            : d
        ),
      };
    case "REMOVE":
      return { ...state, documents: state.documents.filter((d) => d.id !== action.payload) };
    case "SET_UPLOADING":
      return { ...state, isUploading: action.payload };
    case "SET_PROGRESS":
      return { ...state, uploadProgress: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload, isLoading: false, isUploading: false };
    default:
      return state;
  }
}

export function useKnowledge(api: KnowledgeAPI) {
  const [state, dispatch] = useReducer(reducer, {
    documents: [],
    isLoading: false,
    isUploading: false,
    uploadProgress: 0,
    error: null,
  });

  const refresh = useCallback(async () => {
    dispatch({ type: "SET_LOADING", payload: true });
    try {
      const { documents } = await api.listDocuments();
      dispatch({ type: "SET_DOCUMENTS", payload: documents.map(toKnowledgeDocument) });
    } catch (e) {
      dispatch({ type: "SET_ERROR", payload: e instanceof Error ? e.message : "Failed to load" });
    }
  }, [api]);

  const uploadFiles = useCallback(async (files: File[]) => {
    // Optimistic: show files immediately as "uploading"
    dispatch({
      type: "ADD_PENDING",
      payload: files.map((f) => ({ filename: f.name, size: f.size })),
    });
    dispatch({ type: "SET_UPLOADING", payload: true });
    dispatch({ type: "SET_PROGRESS", payload: 0 });

    let completed = 0;
    for (const file of files) {
      try {
        dispatch({ type: "UPDATE_STATUS", payload: { filename: file.name, status: DocumentStatus.PROCESSING } });
        const result = await api.uploadDocument(file);
        dispatch({
          type: "UPDATE_STATUS",
          payload: {
            filename: file.name,
            status: DocumentStatus.READY,
            doc: toKnowledgeDocument({ ...result, session_id: "", upload_date: new Date().toISOString() } as any),
          },
        });
      } catch (e) {
        dispatch({ type: "UPDATE_STATUS", payload: { filename: file.name, status: DocumentStatus.ERROR } });
      }
      completed++;
      dispatch({ type: "SET_PROGRESS", payload: Math.round((completed / files.length) * 100) });
    }

    dispatch({ type: "SET_UPLOADING", payload: false });
    dispatch({ type: "SET_PROGRESS", payload: 0 });
  }, [api]);

  const deleteDocument = useCallback(async (id: string) => {
    try {
      await api.deleteDocument(id);
      dispatch({ type: "REMOVE", payload: id });
    } catch (e) {
      dispatch({ type: "SET_ERROR", payload: e instanceof Error ? e.message : "Delete failed" });
    }
  }, [api]);

  return { ...state, refresh, uploadFiles, deleteDocument, clearError: () => dispatch({ type: "SET_ERROR", payload: null }) };
}
```

---

### 2.2 useCitations

Manages citation state from search responses.

```tsx
// hooks/useCitations.ts

import { useState, useCallback } from "react";
import type { CitationResponse } from "../types/knowledge";

export function useCitations() {
  const [citations, setCitations] = useState<CitationResponse[]>([]);

  const updateFromSearch = useCallback((newCitations: CitationResponse[]) => {
    setCitations(newCitations);
  }, []);

  const getCitation = useCallback(
    (id: string) => citations.find((c) => c.citation_id === id),
    [citations]
  );

  const clear = useCallback(() => setCitations([]), []);

  return { citations, updateFromSearch, getCitation, clear };
}
```

---

### 2.3 useDocumentPicker

Handles "/" command trigger, debounced search, and keyboard navigation.

```tsx
// hooks/useDocumentPicker.ts

import { useState, useRef, useCallback, useEffect } from "react";
import type { KnowledgeDocument } from "../types/knowledge";
import { toKnowledgeDocument } from "../types/knowledge";
import { KnowledgeAPI } from "../services/knowledgeApi";

export function useDocumentPicker(api: KnowledgeAPI) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [focused, setFocused] = useState<{ id: string; filename: string }[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const mountedRef = useRef(true);

  // Cleanup on unmount
  useEffect(() => () => { mountedRef.current = false; }, []);

  // Debounced fetch
  useEffect(() => {
    if (!isOpen) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const { documents: docs } = await api.listDocuments();
        if (!mountedRef.current) return;
        const filtered = docs.filter((d) =>
          d.status === "ready" && (!query || d.filename.toLowerCase().includes(query.toLowerCase()))
        );
        setDocuments(filtered.map(toKnowledgeDocument));
        setSelectedIndex(0);
      } catch { if (mountedRef.current) setDocuments([]); }
      finally { if (mountedRef.current) setIsLoading(false); }
    }, 150);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [isOpen, query, api]);

  const open = useCallback(() => { setIsOpen(true); setQuery(""); }, []);
  const close = useCallback(() => { setIsOpen(false); setQuery(""); }, []);

  const select = useCallback((doc: { id: string; filename: string }) => {
    setFocused((prev) => prev.some((d) => d.id === doc.id) ? prev : [...prev, doc]);
    close();
  }, [close]);

  const removeFocused = useCallback((id: string) => {
    setFocused((prev) => prev.filter((d) => d.id !== id));
  }, []);

  const clearFocused = useCallback(() => setFocused([]), []);

  /** Call this from your input's onKeyDown */
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!isOpen) return;
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((i) => Math.min(i + 1, documents.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter":
      case "Tab":
        e.preventDefault();
        if (documents[selectedIndex]) select(documents[selectedIndex]);
        break;
      case "Escape":
        e.preventDefault();
        close();
        break;
    }
  }, [isOpen, documents, selectedIndex, select, close]);

  return {
    isOpen, query, setQuery, documents, selectedIndex, setSelectedIndex,
    isLoading, focused, open, close, select, removeFocused, clearFocused, handleKeyDown,
  };
}
```

---

### 2.4 useRAGConfig

Wraps config and profile management.

```tsx
// hooks/useRAGConfig.ts

import { useState, useCallback, useEffect } from "react";
import type { ConfigResponse, ProfileInfo } from "../types/knowledge";
import { KnowledgeAPI } from "../services/knowledgeApi";

export function useRAGConfig(api: KnowledgeAPI) {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [activeProfile, setActiveProfile] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const [cfg, prof] = await Promise.all([api.getConfig(), api.listProfiles()]);
      setConfig(cfg);
      setProfiles(prof.profiles);
      setActiveProfile(prof.active || "");
    } catch {}
    finally { setIsLoading(false); }
  }, [api]);

  const patchConfig = useCallback(async (updates: Partial<ConfigResponse>) => {
    try {
      const updated = await api.patchConfig(updates);
      setConfig(updated);
    } catch {
      // Re-fetch to ensure UI matches server state
      await refresh().catch(() => {});
    }
  }, [api, refresh]);

  const applyProfile = useCallback(async (name: string) => {
    try {
      const updated = await api.applyProfile(name);
      setConfig(updated);
      setActiveProfile(name);
    } catch {
      await refresh().catch(() => {});
    }
  }, [api, refresh]);

  useEffect(() => { refresh(); }, [refresh]);

  return { config, profiles, activeProfile, isLoading, refresh, patchConfig, applyProfile };
}
```

---

### 2.5 useChatHistory

Wraps the `chatHistory` storage utility in a React hook. Loads history on mount and exposes a `persist` method. **Call `marked()` once in the caller** and pass the resulting HTML — this avoids a second render pass and keeps the markdown→HTML conversion in one explicit place.

```tsx
// hooks/useChatHistory.ts

import { useState, useCallback } from "react";
import { saveMessage, getMessages, clearMessages, MAX_MESSAGES, PersistedMessage } from "../utils/chatHistory";
import type { CitationResponse } from "../types/knowledge";

export function useChatHistory(sessionId: string) {
  // Load synchronously so callers can read history.length in their own
  // useState lazy initializer on the same render cycle (no async gap).
  const [history, setHistory] = useState<PersistedMessage[]>(() => getMessages(sessionId));

  /**
   * Call after each completed LLM turn.
   * Accepts pre-rendered HTML (not raw markdown) — call marked() once in the
   * caller, then pass the result here and to the display component. This avoids
   * a second lexer+renderer pass for the same string.
   */
  const persist = useCallback((
    userQuery: string,
    answerHtml: string,
    citations: CitationResponse[],
    chunks: Array<{ chunk_id: string; content: string; chunk_index?: number }> = [],
  ) => {
    const message: PersistedMessage = {
      id: `msg_${Date.now()}`,
      userQuery,
      answerHtml,
      citations,
      chunks,
      timestamp: Date.now(),
    };
    saveMessage(sessionId, message);
    // Mirror in local state so the UI updates immediately without re-reading storage
    setHistory((prev) => [...prev.slice(-(MAX_MESSAGES - 1)), message]);
  }, [sessionId]);

  /** Clear on conversation restart. */
  const clear = useCallback(() => {
    clearMessages(sessionId);
    setHistory([]);
  }, [sessionId]);

  return { history, persist, clear };
}
```

**Usage pattern:**

```tsx
const chatHistory = useChatHistory(sessionId);

// After LLM reply (convert markdown once, reuse for display and persistence):
const answerHtml = marked.parse(llmMarkdownReply) as string;
chatHistory.persist(userQuery, answerHtml, result.citations, result.results.map(r => r.chunk));

// Restore previous session into initial message list:
const [messages, setMessages] = useState<Message[]>(() =>
  chatHistory.history.flatMap((p) => [
    { role: "user" as const, content: p.userQuery },
    { role: "assistant" as const, content: p.answerHtml, citations: p.citations, chunks: p.chunks, isHtml: true },
  ])
);

// On restart:
chatHistory.clear();
setMessages([]);
```

> **Stale closure note:** If you save `userQuery` via a state variable and call `persist` inside an async callback, capture the query in a local variable before the `await` so the closure has the right value. See the full example in [Part 4](#part-4-full-integration-example).

---

## Part 3: React Components

All components use **inline styles** — no external CSS required. Replace styles with your design system as needed.

### 3.1 DocumentUploadPanel

Drag-and-drop upload with progress bar and document list. Uses `useKnowledge` hook.

```tsx
// components/DocumentUploadPanel.tsx

import React, { useRef, useState } from "react";
import { KnowledgeDocument, DocumentStatus, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB } from "../types/knowledge";

interface Props {
  documents: KnowledgeDocument[];
  isUploading: boolean;
  uploadProgress: number;
  error: string | null;
  onUpload: (files: File[]) => void;
  onDelete: (id: string) => void;
  onClearError: () => void;
  onValidationError?: (errors: string[]) => void;
}

function validateFiles(files: File[]): { valid: File[]; errors: string[] } {
  const valid: File[] = [];
  const errors: string[] = [];
  for (const file of files) {
    const ext = "." + (file.name.split(".").pop()?.toLowerCase() || "");
    if (!SUPPORTED_EXTENSIONS.includes(ext as any)) {
      errors.push(`${file.name}: unsupported file type`);
    } else if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      errors.push(`${file.name}: exceeds ${MAX_FILE_SIZE_MB}MB limit`);
    } else {
      valid.push(file);
    }
  }
  return { valid, errors };
}

const statusColors: Record<string, string> = {
  [DocumentStatus.READY]: "#22c55e",
  [DocumentStatus.PROCESSING]: "#eab308",
  [DocumentStatus.INDEXING]: "#3b82f6",
  [DocumentStatus.UPLOADING]: "#a855f7",
  [DocumentStatus.ERROR]: "#ef4444",
};

const statusLabels: Record<string, string> = {
  [DocumentStatus.READY]: "Ready",
  [DocumentStatus.PROCESSING]: "Processing...",
  [DocumentStatus.INDEXING]: "Indexing...",
  [DocumentStatus.UPLOADING]: "Uploading...",
  [DocumentStatus.ERROR]: "Error",
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentUploadPanel({ documents, isUploading, uploadProgress, error, onUpload, onDelete, onClearError, onValidationError }: Props) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFiles = (files: File[]) => {
    const { valid, errors } = validateFiles(files);
    if (errors.length > 0) {
      setValidationError(errors.join(", "));
      onValidationError?.(errors);
    }
    if (valid.length > 0) onUpload(valid);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    processFiles(Array.from(e.dataTransfer.files));
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(Array.from(e.target.files || []));
    e.target.value = ""; // Reset for re-upload
  };

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 14, fontWeight: 600 }}>
        Documents ({documents.filter((d) => d.status === DocumentStatus.READY).length})
      </h3>

      {/* Error banner */}
      {(error || validationError) && (
        <div style={{ padding: "8px 12px", marginBottom: 12, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 6, fontSize: 13, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>{error || validationError}</span>
          <button onClick={() => { onClearError(); setValidationError(null); }} style={{ border: "none", background: "none", cursor: "pointer", fontSize: 16 }}>x</button>
        </div>
      )}

      {/* Upload progress */}
      {isUploading && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ height: 4, background: "#e5e7eb", borderRadius: 2, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${uploadProgress}%`, background: "#3b82f6", transition: "width 0.3s" }} />
          </div>
          <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>Uploading... {uploadProgress}%</div>
        </div>
      )}

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragOver ? "#3b82f6" : "#d1d5db"}`,
          borderRadius: 8,
          padding: "20px 16px",
          textAlign: "center",
          cursor: "pointer",
          background: isDragOver ? "#eff6ff" : "#f9fafb",
          transition: "all 0.2s",
          marginBottom: 16,
        }}
      >
        <div style={{ fontSize: 24, marginBottom: 4 }}>📎</div>
        <div style={{ fontSize: 13, color: "#6b7280" }}>
          Drop files here or click to upload
        </div>
        <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 4 }}>
          {SUPPORTED_EXTENSIONS.join(", ")}
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={SUPPORTED_EXTENSIONS.join(",")}
          onChange={handleFileSelect}
          style={{ display: "none" }}
        />
      </div>

      {/* Document list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {documents.map((doc) => (
          <div
            key={doc.id}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "8px 10px", borderRadius: 6, fontSize: 13,
              background: "#f9fafb", border: "1px solid #f3f4f6",
            }}
          >
            <span>📄</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {doc.filename}
              </div>
              <div style={{ fontSize: 11, color: "#9ca3af" }}>
                {formatSize(doc.sizeBytes)}
                {doc.chunkCount > 0 && ` · ${doc.chunkCount} chunks`}
              </div>
            </div>
            <span style={{
              fontSize: 11, padding: "2px 6px", borderRadius: 9999,
              background: `${statusColors[doc.status]}20`,
              color: statusColors[doc.status],
              fontWeight: 500,
            }}>
              {statusLabels[doc.status] || doc.status}
            </span>
            {doc.status === DocumentStatus.READY && (
              <button
                onClick={() => onDelete(doc.id)}
                title="Delete"
                style={{ border: "none", background: "none", cursor: "pointer", padding: 2, fontSize: 14, color: "#9ca3af" }}
              >
                🗑
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

### 3.2 SearchResults

Displays search results with score bars and source metadata.

```tsx
// components/SearchResults.tsx

import React from "react";
import type { SearchResultResponse, RoutingInfo } from "../types/knowledge";

interface Props {
  results: SearchResultResponse[];
  routing?: RoutingInfo;
  isLoading?: boolean;
}

export function SearchResults({ results, routing, isLoading }: Props) {
  if (isLoading) {
    return <div style={{ padding: 16, color: "#6b7280", fontSize: 13 }}>Searching...</div>;
  }

  return (
    <div style={{ fontFamily: "system-ui, sans-serif" }}>
      {routing && (
        <div style={{ padding: "6px 12px", fontSize: 11, color: "#6b7280", background: "#f9fafb", borderBottom: "1px solid #f3f4f6" }}>
          {routing.tier.toUpperCase()} · {routing.rationale}
        </div>
      )}

      {results.map((r, i) => (
        <div key={r.chunk.chunk_id} style={{ padding: "12px 16px", borderBottom: "1px solid #f3f4f6" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
            <span style={{ fontWeight: 600 }}>📄 {r.filename}</span>
            <span style={{ color: "#6b7280" }}>
              {r.chunk.page_number && `p.${r.chunk.page_number} · `}
              {Math.round(Math.max(0, Math.min(1, r.score)) * 100)}%
            </span>
          </div>
          {/* Score bar */}
          <div style={{ height: 3, background: "#e5e7eb", borderRadius: 2, marginBottom: 6, overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 2,
              width: `${Math.max(Math.min(r.score * 100, 100), 2)}%`,
              background: r.score >= 0.8 ? "#22c55e" : r.score >= 0.6 ? "#eab308" : "#9ca3af",
            }} />
          </div>
          <div style={{ fontSize: 13, color: "#374151", lineHeight: 1.5 }}>
            {r.chunk.content.length > 200 ? r.chunk.content.slice(0, 200) + "..." : r.chunk.content}
          </div>
        </div>
      ))}

      {results.length === 0 && (
        <div style={{ padding: 24, textAlign: "center", color: "#9ca3af", fontSize: 13 }}>
          No results found
        </div>
      )}
    </div>
  );
}
```

---

### 3.3 CitationBadge + Preview

Inline `[1]` markers with hover preview showing document name, page, score, and content snippet.

```tsx
// components/CitationBadge.tsx

import React, { useState, useRef, useEffect } from "react";
import type { CitationResponse } from "../types/knowledge";

interface Props {
  /** 0-based indices into the citations array */
  indices: number[];
  citations: CitationResponse[];
  query?: string;
}

function clampScore(score: number): number {
  return Math.max(0, Math.min(1, score));
}

export function CitationBadge({ indices, citations, query }: Props) {
  const [showPreview, setShowPreview] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  // Clean up timers on unmount
  useEffect(() => () => { clearTimeout(timerRef.current); }, []);

  const validIndices = indices.filter((i) => i >= 0 && i < citations.length);
  const validCitations = validIndices.map((i) => citations[i]).filter(Boolean);

  if (validCitations.length === 0) return <span style={{ color: "#9ca3af" }}>[?]</span>;

  const displayNumbers = validIndices.map((i) => i + 1).join(",");
  const primary = validCitations[0];
  const scorePercent = Math.round(clampScore(primary.score) * 100);
  const scoreColor = primary.score >= 0.8 ? "#22c55e" : primary.score >= 0.6 ? "#eab308" : "#9ca3af";

  return (
    <>
      <span
        ref={ref}
        role="button"
        tabIndex={0}
        aria-label={`Citation ${displayNumbers} from ${primary.document_name}`}
        onClick={() => setShowModal(true)}
        onMouseEnter={() => { clearTimeout(timerRef.current); timerRef.current = setTimeout(() => setShowPreview(true), 150); }}
        onMouseLeave={() => { clearTimeout(timerRef.current); timerRef.current = setTimeout(() => setShowPreview(false), 150); }}
        onKeyDown={(e) => { if (e.key === "Enter") setShowModal(true); }}
        style={{
          color: "#3b82f6", cursor: "pointer", fontSize: "0.85em",
          fontWeight: 600, padding: "0 2px", borderRadius: 3,
          background: "#eff6ff", userSelect: "none",
        }}
      >
        [{displayNumbers}]
      </span>

      {/* Hover Preview */}
      {showPreview && (
        <div
          onMouseEnter={() => { clearTimeout(timerRef.current); setShowPreview(true); }}
          onMouseLeave={() => { clearTimeout(timerRef.current); setShowPreview(false); }}
          style={{
            position: "fixed",
            left: ref.current ? ref.current.getBoundingClientRect().left : 0,
            top: ref.current ? ref.current.getBoundingClientRect().bottom + 6 : 0,
            width: 320, padding: 12, background: "#fff",
            border: "1px solid #e5e7eb", borderRadius: 8,
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)", zIndex: 9999,
            fontSize: 13,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
            <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 200 }}>
              📄 {primary.document_name}
            </div>
            <span style={{ color: scoreColor, fontWeight: 600, fontSize: 12 }}>{scorePercent}%</span>
          </div>
          {primary.page_number && (
            <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 6 }}>Page {primary.page_number}</div>
          )}
          <div style={{ color: "#374151", lineHeight: 1.5, fontSize: 12 }}>
            {(primary.content || "").slice(0, 250)}
            {(primary.content || "").length > 250 && "..."}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: "#9ca3af" }}>Click to view full source</div>
        </div>
      )}

      {/* Full Source Modal */}
      {showModal && (
        <div
          onClick={() => setShowModal(false)}
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)",
            display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10000,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: "#fff", borderRadius: 12, padding: 24,
              maxWidth: 600, width: "90%", maxHeight: "80vh", overflow: "auto",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
              <h3 style={{ margin: 0, fontSize: 16 }}>📄 {primary.document_name}</h3>
              <button onClick={() => setShowModal(false)} style={{ border: "none", background: "none", fontSize: 20, cursor: "pointer" }}>x</button>
            </div>
            <div style={{ display: "flex", gap: 16, marginBottom: 16, fontSize: 13, color: "#6b7280" }}>
              {primary.page_number && <span>Page {primary.page_number}</span>}
              <span style={{ color: scoreColor, fontWeight: 600 }}>Relevance: {scorePercent}%</span>
              <span>Source [{validIndices[0] + 1}]</span>
            </div>
            <div style={{ background: "#f9fafb", padding: 16, borderRadius: 8, fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>
              {primary.content}
            </div>
            {primary.highlight && (
              <div style={{ marginTop: 12, padding: "8px 12px", background: "#fefce8", borderRadius: 6, fontSize: 13 }}>
                <strong>Key passage:</strong> {primary.highlight}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
```

---

### 3.4 MessageWithCitations

Renders LLM response text with inline citation badges. Two variants depending on what your LLM returns.

> **Plain text vs HTML:** If your LLM returns plain text, use the plain-text variant below — text is rendered safely with `{seg.content}`. If your LLM returns markdown-formatted responses, **convert to HTML with `marked()` first**, then use the HTML variant which renders via `dangerouslySetInnerHTML`. The HTML variant is required when using `useChatHistory.persist()` since it stores HTML, not raw markdown.
>
> **Citation numbering contract:** The parser maps `[N]` in the LLM text to `citations[N-1]` (0-based). The backend returns `citations` sorted by **relevance score**, not by mention order. For inline badges to point at the right source, your LLM prompt must instruct the model to number citations in the order they appear in the `citations` array (e.g., "cite sources as [1], [2], ... in the order provided"). If your LLM renumbers freely, the badges will open the wrong source.

**Plain-text variant** (LLM returns plain text):

```tsx
// components/MessageWithCitations.tsx

import React from "react";
import type { CitationResponse } from "../types/knowledge";
import { parseInlineCitations } from "../utils/parseCitations";
import { CitationBadge } from "./CitationBadge";

interface Props {
  /** Plain text from LLM. Citation markers [1], [2] are parsed automatically. */
  text: string;
  citations: CitationResponse[];
  query?: string;
}

export function MessageWithCitations({ text, citations, query }: Props) {
  const segments = parseInlineCitations(text);

  return (
    <div style={{ fontSize: 14, lineHeight: 1.7 }}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          return <span key={i}>{seg.content}</span>;
        }
        if (seg.type === "citation" && seg.citationIndices) {
          return (
            <CitationBadge
              key={i}
              indices={seg.citationIndices}
              citations={citations}
              query={query}
            />
          );
        }
        // Invalid citation — render as plain text
        return <span key={i} style={{ color: "#9ca3af" }}>{seg.content}</span>;
      })}
    </div>
  );
}
```

**HTML variant** (LLM returns markdown, or message was restored from `useChatHistory`):

```tsx
// Pass marked(llmReply) as `text` — citation markers survive the markdown→HTML conversion.
// Text segments are injected via dangerouslySetInnerHTML to preserve formatting.

export function MessageWithCitationsHtml({ text, citations, query }: Props) {
  const segments = parseInlineCitations(text);

  return (
    <div style={{ fontSize: 14, lineHeight: 1.7 }}>
      {segments.map((seg, i) => {
        if (seg.type === "text") {
          // ⚠️ XSS: marked does not sanitize HTML by default. If your source documents
          // can contain user-controlled content, run the HTML through DOMPurify first:
          //   import DOMPurify from "dompurify";
          //   const clean = DOMPurify.sanitize(seg.content);
          // Otherwise safe when input comes exclusively from your own LLM/marked pipeline.
          return <span key={i} dangerouslySetInnerHTML={{ __html: seg.content }} />;
        }
        if (seg.type === "citation" && seg.citationIndices) {
          return (
            <CitationBadge
              key={i}
              indices={seg.citationIndices}
              citations={citations}
              query={query}
            />
          );
        }
        return null;
      })}
    </div>
  );
}
```

**Choosing the right variant:**

| Scenario | Variant |
| --- | --- |
| LLM returns plain text | `MessageWithCitations` |
| LLM returns markdown | `MessageWithCitationsHtml` + `marked()` before passing |
| Restored from `useChatHistory` | `MessageWithCitationsHtml` (history stores HTML) |

---

### 3.5 SourcesSidebar

Displays all citations from the current response as an expandable list.

> **`chunks` prop:** The `CitationResponse.content` field from the REST API is a short preview (often truncated). Pass the full `chunks` array from `searchWithCitations` to get the complete text in expanded rows. Citation and chunk are matched by `chunk_id`.

```tsx
// components/SourcesSidebar.tsx

import React, { useState } from "react";
import type { CitationResponse } from "../types/knowledge";

interface Props {
  citations: CitationResponse[];
  /**
   * Optional full chunk data from searchWithCitations().
   * When provided, chunk.content is used instead of citation.content
   * (which may be a truncated preview from the REST API).
   */
  chunks?: Array<{ chunk_id: string; content: string; chunk_index?: number }>;
}

export function SourcesSidebar({ citations, chunks }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (citations.length === 0) return null;

  // Prefer full chunk content over the truncated REST preview
  const getContent = (c: CitationResponse): string => {
    const chunk = chunks?.find((ch) => ch.chunk_id === c.chunk_id);
    return chunk?.content || c.content || "No content preview available.";
  };

  return (
    <div style={{ padding: 12, fontFamily: "system-ui, sans-serif" }}>
      <h4 style={{ margin: "0 0 8px", fontSize: 13, fontWeight: 600, color: "#374151" }}>
        Sources ({citations.length})
      </h4>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {citations.map((c, i) => {
          const isExpanded = expandedId === c.citation_id;
          const clamped = Math.max(0, Math.min(1, c.score));
          const scorePercent = Math.round(clamped * 100);
          const scoreColor = c.score >= 0.8 ? "#22c55e" : c.score >= 0.6 ? "#eab308" : "#9ca3af";

          return (
            <div key={c.citation_id} style={{ border: "1px solid #f3f4f6", borderRadius: 6, overflow: "hidden" }}>
              <div
                onClick={() => setExpandedId(isExpanded ? null : c.citation_id)}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "8px 10px", cursor: "pointer",
                  background: isExpanded ? "#f9fafb" : "#fff",
                  fontSize: 13,
                }}
              >
                <span style={{
                  fontSize: 11, fontWeight: 700, color: "#3b82f6",
                  background: "#eff6ff", padding: "1px 5px", borderRadius: 3,
                }}>
                  [{i + 1}]
                </span>
                <span style={{ flex: 1, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {c.document_name}
                </span>
                {c.page_number && <span style={{ fontSize: 11, color: "#9ca3af" }}>p.{c.page_number}</span>}
                <span style={{ fontSize: 11, fontWeight: 600, color: scoreColor }}>{scorePercent}%</span>
                <span style={{ fontSize: 10 }}>{isExpanded ? "▲" : "▼"}</span>
              </div>
              {isExpanded && (
                <div style={{ padding: "8px 10px", fontSize: 12, color: "#374151", lineHeight: 1.5, background: "#f9fafb", borderTop: "1px solid #f3f4f6" }}>
                  {getContent(c)}
                  {c.highlight && (
                    <div style={{ marginTop: 8, padding: "6px 8px", background: "#fefce8", borderRadius: 4, fontSize: 11 }}>
                      <strong>Highlight:</strong> {c.highlight}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

---

### 3.6 DocumentPicker

"/" command popup for selecting documents to scope searches.

```tsx
// components/DocumentPicker.tsx

import React from "react";
import type { KnowledgeDocument } from "../types/knowledge";

interface Props {
  isOpen: boolean;
  documents: KnowledgeDocument[];
  selectedIndex: number;
  query: string;
  isLoading: boolean;
  onSelect: (doc: { id: string; filename: string }) => void;
  onIndexChange: (i: number) => void;
  onClose: () => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentPickerPopup({ isOpen, documents, selectedIndex, query, isLoading, onSelect, onIndexChange, onClose }: Props) {
  if (!isOpen) return null;

  return (
    <div style={{
      position: "absolute", bottom: "100%", left: 0, right: 0,
      background: "#fff", border: "1px solid #e5e7eb", borderRadius: 8,
      boxShadow: "0 4px 12px rgba(0,0,0,0.1)", zIndex: 100,
      maxHeight: 280, overflow: "hidden", fontSize: 13,
    }}>
      <div style={{ padding: "8px 12px", borderBottom: "1px solid #f3f4f6", fontSize: 12, color: "#6b7280" }}>
        📚 Knowledge Base Documents {query && <span style={{ color: "#3b82f6" }}>"{query}"</span>}
      </div>

      <div style={{ maxHeight: 200, overflow: "auto" }}>
        {isLoading ? (
          <div style={{ padding: 16, textAlign: "center", color: "#9ca3af" }}>Searching...</div>
        ) : documents.length === 0 ? (
          <div style={{ padding: 16, textAlign: "center", color: "#9ca3af" }}>
            {query ? `No documents matching "${query}"` : "No documents available"}
          </div>
        ) : (
          documents.map((doc, i) => (
            <div
              key={doc.id}
              onClick={() => onSelect({ id: doc.id, filename: doc.filename })}
              onMouseEnter={() => onIndexChange(i)}
              style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "8px 12px", cursor: "pointer",
                background: i === selectedIndex ? "#eff6ff" : "#fff",
              }}
            >
              <span>📄</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 500 }}>{doc.filename}</div>
                <div style={{ fontSize: 11, color: "#9ca3af" }}>
                  {doc.chunkCount ? `${doc.chunkCount} chunks` : ""}
                  {doc.sizeBytes ? ` · ${formatSize(doc.sizeBytes)}` : ""}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      <div style={{ padding: "6px 12px", borderTop: "1px solid #f3f4f6", fontSize: 11, color: "#9ca3af", display: "flex", gap: 12 }}>
        <span>↑↓ navigate</span>
        <span>Enter select</span>
        <span>Esc close</span>
      </div>
    </div>
  );
}

/** Selected documents displayed as removable chips */
export function FocusedDocumentChips({ documents, onRemove }: {
  documents: { id: string; filename: string }[];
  onRemove: (id: string) => void;
}) {
  if (documents.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", padding: "4px 0", fontSize: 12 }}>
      <span style={{ color: "#6b7280", lineHeight: "24px" }}>📚 Searching in:</span>
      {documents.map((doc) => (
        <span key={doc.id} style={{
          display: "inline-flex", alignItems: "center", gap: 4,
          padding: "2px 8px", background: "#eff6ff", borderRadius: 9999,
          color: "#3b82f6", fontWeight: 500,
        }}>
          {doc.filename}
          <button onClick={() => onRemove(doc.id)} style={{ border: "none", background: "none", cursor: "pointer", padding: 0, color: "#93c5fd", fontSize: 14 }}>x</button>
        </span>
      ))}
    </div>
  );
}
```

---

### 3.7 ProfileSwitcher

Dropdown for switching between RAG quality presets.

```tsx
// components/ProfileSwitcher.tsx

import React from "react";
import type { ProfileInfo } from "../types/knowledge";

interface Props {
  profiles: ProfileInfo[];
  activeProfile: string;
  onApply: (name: string) => void;
  isLoading?: boolean;
}

const profileEmojis: Record<string, string> = {
  speed: "⚡", standard: "⚖️", balanced: "🎯", max_quality: "🔬", quality: "🔬",
};

export function ProfileSwitcher({ profiles, activeProfile, onApply, isLoading }: Props) {
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", fontFamily: "system-ui, sans-serif" }}>
      {profiles.map((p) => (
        <button
          key={p.name}
          onClick={() => onApply(p.name)}
          disabled={isLoading}
          title={p.description}
          style={{
            padding: "6px 12px", borderRadius: 6, fontSize: 13,
            fontWeight: p.name === activeProfile ? 600 : 400,
            border: p.name === activeProfile ? "2px solid #3b82f6" : "1px solid #d1d5db",
            background: p.name === activeProfile ? "#eff6ff" : "#fff",
            color: p.name === activeProfile ? "#1d4ed8" : "#374151",
            cursor: isLoading ? "wait" : "pointer",
          }}
        >
          {profileEmojis[p.name] || "⚙️"} {p.name}
        </button>
      ))}
    </div>
  );
}
```

---

### 3.8 RAGConfigPanel

Feature toggles and parameter sliders for per-session config patching.

```tsx
// components/RAGConfigPanel.tsx

import React from "react";
import type { ConfigResponse } from "../types/knowledge";

interface Props {
  config: ConfigResponse;
  onPatch: (updates: Partial<ConfigResponse>) => void;
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div
      role="switch"
      aria-checked={value}
      aria-label={label}
      tabIndex={0}
      onClick={() => onChange(!value)}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onChange(!value); } }}
      style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", fontSize: 13, cursor: "pointer" }}
    >
      <span>{label}</span>
      <div style={{
        width: 36, height: 20, borderRadius: 10,
        background: value ? "#3b82f6" : "#d1d5db",
        position: "relative", transition: "background 0.2s",
      }}>
        <div style={{
          width: 16, height: 16, borderRadius: "50%",
          background: "#fff", position: "absolute", top: 2,
          left: value ? 18 : 2, transition: "left 0.2s",
        }} />
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <div style={{ padding: "6px 0", fontSize: 13 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 600 }}>{step < 1 ? value.toFixed(2) : value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%" }}
      />
    </div>
  );
}

export function RAGConfigPanel({ config, onPatch }: Props) {
  // Debounce rapid changes (especially from slider drag)
  const timerRef = React.useRef<ReturnType<typeof setTimeout>>();
  const pendingRef = React.useRef<Partial<ConfigResponse>>({});

  React.useEffect(() => () => clearTimeout(timerRef.current), []);

  const patch = React.useCallback((updates: Partial<ConfigResponse>) => {
    pendingRef.current = { ...pendingRef.current, ...updates };
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      onPatch(pendingRef.current);
      pendingRef.current = {};
    }, 300);
  }, [onPatch]);

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif", maxWidth: 400 }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 14, fontWeight: 600 }}>RAG Configuration</h3>

      <div style={{ marginBottom: 16 }}>
        <h4 style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", textTransform: "uppercase" }}>Search Features</h4>
        <Toggle label="Hybrid Search" value={config.enable_hybrid_search} onChange={(v) => patch({ enable_hybrid_search: v })} />
        <Toggle label="Reranking" value={config.enable_reranking} onChange={(v) => patch({ enable_reranking: v })} />
        <Toggle label="MMR Diversity" value={config.enable_mmr} onChange={(v) => patch({ enable_mmr: v })} />
        <Toggle label="HyDE Expansion" value={config.enable_hyde} onChange={(v) => patch({ enable_hyde: v })} />
        <Toggle label="Agentic RAG" value={config.use_agentic_rag} onChange={(v) => patch({ use_agentic_rag: v })} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <h4 style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", textTransform: "uppercase" }}>Parameters</h4>
        <Slider label="Top-K (before reranking)" value={config.default_top_k} min={1} max={100} step={1} onChange={(v) => patch({ default_top_k: v })} />
        <Slider label="Rerank Top-K" value={config.rerank_top_k} min={1} max={100} step={1} onChange={(v) => patch({ rerank_top_k: v })} />
        <Slider label="Similarity Threshold" value={config.similarity_threshold} min={-1} max={1} step={0.05} onChange={(v) => patch({ similarity_threshold: v })} />
        <Slider label="Hybrid Alpha" value={config.hybrid_alpha} min={0} max={1} step={0.05} onChange={(v) => patch({ hybrid_alpha: v })} />
      </div>

      <div>
        <h4 style={{ margin: "0 0 8px", fontSize: 12, color: "#6b7280", textTransform: "uppercase" }}>Quality Gates</h4>
        <Toggle label="CRAG (Corrective RAG)" value={config.enable_crag} onChange={(v) => patch({ enable_crag: v })} />
        <Toggle label="Hallucination Check" value={config.enable_hallucination_check} onChange={(v) => patch({ enable_hallucination_check: v })} />
        <Toggle label="Answer Quality Check" value={config.enable_answer_quality_check} onChange={(v) => patch({ enable_answer_quality_check: v })} />
      </div>
    </div>
  );
}
```

---

## Part 4: Full Integration Example

Wire everything together into a complete RAG chat UI with per-message citation persistence that survives page refreshes.

```tsx
// App.tsx — Complete RAG Chat Application (with persistence)

import React, { useState, useRef } from "react";
import { marked } from "marked";
import { KnowledgeAPI } from "./services/knowledgeApi";
import { useKnowledge } from "./hooks/useKnowledge";
import { useDocumentPicker } from "./hooks/useDocumentPicker";
import { useRAGConfig } from "./hooks/useRAGConfig";
import { useChatHistory } from "./hooks/useChatHistory";
import { DocumentUploadPanel } from "./components/DocumentUploadPanel";
import { MessageWithCitationsHtml } from "./components/MessageWithCitations";
import { SourcesSidebar } from "./components/SourcesSidebar";
import { DocumentPickerPopup, FocusedDocumentChips } from "./components/DocumentPicker";
import { ProfileSwitcher } from "./components/ProfileSwitcher";
import type { CitationResponse } from "./types/knowledge";

// ⚠️ SSR note: localStorage is not available during server-side rendering (Next.js, Remix).
// If you use SSR, move SESSION_ID initialization inside the component with useEffect,
// or guard with: if (typeof window === "undefined") return null;
const SESSION_ID = localStorage.getItem("rag_session_id") ?? (() => {
  const id = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  localStorage.setItem("rag_session_id", id);
  return id;
})();

const api = new KnowledgeAPI("http://localhost:8000");

interface Message {
  role: "user" | "assistant";
  /** Always HTML for assistant messages (call marked() before setting). Plain text for user messages. */
  content: string;
  citations?: CitationResponse[];
  chunks?: Array<{ chunk_id: string; content: string; chunk_index?: number }>;
  /** True when this message was loaded from sessionStorage, not streamed live. */
  isRestored?: boolean;
}

export default function RAGChat() {
  const knowledge = useKnowledge(api);
  const picker = useDocumentPicker(api);
  const ragConfig = useRAGConfig(api);
  const chatHistory = useChatHistory(SESSION_ID);

  // Restore previous session on first render.
  // Works because useChatHistory loads from storage synchronously (lazy useState),
  // so chatHistory.history is populated before this initializer runs.
  const [messages, setMessages] = useState<Message[]>(() =>
    chatHistory.history.flatMap((p) => [
      { role: "user" as const, content: p.userQuery },
      {
        role: "assistant" as const,
        content: p.answerHtml,
        citations: p.citations,
        chunks: p.chunks,
        isRestored: true,
      },
    ])
  );
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize session on mount
  React.useEffect(() => {
    api.createSession(SESSION_ID)
      .then(() => knowledge.refresh())
      .catch((e) => console.error("Session init failed:", e));
  }, []);

  const handleSend = async () => {
    if (!input.trim()) return;
    // Capture query synchronously — avoids stale closure in the async callback below
    const query = input;
    setInput("");

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    inputRef.current?.focus();

    try {
      const focusedIds = picker.focused.map((d) => d.id);
      const result = await api.searchWithCitations(query, {
        top_k: 5,
        document_ids: focusedIds.length > 0 ? focusedIds : undefined,
      });

      // Replace with your actual LLM call; inject result.context into the prompt
      const assistantMarkdown = `Based on the documents:\n\n${result.context}`;

      // Extract chunks once — used for both immediate render and persistence below
      const chunks = result.results.map((r) => r.chunk);
      // Convert markdown → HTML once; reuse for both display and persistence.
      // marked.parse() is the explicit synchronous form — marked() returns string|Promise<string>.
      const assistantHtml = marked.parse(assistantMarkdown) as string;

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: assistantHtml,
          citations: result.citations,
          chunks,
          isRestored: false,
        },
      ]);

      // Persist — pass HTML (already converted above), not raw markdown
      chatHistory.persist(query, assistantHtml, result.citations, chunks);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "<p>Search failed. Please try again.</p>", isRestored: false },
      ]);
    }
  };

  const handleRestart = () => {
    chatHistory.clear();
    setMessages([]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (picker.isOpen) {
      picker.setQuery(val);
    } else if (val === "/") {
      picker.open();
      setInput("");
    } else {
      setInput(val);
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "system-ui, sans-serif" }}>
      {/* Left: Documents */}
      <div style={{ width: 280, borderRight: "1px solid #e5e7eb", overflow: "auto" }}>
        <DocumentUploadPanel
          documents={knowledge.documents}
          isUploading={knowledge.isUploading}
          uploadProgress={knowledge.uploadProgress}
          error={knowledge.error}
          onUpload={knowledge.uploadFiles}
          onDelete={knowledge.deleteDocument}
          onClearError={knowledge.clearError}
        />
      </div>

      {/* Center: Chat */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        {/* Toolbar */}
        <div style={{ padding: "8px 16px", borderBottom: "1px solid #f3f4f6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <ProfileSwitcher
            profiles={ragConfig.profiles}
            activeProfile={ragConfig.activeProfile}
            onApply={ragConfig.applyProfile}
            isLoading={ragConfig.isLoading}
          />
          <button
            onClick={handleRestart}
            title="Clear conversation"
            style={{ padding: "4px 10px", fontSize: 12, borderRadius: 6, border: "1px solid #d1d5db", cursor: "pointer", background: "#fff" }}
          >
            ↺ Restart
          </button>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
          {messages.map((msg, i) => (
            <div key={i} style={{ marginBottom: 16, maxWidth: 640 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: "#6b7280", marginBottom: 4 }}>
                {msg.role === "user" ? "You" : "Assistant"}
              </div>

              {msg.role === "user" && (
                <div style={{ fontSize: 14, lineHeight: 1.7 }}>{msg.content}</div>
              )}

              {msg.role === "assistant" && (
                <>
                  {msg.isRestored && (
                    <div style={{ fontSize: 11, color: "#9ca3af", marginBottom: 4 }}>↺ Restored from previous session</div>
                  )}
                  {/* Both live and restored messages store HTML — use the HTML variant */}
                  <MessageWithCitationsHtml
                    text={msg.content}
                    citations={msg.citations ?? []}
                  />
                  {(msg.citations?.length ?? 0) > 0 && (
                    <SourcesSidebar
                      citations={msg.citations!}
                      chunks={msg.chunks}
                    />
                  )}
                </>
              )}
            </div>
          ))}
        </div>

        {/* Input area */}
        <div style={{ padding: 16, borderTop: "1px solid #e5e7eb", position: "relative" }}>
          <FocusedDocumentChips documents={picker.focused} onRemove={picker.removeFocused} />
          <DocumentPickerPopup
            isOpen={picker.isOpen}
            documents={picker.documents}
            selectedIndex={picker.selectedIndex}
            query={picker.query}
            isLoading={picker.isLoading}
            onSelect={picker.select}
            onIndexChange={picker.setSelectedIndex}
            onClose={picker.close}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <input
              ref={inputRef}
              value={picker.isOpen ? picker.query : input}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                picker.handleKeyDown(e);
                if (e.key === "Enter" && !picker.isOpen) handleSend();
              }}
              placeholder='Ask a question... (type "/" to focus documents)'
              style={{
                flex: 1, padding: "10px 14px", borderRadius: 8,
                border: "1px solid #d1d5db", fontSize: 14, outline: "none",
              }}
            />
            <button
              onClick={handleSend}
              style={{
                padding: "10px 20px", borderRadius: 8, border: "none",
                background: "#3b82f6", color: "#fff", fontWeight: 600,
                cursor: "pointer", fontSize: 14,
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Right: Sources for the most recent query */}
      <div style={{ width: 260, borderLeft: "1px solid #e5e7eb", overflow: "auto" }}>
        {(() => {
          const last = [...messages].reverse().find((m) => m.role === "assistant" && m.citations?.length);
          return last ? <SourcesSidebar citations={last.citations!} chunks={last.chunks} /> : null;
        })()}
      </div>
    </div>
  );
}
```

This gives you a complete RAG chat UI with:

- Document upload panel with drag-and-drop
- Profile switcher (speed / standard / balanced / max_quality)
- Chat with inline citation badges and hover previews
- Per-message citations — each message owns its own `citations` + `chunks` array
- Page-refresh persistence via `useChatHistory` (sessionStorage)
- "/" command for document-focused search
- Sources sidebar showing the most-recent query's sources
- Restart button that clears both UI state and persisted history

## See Also

- [UI Integration Guide](integration.md)
- [User Guide](../user_guide.md)
