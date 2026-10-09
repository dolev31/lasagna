/**
 * API client for the RAG configurator.
 *
 * Uses the existing knowledge config endpoints + new configurator endpoints.
 */

import type {
  SchemaResponse,
  KnowledgeConfigResponse,
  ValidationResponse,
} from "./types";

const SESSION_KEY = "rag_configurator_session_id";

function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = "configurator-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function qs(): string {
  return `session_id=${encodeURIComponent(getSessionId())}`;
}

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

export async function fetchSchema(): Promise<SchemaResponse> {
  const resp = await fetch("/configurator/api/schema");
  return json<SchemaResponse>(resp);
}

// ---------------------------------------------------------------------------
// Config & profiles
// ---------------------------------------------------------------------------

export async function fetchConfig(): Promise<KnowledgeConfigResponse> {
  const resp = await fetch(`/api/config/knowledge?${qs()}`);
  return json<KnowledgeConfigResponse>(resp);
}

export async function saveActiveConfig(
  activeProfile: string,
  config: Record<string, unknown>,
): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/active?${qs()}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active_profile: activeProfile, config }),
  });
  return json(resp);
}

export async function saveProfile(
  profileName: string,
  config: Record<string, unknown>,
  setActive = true,
): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/profile?${qs()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_name: profileName, config, set_active: setActive }),
  });
  return json(resp);
}

export async function deleteProfile(profileName: string): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/profile/delete?${qs()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile_name: profileName }),
  });
  return json(resp);
}

export async function renameProfile(oldName: string, newName: string): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/profile/rename?${qs()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_name: oldName, new_name: newName }),
  });
  return json(resp);
}

export async function resetConfig(): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/reset?${qs()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope: "active" }),
  });
  return json(resp);
}

export async function exportConfig(): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/export?${qs()}`);
  return json(resp);
}

export async function importConfig(data: unknown): Promise<unknown> {
  const resp = await fetch(`/api/config/knowledge/import?${qs()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return json(resp);
}

export async function rebuildIndex(): Promise<unknown> {
  const resp = await fetch("/api/knowledge/reindex", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: getSessionId(), force_rebuild: true }),
  });
  return json(resp);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export async function validateConfig(
  config: Record<string, unknown>,
  baseline?: Record<string, unknown>,
): Promise<ValidationResponse> {
  const resp = await fetch("/configurator/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, baseline }),
  });
  return json<ValidationResponse>(resp);
}
