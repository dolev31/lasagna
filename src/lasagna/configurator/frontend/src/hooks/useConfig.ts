import { useState, useEffect, useCallback, useRef } from "react";
import type { RAGProfile, SchemaResponse } from "../api/types";
import * as api from "../api/client";
import { REINDEX_KEYS, PROTECTED_PROFILES } from "../schema/reindexKeys";

export type SaveStatus = "idle" | "saving" | "success" | "error";

/** Deep-merge override into base (returns new object). */
function deepMerge(
  base: Record<string, unknown>,
  override: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (
      typeof value === "object" &&
      value !== null &&
      !Array.isArray(value) &&
      typeof result[key] === "object" &&
      result[key] !== null &&
      !Array.isArray(result[key])
    ) {
      result[key] = deepMerge(
        result[key] as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      result[key] = value;
    }
  }
  return result;
}

export interface ConfigState {
  // Schema
  schema: SchemaResponse | null;
  schemaLoading: boolean;

  // Profiles
  profiles: RAGProfile[];
  activeProfile: string;
  isProtected: boolean;

  // Config (working copy)
  config: Record<string, unknown>;
  originalConfig: Record<string, unknown>;

  // Derived
  unsavedCount: number;
  reindexNeeded: boolean;
  reindexChangedKeys: string[];

  // Status
  loading: boolean;
  saveStatus: SaveStatus;
  error: string | null;

  // Actions
  setActiveProfile: (name: string) => void;
  updateParam: (key: string, value: unknown) => void;
  save: () => Promise<void>;
  reset: () => Promise<void>;
  saveAs: (name: string) => Promise<void>;
  deleteProfile: (name: string) => Promise<void>;
  reload: () => Promise<void>;
  exportConfig: () => Promise<void>;
  importConfig: (data: unknown) => Promise<void>;
}

export function useConfig(): ConfigState {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [profiles, setProfiles] = useState<RAGProfile[]>([]);
  const [activeProfile, setActiveProfileState] = useState("default");
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [originalConfig, setOriginalConfig] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load schema
  useEffect(() => {
    api.fetchSchema()
      .then(setSchema)
      .catch((e) => console.error("Failed to load schema:", e))
      .finally(() => setSchemaLoading(false));
  }, []);

  // Load config
  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.fetchConfig();
      if (data.profiles?.length) {
        setProfiles(data.profiles);
        const active = data.active_profile || data.profiles[0].name;
        setActiveProfileState(active);
        const profile = data.profiles.find((p) => p.name === active);
        if (profile) {
          const merged = deepMerge(profile.config, data.session_overrides || {});
          setConfig(merged);
          setOriginalConfig(merged);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load config");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  // Switch profile
  const setActiveProfile = useCallback(
    (name: string) => {
      const profile = profiles.find((p) => p.name === name);
      if (profile) {
        setActiveProfileState(name);
        setConfig({ ...profile.config });
        setOriginalConfig({ ...profile.config });
      }
    },
    [profiles],
  );

  // Update a parameter
  const updateParam = useCallback((key: string, value: unknown) => {
    setConfig((prev) => {
      // Handle nested params (e.g. "chunk_cite_retrieval.strategy")
      if (key.includes(".")) {
        const [parent, child] = key.split(".", 2);
        const parentObj = (prev[parent] as Record<string, unknown>) || {};
        return { ...prev, [parent]: { ...parentObj, [child]: value } };
      }
      return { ...prev, [key]: value };
    });
  }, []);

  // Compute unsaved changes
  const unsavedCount = computeDiffCount(config, originalConfig);
  const { reindexNeeded, reindexChangedKeys } = computeReindex(config, originalConfig);
  const isProtected = PROTECTED_PROFILES.has(activeProfile);

  // Save
  const save = useCallback(async () => {
    setSaveStatus("saving");
    try {
      // Validate before saving
      const validation = await api.validateConfig(config);
      if (!validation.valid) {
        const msgs = validation.errors.map((e) => `${e.key}: ${e.message}`).join("; ");
        setError(msgs);
        setSaveStatus("error");
        if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
        saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 3000);
        return;
      }
      await api.saveActiveConfig(activeProfile, config);
      setOriginalConfig({ ...config });
      setError(null);
      setSaveStatus("success");
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
    } catch {
      setSaveStatus("error");
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
    }
  }, [activeProfile, config]);

  // Reset
  const reset = useCallback(async () => {
    setConfig({ ...originalConfig });
  }, [originalConfig]);

  // Save As
  const saveAs = useCallback(
    async (name: string) => {
      setSaveStatus("saving");
      try {
        // Validate before saving
        const validation = await api.validateConfig(config);
        if (!validation.valid) {
          const msgs = validation.errors.map((e) => `${e.key}: ${e.message}`).join("; ");
          setError(msgs);
          setSaveStatus("error");
          if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
          saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 3000);
          return;
        }
        await api.saveProfile(name, config);
        await reload();
        setActiveProfileState(name);
        setError(null);
        setSaveStatus("success");
        if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
        saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
      } catch {
        setSaveStatus("error");
        if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
        saveTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
      }
    },
    [config, reload],
  );

  // Delete
  const deleteProfileFn = useCallback(
    async (name: string) => {
      await api.deleteProfile(name);
      await reload();
    },
    [reload],
  );

  // Export
  const exportConfigFn = useCallback(async () => {
    const data = await api.exportConfig();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag-config-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Import
  const importConfigFn = useCallback(
    async (data: unknown) => {
      await api.importConfig(data);
      await reload();
    },
    [reload],
  );

  // Cleanup
  useEffect(() => {
    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, []);

  return {
    schema,
    schemaLoading,
    profiles,
    activeProfile,
    isProtected,
    config,
    originalConfig,
    unsavedCount,
    reindexNeeded,
    reindexChangedKeys,
    loading,
    saveStatus,
    error,
    setActiveProfile,
    updateParam,
    save,
    reset,
    saveAs,
    deleteProfile: deleteProfileFn,
    reload,
    exportConfig: exportConfigFn,
    importConfig: importConfigFn,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function computeDiffCount(
  current: Record<string, unknown>,
  original: Record<string, unknown>,
): number {
  let count = 0;
  const allKeys = new Set([...Object.keys(current), ...Object.keys(original)]);
  for (const key of allKeys) {
    const a = current[key];
    const b = original[key];
    if (typeof a === "object" && typeof b === "object" && a && b) {
      count += computeDiffCount(
        a as Record<string, unknown>,
        b as Record<string, unknown>,
      );
    } else if (a !== b) {
      count++;
    }
  }
  return count;
}

function computeReindex(
  current: Record<string, unknown>,
  original: Record<string, unknown>,
): { reindexNeeded: boolean; reindexChangedKeys: string[] } {
  const changed: string[] = [];
  for (const key of REINDEX_KEYS) {
    if (current[key] !== original[key] && current[key] !== undefined) {
      changed.push(key);
    }
  }
  return { reindexNeeded: changed.length > 0, reindexChangedKeys: changed };
}
