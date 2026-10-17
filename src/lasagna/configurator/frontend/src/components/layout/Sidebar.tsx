import React, { useState, useMemo } from "react";
import { Plus, Lock, Trash2, Star } from "lucide-react";
import type { RAGProfile, ParameterDef } from "../../api/types";
import { REINDEX_KEYS, PROTECTED_PROFILES } from "../../schema/reindexKeys";

/**
 * The profile to start with, as the README, `lasagna init` and doctor say: a
 * 92 MB model and no LLM. Each profile's description is the server's
 * (BUILTIN_PROFILE_DESCRIPTIONS), which states what it costs.
 */
const STARTING_PROFILE = "speed";

interface Props {
  profiles: RAGProfile[];
  activeProfile: string;
  onSelectProfile: (name: string) => void;
  onNewProfile: () => void;
  onDeleteProfile: (name: string) => void;
  config: Record<string, unknown>;
  originalConfig: Record<string, unknown>;
  schemaParams?: ParameterDef[];
}

export default function Sidebar({
  profiles, activeProfile, onSelectProfile, onNewProfile, onDeleteProfile,
  config, originalConfig, schemaParams,
}: Props) {
  const [hoverDelete, setHoverDelete] = useState<string | null>(null);

  // Build a lookup from config key -> friendly label
  const labelMap = useMemo(() => {
    const map: Record<string, string> = {};
    if (schemaParams) {
      for (const p of schemaParams) {
        map[p.key] = p.label;
      }
    }
    return map;
  }, [schemaParams]);

  function friendlyLabel(key: string): string {
    return labelMap[key] || key;
  }

  // Compute diff entries
  const diffs: { key: string; label: string; oldVal: string; newVal: string; reindex: boolean }[] = [];
  const allKeys = new Set([...Object.keys(config), ...Object.keys(originalConfig)]);
  for (const key of allKeys) {
    const oldVal = originalConfig[key];
    const newVal = config[key];
    if (typeof oldVal === "object" && typeof newVal === "object" && oldVal && newVal) {
      const subOld = oldVal as Record<string, unknown>;
      const subNew = newVal as Record<string, unknown>;
      const subKeys = new Set([...Object.keys(subOld), ...Object.keys(subNew)]);
      for (const sk of subKeys) {
        if (subOld[sk] !== subNew[sk] && subNew[sk] !== undefined) {
          const fullKey = `${key}.${sk}`;
          diffs.push({
            key: fullKey,
            label: friendlyLabel(fullKey),
            oldVal: String(subOld[sk] ?? "\u2014"),
            newVal: String(subNew[sk]),
            reindex: REINDEX_KEYS.includes(fullKey),
          });
        }
      }
    } else if (oldVal !== newVal && newVal !== undefined) {
      diffs.push({
        key,
        label: friendlyLabel(key),
        oldVal: String(oldVal ?? "\u2014"),
        newVal: String(newVal),
        reindex: REINDEX_KEYS.includes(key),
      });
    }
  }

  return (
    <aside className="cfg-sidebar">
      <div className="cfg-sidebar-section">
        <div className="cfg-sidebar-label">Profiles</div>
        <div className="cfg-profile-list">
          {profiles.map((p) => {
            const isBuiltin = PROTECTED_PROFILES.has(p.name);
            const isActive = p.name === activeProfile;
            const isStart = p.name === STARTING_PROFILE;
            const description = p.description;
            return (
              <div
                key={p.name}
                className={`cfg-profile-item ${isActive ? "active" : ""}`}
                onClick={() => onSelectProfile(p.name)}
                onMouseEnter={() => setHoverDelete(p.name)}
                onMouseLeave={() => setHoverDelete(null)}
              >
                <div className="cfg-profile-info">
                  <div className="cfg-profile-name-row">
                    {isBuiltin && <Lock size={12} style={{ color: "var(--color-text-muted)", flexShrink: 0 }} />}
                    <span className="name">{p.name}</span>
                    {isStart && (
                      <span className="badge recommended" title="Start here: a 92 MB model, no LLM">
                        <Star size={9} /> start here
                      </span>
                    )}
                    {!isBuiltin && (
                      <span className="badge custom">custom</span>
                    )}
                  </div>
                  {description && (
                    <div className="cfg-profile-desc">{description}</div>
                  )}
                </div>
                {!isBuiltin && hoverDelete === p.name && (
                  <Trash2
                    size={13}
                    style={{ color: "var(--color-error)", cursor: "pointer", flexShrink: 0 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteProfile(p.name);
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="cfg-sidebar-actions">
        <button className="cfg-new-profile-btn" onClick={onNewProfile}>
          <Plus size={14} />
          New Profile
        </button>
      </div>

      <div className="cfg-diff-panel">
        <div className="cfg-diff-title">
          Changes {diffs.length > 0 && `(${diffs.length})`}
        </div>
        {diffs.length === 0 ? (
          <div className="cfg-diff-empty">No unsaved changes</div>
        ) : (
          <div className="cfg-diff-list">
            {diffs.map((d) => (
              <div key={d.key} className={`cfg-diff-entry ${d.reindex ? "reindex" : ""}`}>
                {d.reindex && <span className="cfg-diff-dot" title="Requires re-indexing documents">&#9679;</span>}
                <span className="label">{d.label}:</span>
                <span className="arrow">{d.oldVal} &rarr; {d.newVal}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
