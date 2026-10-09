import React from "react";
import { Save, RotateCcw, RefreshCw } from "lucide-react";
import type { SaveStatus } from "../../hooks/useConfig";

interface Props {
  unsavedCount: number;
  reindexNeeded: boolean;
  saveStatus: SaveStatus;
  isProtected: boolean;
  onSave: () => void;
  onReset: () => void;
  onSaveAs: () => void;
  onRebuildIndex: () => void;
}

export default function Footer({
  unsavedCount, reindexNeeded, saveStatus, isProtected,
  onSave, onReset, onSaveAs, onRebuildIndex,
}: Props) {
  const saveLabel =
    saveStatus === "saving" ? "Saving..." :
    saveStatus === "success" ? "Saved!" :
    saveStatus === "error" ? "Error" :
    "Save";

  const saveBtnClass =
    saveStatus === "success" ? "cfg-btn success" :
    saveStatus === "error" ? "cfg-btn danger" :
    "cfg-btn primary";

  return (
    <footer className="cfg-footer">
      <div className="cfg-footer-left">
        {unsavedCount > 0 ? (
          <span className="cfg-footer-changes">
            {unsavedCount} unsaved change{unsavedCount !== 1 ? "s" : ""}
          </span>
        ) : (
          <span>All changes saved</span>
        )}
      </div>
      <div className="cfg-footer-right">
        {reindexNeeded && (
          <button className="cfg-btn warning" onClick={onRebuildIndex}>
            <RefreshCw size={14} />
            Rebuild Index
          </button>
        )}
        <button
          className="cfg-btn"
          onClick={onReset}
          disabled={unsavedCount === 0}
        >
          <RotateCcw size={14} />
          Reset
        </button>
        {isProtected && unsavedCount > 0 ? (
          <button className="cfg-btn primary" onClick={onSaveAs}>
            <Save size={14} />
            Save As...
          </button>
        ) : (
          <button
            className={saveBtnClass}
            onClick={onSave}
            disabled={unsavedCount === 0 || saveStatus === "saving"}
          >
            <Save size={14} />
            {saveLabel}
          </button>
        )}
      </div>
    </footer>
  );
}
