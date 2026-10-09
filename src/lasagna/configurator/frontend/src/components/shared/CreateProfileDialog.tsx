import React, { useState } from "react";
import { PROTECTED_PROFILES } from "../../schema/reindexKeys";

interface Props {
  onSave: (name: string) => void;
  onCancel: () => void;
}

export default function CreateProfileDialog({ onSave, onCancel }: Props) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSave = () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Profile name is required.");
      return;
    }
    if (PROTECTED_PROFILES.has(trimmed.toLowerCase())) {
      setError("This name is reserved for built-in profiles.");
      return;
    }
    if (trimmed.length > 100) {
      setError("Name must be 100 characters or less.");
      return;
    }
    onSave(trimmed);
  };

  return (
    <div className="cfg-dialog-overlay" onClick={onCancel}>
      <div className="cfg-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>Save Profile As</h3>
        <input
          className="cfg-dialog-input"
          type="text"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setError(null);
          }}
          placeholder="Profile name..."
          autoFocus
          onKeyDown={(e) => e.key === "Enter" && handleSave()}
        />
        {error && <div className="cfg-dialog-error">{error}</div>}
        <div className="cfg-dialog-actions">
          <button className="cfg-btn" onClick={onCancel}>Cancel</button>
          <button className="cfg-btn primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}
