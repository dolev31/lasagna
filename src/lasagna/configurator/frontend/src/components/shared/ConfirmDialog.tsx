import React from "react";

interface Props {
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  title, message, confirmLabel = "Confirm", danger, onConfirm, onCancel,
}: Props) {
  return (
    <div className="cfg-dialog-overlay" onClick={onCancel}>
      <div className="cfg-dialog" onClick={(e) => e.stopPropagation()}>
        <h3>{title}</h3>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", marginBottom: "var(--space-xl)", lineHeight: 1.5 }}>
          {message}
        </p>
        <div className="cfg-dialog-actions">
          <button className="cfg-btn" onClick={onCancel}>Cancel</button>
          <button className={`cfg-btn ${danger ? "danger" : "primary"}`} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
