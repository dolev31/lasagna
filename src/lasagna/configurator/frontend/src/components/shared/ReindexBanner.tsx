import React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  changedKeys: string[];
  onRebuild: () => void;
}

export default function ReindexBanner({ changedKeys, onRebuild }: Props) {
  if (changedKeys.length === 0) return null;

  return (
    <div className="cfg-reindex-banner">
      <AlertTriangle size={16} />
      <div className="message">
        <strong>Documents need to be re-processed.</strong>
        {" "}You changed settings that affect how documents are stored ({changedKeys.length} setting{changedKeys.length !== 1 ? "s" : ""}).
        These changes won't apply to existing documents until you rebuild.
      </div>
      <button className="cfg-btn warning" onClick={onRebuild} style={{ flexShrink: 0 }}>
        <RefreshCw size={13} />
        Rebuild Now
      </button>
    </div>
  );
}
