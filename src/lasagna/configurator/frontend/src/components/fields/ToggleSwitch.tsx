import React from "react";
import HelpTooltip from "./HelpTooltip";

interface Props {
  label: string;
  description: string;
  helpText?: string;
  configKey?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export default function ToggleSwitch({ label, description, helpText, configKey, checked, onChange }: Props) {
  return (
    <div className="cfg-toggle-row">
      <div className="cfg-toggle-info">
        <div className="cfg-toggle-label">
          {label}
          {helpText && <HelpTooltip text={helpText} />}
        </div>
        <div className="cfg-toggle-desc">{description}</div>
        {configKey && <div className="cfg-config-key">{configKey}</div>}
      </div>
      <label className="cfg-toggle-switch">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="cfg-toggle-track" />
      </label>
    </div>
  );
}
