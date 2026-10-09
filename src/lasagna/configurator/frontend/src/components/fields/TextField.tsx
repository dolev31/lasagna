import React from "react";
import HelpTooltip from "./HelpTooltip";

interface Props {
  label: string;
  description?: string;
  helpText?: string;
  configKey?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export default function TextField({
  label, description, helpText, configKey, value, onChange, placeholder,
}: Props) {
  return (
    <div className="cfg-select-row">
      <span className="cfg-select-label">
        {label}
        {helpText && <HelpTooltip text={helpText} />}
      </span>
      {description && <div className="cfg-slider-desc">{description}</div>}
      {configKey && <div className="cfg-config-key">{configKey}</div>}
      <input
        type="text"
        className="cfg-text-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}
