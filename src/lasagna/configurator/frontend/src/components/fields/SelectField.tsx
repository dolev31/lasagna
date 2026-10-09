import React from "react";
import type { EnumOption } from "../../api/types";
import HelpTooltip from "./HelpTooltip";

interface Props {
  label: string;
  description?: string;
  helpText?: string;
  configKey?: string;
  value: string;
  onChange: (value: string) => void;
  options: EnumOption[];
}

export default function SelectField({
  label, description, helpText, configKey, value, onChange, options,
}: Props) {
  const selected = options.find((o) => o.value === value);

  return (
    <div className="cfg-select-row">
      <span className="cfg-select-label">
        {label}
        {helpText && <HelpTooltip text={helpText} />}
      </span>
      {description && <div className="cfg-slider-desc">{description}</div>}
      {configKey && <div className="cfg-config-key">{configKey}</div>}
      <select
        className="cfg-select-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {selected?.description && (
        <div className="cfg-select-option-desc">{selected.description}</div>
      )}
    </div>
  );
}
