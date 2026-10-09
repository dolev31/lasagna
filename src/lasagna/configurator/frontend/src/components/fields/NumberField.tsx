import React from "react";
import HelpTooltip from "./HelpTooltip";

interface Props {
  label: string;
  description?: string;
  helpText?: string;
  configKey?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export default function NumberField({
  label, description, helpText, configKey, value, onChange, min, max, step,
}: Props) {
  return (
    <div className="cfg-number-row">
      <div className="cfg-number-header">
        <span className="cfg-number-label">
          {label}
          {helpText && <HelpTooltip text={helpText} />}
        </span>
      </div>
      {description && <div className="cfg-slider-desc">{description}</div>}
      {configKey && <div className="cfg-config-key">{configKey}</div>}
      <input
        type="number"
        className="cfg-number-input"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
      />
    </div>
  );
}
