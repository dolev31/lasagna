import React from "react";
import HelpTooltip from "./HelpTooltip";

interface Props {
  label: string;
  description?: string;
  helpText?: string;
  configKey?: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  formatValue?: (v: number) => string;
}

export default function SliderField({
  label, description, helpText, configKey, value, onChange, min, max, step = 1, formatValue,
}: Props) {
  const display = formatValue ? formatValue(value) : String(value);

  return (
    <div className="cfg-slider-row">
      <div className="cfg-slider-header">
        <span className="cfg-slider-label">
          {label}
          {helpText && <HelpTooltip text={helpText} />}
        </span>
        <span className="cfg-slider-value">{display}</span>
      </div>
      {description && <div className="cfg-slider-desc">{description}</div>}
      {configKey && <div className="cfg-config-key">{configKey}</div>}
      <input
        type="range"
        className="cfg-slider-input"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="cfg-slider-range">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
