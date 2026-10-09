import React from "react";
import FieldGroup from "../fields/FieldGroup";
import ToggleSwitch from "../fields/ToggleSwitch";
import SelectField from "../fields/SelectField";
import SliderField from "../fields/SliderField";
import NumberField from "../fields/NumberField";
import type { ParameterDef } from "../../api/types";

interface Props {
  config: Record<string, unknown>;
  onUpdate: (key: string, value: unknown) => void;
  params: ParameterDef[];
}

/** Helper to resolve a param value from config (handles nested keys). */
function getValue(config: Record<string, unknown>, param: ParameterDef): unknown {
  if (param.parent_key) {
    const parent = config[param.parent_key] as Record<string, unknown> | undefined;
    const childKey = param.key.split(".").pop()!;
    return parent?.[childKey] ?? param.default;
  }
  return config[param.key] ?? param.default;
}

function byGroup(params: ParameterDef[], group: string) {
  return params.filter((p) => p.group === group);
}

export default function EssentialTab({ config, onUpdate, params }: Props) {
  const featureToggles = byGroup(params, "feature_toggles");
  const embeddingParams = byGroup(params, "embedding");
  const quickTuning = byGroup(params, "quick_tuning");

  return (
    <div>
      {/* Feature Toggles */}
      <FieldGroup title="Features" description="Turn capabilities on or off to customize your search experience">
        <div className="cfg-toggles-grid">
          {featureToggles.map((p) => (
            <ToggleSwitch
              key={p.key}
              label={p.label}
              description={p.description}
              helpText={p.help_text}
              configKey={p.key}
              checked={getValue(config, p) as boolean}
              onChange={(v) => onUpdate(p.key, v)}
            />
          ))}
        </div>
      </FieldGroup>

      {/* Embedding Model */}
      <FieldGroup title="AI Understanding Model" description="Controls how well your system understands document content">
        {embeddingParams.map((p) => {
          const value = getValue(config, p);
          if (p.type === "enum" && p.enum_options) {
            return (
              <SelectField
                key={p.key}
                label={p.label}
                description={p.description}
                helpText={p.help_text}
                configKey={p.key}
                value={value as string}
                onChange={(v) => {
                  onUpdate(p.key, v);
                  // Auto-fill dimension when model changes
                  if (p.key === "embedding_model") {
                    const dims: Record<string, number> = {
                      "all-MiniLM-L6-v2": 384,
                      "BAAI/bge-small-en-v1.5": 384,
                      "BAAI/bge-base-en-v1.5": 768,
                      "BAAI/bge-large-en-v1.5": 1024,
                      "BAAI/bge-m3": 1024,
                      "intfloat/e5-small-v2": 384,
                      "intfloat/e5-base-v2": 768,
                      "intfloat/e5-large-v2": 1024,
                      "intfloat/multilingual-e5-large": 1024,
                    };
                    if (dims[v]) onUpdate("embedding_dim", dims[v]);
                  }
                }}
                options={p.enum_options}
              />
            );
          }
          if (p.type === "boolean") {
            return (
              <ToggleSwitch
                key={p.key}
                label={p.label}
                description={p.description}
                helpText={p.help_text}
                configKey={p.key}
                checked={value as boolean}
                onChange={(v) => onUpdate(p.key, v)}
              />
            );
          }
          if (p.type === "integer") {
            return (
              <NumberField
                key={p.key}
                label={p.label}
                description={p.description}
                helpText={p.help_text}
                configKey={p.key}
                value={value as number}
                onChange={(v) => onUpdate(p.key, v)}
                min={p.min_value}
                max={p.max_value}
                step={p.step}
              />
            );
          }
          return null;
        })}
      </FieldGroup>

      {/* Quick Tuning */}
      <FieldGroup title="Quick Tuning" description="Adjust the most impactful search settings with simple sliders">
        {quickTuning.map((p) => (
          <SliderField
            key={p.key}
            label={p.label}
            description={p.description}
            helpText={p.help_text}
            configKey={p.key}
            value={getValue(config, p) as number}
            onChange={(v) => onUpdate(p.key, v)}
            min={p.min_value ?? 0}
            max={p.max_value ?? 1}
            step={p.step ?? 1}
          />
        ))}
      </FieldGroup>
    </div>
  );
}
