import React from "react";
import FieldGroup from "../fields/FieldGroup";
import ToggleSwitch from "../fields/ToggleSwitch";
import SelectField from "../fields/SelectField";
import SliderField from "../fields/SliderField";
import NumberField from "../fields/NumberField";
import TextField from "../fields/TextField";
import type { ParameterDef } from "../../api/types";

interface Props {
  config: Record<string, unknown>;
  onUpdate: (key: string, value: unknown) => void;
  params: ParameterDef[];
}

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

function renderParam(p: ParameterDef, config: Record<string, unknown>, onUpdate: (key: string, value: unknown) => void) {
  const value = getValue(config, p);

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
  if (p.type === "enum" && p.enum_options) {
    return (
      <SelectField
        key={p.key}
        label={p.label}
        description={p.description}
        helpText={p.help_text}
        configKey={p.key}
        value={value as string}
        onChange={(v) => onUpdate(p.key, v)}
        options={p.enum_options}
      />
    );
  }
  if (p.type === "float") {
    return (
      <SliderField
        key={p.key}
        label={p.label}
        description={p.description}
        helpText={p.help_text}
        configKey={p.key}
        value={value as number}
        onChange={(v) => onUpdate(p.key, v)}
        min={p.min_value ?? 0}
        max={p.max_value ?? 1}
        step={p.step ?? 0.05}
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
  if (p.type === "string") {
    return (
      <TextField
        key={p.key}
        label={p.label}
        description={p.description}
        helpText={p.help_text}
        configKey={p.key}
        value={value as string}
        onChange={(v) => onUpdate(p.key, v)}
      />
    );
  }
  return null;
}

export default function AdvancedTab({ config, onUpdate, params }: Props) {
  const groups = [
    { key: "agentic_rag", title: "Deep Search Agent", desc: "AI agent that searches multiple times to build thorough answers" },
    { key: "citations", title: "Citations & References", desc: "How source references are tracked and displayed" },
    { key: "caching", title: "Performance Cache", desc: "Speed up repeated searches by caching results" },
    { key: "observability", title: "Monitoring & Logging", desc: "Track performance metrics and debug search behavior" },
    { key: "vector_backend", title: "Vector Database", desc: "Choose where document embeddings are stored" },
  ];

  return (
    <div>
      {groups.map((g) => {
        const groupParams = byGroup(params, g.key);
        if (groupParams.length === 0) return null;
        return (
          <FieldGroup key={g.key} title={g.title} description={g.desc} defaultExpanded={false}>
            {groupParams.map((p) => renderParam(p, config, onUpdate))}
          </FieldGroup>
        );
      })}
    </div>
  );
}
