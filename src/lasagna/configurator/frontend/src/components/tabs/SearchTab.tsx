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

export default function SearchTab({ config, onUpdate, params }: Props) {
  const groups = [
    { key: "search_params", title: "Search Tuning", desc: "Fine-tune how results are found and ranked" },
    { key: "reranking", title: "Result Re-Ranking", desc: "A second pass that puts the best results first" },
    { key: "query_enhancement", title: "Smarter Searching", desc: "Automatically rephrase and break down questions for better results" },
    { key: "chunk_retrieval", title: "How Many Results to Use", desc: "Controls how much content is included in the final answer" },
  ];

  return (
    <div>
      {groups.map((g) => {
        const groupParams = byGroup(params, g.key);
        if (groupParams.length === 0) return null;
        return (
          <FieldGroup key={g.key} title={g.title} description={g.desc}>
            {groupParams.map((p) => renderParam(p, config, onUpdate))}
          </FieldGroup>
        );
      })}
    </div>
  );
}
