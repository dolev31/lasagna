// ---------------------------------------------------------------------------
// Schema types (from GET /configurator/api/schema)
// ---------------------------------------------------------------------------

export interface EnumOption {
  value: string;
  label: string;
  description: string;
}

export interface ParameterDef {
  key: string;
  type: "boolean" | "integer" | "float" | "string" | "enum" | "object";
  label: string;
  description: string;
  help_text: string;
  tab: "essential" | "search" | "processing" | "advanced";
  group: string;
  default: unknown;
  requires_reindex: boolean;
  hidden?: boolean;
  min_value?: number;
  max_value?: number;
  step?: number;
  enum_options?: EnumOption[];
  parent_key?: string;
}

export interface TabDef {
  key: string;
  label: string;
  description: string;
}

export interface GroupDef {
  key: string;
  label: string;
  tab: string;
}

export interface SchemaResponse {
  tabs: TabDef[];
  groups: GroupDef[];
  parameters: ParameterDef[];
}

// ---------------------------------------------------------------------------
// Profile / config types (from existing knowledge API)
// ---------------------------------------------------------------------------

export interface RAGProfile {
  name: string;
  description: string;
  config: Record<string, unknown>;
}

export interface KnowledgeConfigResponse {
  schema_version: number;
  defaults_source: string;
  profiles: RAGProfile[];
  active_profile: string;
  session_overrides: Record<string, unknown>;
  rag_enabled: boolean;
}

// ---------------------------------------------------------------------------
// Validation types (from POST /configurator/api/validate)
// ---------------------------------------------------------------------------

export interface ValidationIssue {
  key: string;
  message: string;
}

export interface ValidationResponse {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  requires_reindex: boolean;
  reindex_keys: string[];
}
