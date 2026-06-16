export type AdapterName = "mock" | "ollama" | "transformers" | "nnsight" | "pytorch" | "openai" | "anthropic" | "gemini";
export type OutputPolicy = "raw" | "redacted";
export type Quantization = "none" | "4bit" | "8bit";
export type InterventionAction = "none" | "mute" | "scale" | "boost";
export type InterventionTarget = "layer" | "head" | "feature";

export interface InterventionConfig {
  enabled: boolean;
  target_type: InterventionTarget;
  layer: number;
  head: number | null;
  action: InterventionAction;
  scale: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface BenchmarkCase {
  id: string;
  category: string;
  prompt: string;
  expected_refusal: boolean;
}

export interface BenchmarkResult extends BenchmarkCase {
  peak: number;
  state: string;
  refused: boolean | null;
  text: string;
  errors: string[];
  elapsed: number;
  verdict: "PASS" | "FAIL:bypass" | "FAIL:overblock" | "ERROR";
}

export type JailbreakMode = "default" | "advanced" | "broker_math" | "broker_full" | "broker_half" | "pid_control" | "orthogonal_steer" | "activation_patch" | "gradient_steer" | "surgical" | "caa_dynamic" | "token_window" | "progressive" | "mlp_clamp";

export interface CompareResult {
  mode: JailbreakMode | "baseline";
  jailbreak: boolean;
  peak: number;
  state: string;
  refused: boolean | null;
  text: string;
  elapsed: number;
  errors: string[];
}

export type PromptCraftType = "none" | "base64" | "rot13" | "leetspeak" | "dan" | "developer" | "crescendo" | "aim" | "indirect_injection" | "many_shot" | "gcg_suffix" | "virtualization";

export interface RunRequest {
  prompt: string;
  adapter: AdapterName;
  model: string;
  api_key?: string;
  response_language: "en" | "tr" | "de" | "es";
  output_policy: OutputPolicy;
  max_new_tokens: number;
  temperature: number;
  prompt_craft: PromptCraftType;
  jailbreak: boolean;
  jailbreak_mode: "default" | "advanced" | "broker_math" | "broker_full" | "broker_half" | "pid_control" | "orthogonal_steer" | "activation_patch" | "gradient_steer" | "surgical" | "caa_dynamic" | "token_window" | "progressive" | "mlp_clamp";
  use_mlp_ablation: boolean;
  use_helpfulness_boost: boolean;
  use_norm_regulation: boolean;
  quantization: Quantization;
  intervention: InterventionConfig;
  interventions: InterventionConfig[];
  history: ChatTurn[];
}

export interface ModelInfo {
  id: string;
  label: string;
  adapter: AdapterName;
  description: string;
}

export interface StreamEvent<T = Record<string, unknown>> {
  type: string;
  ts: number;
  data: T;
}

export interface LayerMetric {
  layer: number;
  activity: number;
  safety: number;
  uncertainty: number;
}

export interface Candidate {
  token: string;
  prob: number;
}

export interface ConceptScore {
  name: string;
  score: number;
}

export interface LensToken {
  layer: number;
  token: string;
  prob: number;
}

export interface SafetyTrace {
  score: number;
  state: string;
  first_trigger_layer: number | null;
  locked_layer: number | null;
  notes: string;
}

export interface AttentionTrace {
  tokens: string[];
  weights: number[];
}

export interface HeadScore {
  head: number;
  score: number;
}

export interface HeadMapLayer {
  layer: number;
  heads: HeadScore[];
}

export interface HeadMap {
  n_heads: number;
  layers: HeadMapLayer[];
}

export interface ThinkPhaseSpan {
  phase: "think" | "answer";
  start: number;
  end: number;
  steps: number;
  layer_avg: number[];
}

export interface ThinkPhaseSummary {
  has_think: boolean;
  think_steps: number;
  answer_steps: number;
  think_avg: number[];
  answer_avg: number[];
  delta: number[];
  dominant_think_layers: number[];
  spans: ThinkPhaseSpan[];
}

export interface BlackBoxMetrics {
  prompt_eval_count?: number;
  eval_count?: number;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_duration?: number;
  eval_duration?: number;
}
