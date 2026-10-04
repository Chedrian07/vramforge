// Korean display labels for contract enums. Typed as Record<Enum, string> so that a contract change
// (lib/api/schema.d.ts) that adds a value fails typecheck until it gets a label.
import type {
  AllocationCategory,
  Branch,
  DataPreservation,
  DatasetFormat,
  Evidence,
  EvidenceLevel,
  HardwareFit,
  JobStatus,
  Objective,
  Phase,
  PreservationCheckName,
  ScanCoverage,
  Schemas,
  Severity,
  Strategy,
  TrainingReadiness,
} from "@/lib/api/types";

export const OBJECTIVE_LABEL: Record<Objective, string> = { sft: "SFT", dpo: "DPO", grpo: "GRPO" };

export const STRATEGY_LABEL: Record<Strategy, string> = {
  full: "Full",
  lora: "LoRA",
  qlora: "QLoRA",
};

export const JOB_STATUS_LABEL: Record<JobStatus, string> = {
  QUEUED: "대기 중",
  RESOLVING: "출처 확인 중",
  INSPECTING: "구조 확인 중",
  NEEDS_INPUT: "입력 필요",
  TOKENIZING: "데이터 토큰화 중",
  VALIDATING_DATA: "데이터 검증 중",
  PLANNING_BATCHES: "배치 분석 중",
  ESTIMATING: "메모리 산정 중",
  COMPLETED: "완료",
  CANCEL_REQUESTED: "취소 요청됨",
  CANCELLED: "취소됨",
  FAILED: "실패",
  PARTIAL: "부분 결과",
};

/** The four user-facing stages of plan.md §3.1 and the job states each one covers. */
export const PROGRESS_STEPS: ReadonlyArray<{
  key: string;
  label: string;
  statuses: readonly JobStatus[];
}> = [
  { key: "structure", label: "구조 확인", statuses: ["RESOLVING", "INSPECTING"] },
  { key: "tokenize", label: "데이터 토큰화", statuses: ["TOKENIZING", "VALIDATING_DATA"] },
  { key: "batch", label: "배치 분석", statuses: ["PLANNING_BATCHES"] },
  { key: "memory", label: "메모리 산정", statuses: ["ESTIMATING"] },
];

export const SCAN_COVERAGE_LABEL: Record<ScanCoverage, string> = {
  not_started: "시작 전",
  partial: "부분",
  complete: "전체 완료",
  failed: "실패",
};

export const DATA_PRESERVATION_LABEL: Record<DataPreservation, string> = {
  pending: "확인 전",
  verified: "보존 확인",
  violated: "위반",
  unknown: "알 수 없음",
};

export const READINESS_LABEL: Record<TrainingReadiness, string> = {
  ready: "준비됨",
  conditional: "조건부",
  unsupported: "미지원",
};

export const EVIDENCE_LEVEL_LABEL: Record<EvidenceLevel, string> = {
  metadata_only: "메타데이터만",
  analytic: "정적 추정",
  calibrated: "보정된 추정",
  measured: "실측",
};

export const HARDWARE_FIT_LABEL: Record<HardwareFit, string> = {
  not_evaluated: "판정 안 함",
  expected_fit: "예상 적합",
  low_margin: "여유 부족",
  exceeds: "용량 초과",
  unknown: "판정 보류",
};

export type FitReason = Schemas["HardwareFitResult"]["reason"];

/** plan.md §10.3 display wording per fit reason. */
export const FIT_REASON_LABEL: Record<FitReason, string> = {
  not_evaluated: "용량만 표시, 적합 판정 없음",
  floor_exceeds_capacity: "확정된 구성만으로 용량 초과",
  high_exceeds_capacity: "예상 용량 초과, 실측/설정 검토 필요",
  margin_insufficient: "여유 부족",
  fits_with_margin: "선택한 가정에서 예상 적합",
  unknown_components: "판정 보류 (미확정 footprint)",
  unsupported: "판정 보류 (지원 미확정 backend)",
  load_budget_insufficient: "모델 로딩 단계 여유 부족",
  scan_incomplete: "판정 보류 (전체 데이터 스캔 미완료)",
};

export const EVIDENCE_LABEL: Record<Evidence, string> = {
  analytic: "분석식",
  calibrated: "보정",
  measured: "실측",
  assumption: "가정",
  unknown: "미상",
};

export const SEVERITY_LABEL: Record<Severity, string> = {
  info: "정보",
  warning: "주의",
  error: "오류",
};

export const PHASE_LABEL: Record<Phase, string> = {
  MODEL_LOAD_AND_QUANTIZE: "모델 로딩·양자화",
  REFERENCE_PRECOMPUTE: "reference 사전 계산",
  ROLLOUT_PREFILL_AND_DECODE: "rollout 생성",
  REWARD: "reward",
  POLICY_FORWARD_BACKWARD: "policy forward·backward",
  OPTIMIZER_STEP: "optimizer step",
  WEIGHT_SYNC: "가중치 동기화",
  EVALUATION: "평가",
  CHECKPOINT_SAVE_OR_CONSOLIDATE: "체크포인트 저장",
};

export const CATEGORY_LABEL: Record<AllocationCategory, string> = {
  weights_base: "기본 가중치",
  weights_adapter: "adapter 가중치",
  weights_other_models: "다른 모델 가중치",
  gradients: "gradient",
  optimizer_states: "optimizer state",
  master_weights: "master weight",
  saved_activations: "저장 activation",
  recompute_working_set: "재계산 작업 집합",
  logits_and_loss: "logits·loss",
  generation_cache: "생성 KV cache",
  recurrent_state: "recurrent state",
  rollout_buffers: "rollout 버퍼",
  load_transient: "로딩 임시",
  workspace: "workspace",
  communication_buffers: "통신 버퍼",
  allocator_slack: "allocator 여유",
  non_framework: "CUDA context 등",
};

export const BRANCH_LABEL: Record<Branch, string> = {
  prompt: "prompt",
  completion: "completion",
  sequence: "전체 시퀀스",
  loss_tokens: "loss token",
  chosen: "chosen 응답",
  rejected: "rejected 응답",
  chosen_sequence: "prompt + chosen",
  rejected_sequence: "prompt + rejected",
  pair_max: "pair 최대",
};

export const PRESERVATION_CHECK_LABEL: Record<PreservationCheckName, string> = {
  full_read: "전체 읽기",
  no_length_drop: "길이에 따른 삭제 없음",
  no_split_or_concat: "분할·연결 없음",
  template_content_preserved: "템플릿 content 보존",
  last_batch_included: "마지막 batch 포함",
  packing_safe: "packing 안전성",
  context_within_limit: "context 상한 이내",
  eval_scope_consistent: "평가 범위 일치",
};

export const DATASET_FORMAT_LABEL: Record<DatasetFormat, string> = {
  auto: "자동 감지",
  preference: "preference (prompt · chosen · rejected)",
  prompt_completion: "prompt-completion",
  prompt_only: "prompt only",
  messages: "messages (대화)",
  text: "text (언어 모델링)",
};

export const REFERENCE_STRATEGY_LABEL: Record<Schemas["ReferenceStrategy"], string> = {
  auto: "자동",
  frozen_base_switch: "adapter off (frozen base 전환)",
  standalone_model: "별도 reference 모델",
  precomputed_log_probs: "log-prob 사전 계산",
};

export const REWARD_KIND_LABEL: Record<Schemas["RewardKind"], string> = {
  unspecified: "미지정",
  cpu_rule: "CPU 규칙 함수",
  remote: "원격 reward",
  local_model: "로컬 reward 모델",
};

export const ROLLOUT_BACKEND_LABEL: Record<Schemas["RolloutBackend"], string> = {
  transformers_shared_policy: "Transformers 공유 policy",
  vllm_colocate: "vLLM colocate",
  vllm_server: "vLLM server",
};

export const ATTENTION_BACKEND_LABEL: Record<Schemas["AttentionBackend"], string> = {
  auto: "자동",
  sdpa: "SDPA",
  eager: "eager",
  flash_attention_2: "FlashAttention 2",
};

export const LINEAR_ATTENTION_KERNEL_LABEL: Record<Schemas["LinearAttentionKernel"], string> = {
  auto: "자동 (환경 프로필)",
  torch_fallback: "torch fallback",
  fla: "fla (flash-linear-attention)",
};

export const LOSS_KERNEL_LABEL: Record<Schemas["LossKernel"], string> = {
  auto: "자동",
  standard: "standard",
  chunked: "chunked",
  liger: "Liger",
};

export const OPTIMIZER_LABEL: Record<Schemas["OptimizerName"], string> = {
  adamw_torch: "AdamW (torch)",
  adamw_torch_fused: "AdamW (torch fused)",
  adamw_8bit: "AdamW 8-bit",
  paged_adamw_8bit: "Paged AdamW 8-bit",
};

export const PRECISION_LABEL: Record<Schemas["Precision"], string> = {
  auto: "자동",
  bf16: "bf16",
  fp16: "fp16",
  fp32: "fp32",
};

export const LOAD_DTYPE_LABEL: Record<Schemas["LoadDtype"], string> = {
  auto: "자동 (프로필 preset)",
  bfloat16: "bfloat16",
  float16: "float16",
  float32: "float32",
};

export const COMPUTE_DTYPE_LABEL: Record<Schemas["ComputeDtype"], string> = {
  auto: "자동",
  bfloat16: "bfloat16",
  float16: "float16",
  float32: "float32",
};

export const QUANT_FORMAT_LABEL: Record<Schemas["QuantFormat"], string> = {
  nf4: "NF4",
  fp4: "FP4",
};

export const LORA_BIAS_LABEL: Record<Schemas["LoraBias"], string> = {
  none: "none",
  all: "all",
  lora_only: "lora_only",
};

export const LOADING_SCOPE_LABEL: Record<Schemas["LoadingScope"], string> = {
  auto_verified: "자동 (검증된 범위)",
  full_checkpoint: "전체 checkpoint",
  text_only: "텍스트 모듈만",
};

export const EMPTY_SYSTEM_POLICY_LABEL: Record<Schemas["EmptySystemPolicy"], string> = {
  omit: "빈 system 생략",
  keep: "빈 system 유지",
};

export const SOURCE_TYPE_LABEL: Record<Schemas["SourceType"], string> = {
  huggingface: "Hugging Face",
  local: "로컬 (서버 기준)",
  upload: "업로드 파일",
};

/** Unit of one microbatch row per objective (schemas/request.py TrainingConfig comment). */
export const MICROBATCH_UNIT: Record<Objective, string> = {
  sft: "샘플",
  dpo: "pair",
  grpo: "update completion",
};

/** How each objective transforms the mapped columns (plan.md §7.2). */
export const OBJECTIVE_TRANSFORMATION: Record<Objective, string> = {
  sft: "SFT: preference 데이터는 prompt + chosen만 학습하고 rejected는 사용하지 않습니다 (선택한 데이터 변환이며 truncation이 아님). prompt-completion·messages·text 데이터는 매핑한 내용을 그대로 학습합니다.",
  dpo: "DPO: prompt + chosen, prompt + rejected 두 branch를 각각 그대로 사용합니다.",
  grpo: "GRPO: prompt만 사용합니다. 응답 길이는 completion budget 시나리오로 따로 계산합니다.",
};

export function labelOr<K extends string>(map: Record<K, string>, key: K | null | undefined, fallback = "—"): string {
  return key == null ? fallback : (map[key] ?? key);
}
