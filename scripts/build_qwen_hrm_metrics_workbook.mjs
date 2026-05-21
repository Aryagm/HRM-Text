import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = process.cwd();
const METRICS_DIR = path.join(ROOT, "outputs", "qwen_hrm_metrics");
const OUT_DIR = path.join(METRICS_DIR);
const OUT_FILE = path.join(OUT_DIR, "qwen_hrm_conversion_metrics.xlsx");

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  const [header, ...data] = rows;
  return data.filter((r) => r.length === header.length).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

async function readRows(file) {
  return parseCSV(await fs.readFile(path.join(METRICS_DIR, file), "utf8")).map((row) => ({
    source_file: file,
    ...row,
  }));
}

function uniqueRuns(rows) {
  const runs = new Map();
  for (const row of rows) {
    const key = `${row.source_file}|${row.model}|${row.run_id || row.mode || ""}`;
    if (!runs.has(key)) {
      runs.set(key, row);
    }
  }
  return [...runs.values()];
}

function asNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : value;
}

function safeCell(value) {
  if (typeof value === "string" && /^[\s]*[=+\-@]/.test(value)) {
    return `'${value}`;
  }
  return value;
}

function modelLabel(model) {
  if (model.includes("0.6B")) return "Qwen3-0.6B-4bit";
  if (model.includes("1.7B")) return "Qwen3-1.7B-4bit";
  if (model.includes("hrm-text-1b-mlx-4bit")) return "HRM-Text-1B MLX 4-bit";
  if (model.includes("hrm-text-1b-mlx-bf16")) return "HRM-Text-1B MLX BF16";
  return model;
}

function modeLabel(row) {
  return row.label === "base" ? "Base" : `HRM H=${row.h_cycles} blend=${row.logit_blend}`;
}

function exactModeLabel(row) {
  if (row.mode === "base") return "Base";
  if (row.mode === "hrm-text") return "HRM-Text";
  return "Qwen-HRM";
}

function exactRunSummary(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.source_file}|${row.model}|${row.mode}`;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }
  return [...groups.values()].map((group) => {
    const first = group[0];
    const elapsed = group.reduce((sum, row) => sum + Number(row.elapsed_s || 0), 0);
    return {
      ...first,
      elapsed_total_s: elapsed,
      avg_case_s: elapsed / group.length,
      correct_cases: group.filter((row) => String(row.is_correct).toLowerCase() === "true").map((row) => row.case).join(", "),
    };
  });
}

function rerankRunSummary(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.source_file}|${row.model}|${row.verifier_mode}|${row.logit_fusion}|${row.scoring}`;
    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key).push(row);
  }
  return [...groups.values()].map((group) => {
    const first = group[0];
    return {
      ...first,
      correct_cases: group.filter((row) => String(row.is_correct).toLowerCase() === "true").map((row) => row.case).join(", "),
    };
  });
}

function simpleRunSummary(rows, keys) {
  const groups = new Map();
  for (const row of rows) {
    const key = keys.map((item) => row[item] || "").join("|");
    if (!groups.has(key)) {
      groups.set(key, row);
    }
  }
  return [...groups.values()];
}

function writeTable(sheet, startCell, headers, rows) {
  const matrix = [headers, ...rows].map((row) => row.map(safeCell));
  const startCol = startCell.match(/[A-Z]+/)[0];
  const startRow = Number(startCell.match(/\d+/)[0]);
  const startColIdx = startCol.charCodeAt(0) - "A".charCodeAt(0);
  const endColIdx = startColIdx + headers.length - 1;
  const endCol = String.fromCharCode("A".charCodeAt(0) + endColIdx);
  const endRow = startRow + matrix.length - 1;
  const range = sheet.getRange(`${startCell}:${endCol}${endRow}`);
  range.values = matrix;
  range.getRow(0).format = {
    fill: { type: "solid", color: "#1F4E78" },
    font: { color: "#FFFFFF", bold: true },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  range.format.wrapText = true;
  range.format.autofitColumns();
  return range;
}

const final06 = await readRows("final_qwen3_0_6b_corrected.csv");
const final17 = await readRows("final_qwen3_1_7b_corrected.csv");
const sweep06 = await readRows("sweep_qwen3_0_6b_expanded_focused.csv").catch(() => []);
const split06 = await readRows("sweep_qwen3_0_6b_split.csv").catch(() => []);
const split17 = await readRows("sweep_qwen3_1_7b_split.csv").catch(() => []);
const tight17 = await readRows("sweep_qwen3_1_7b_split10_tight.csv").catch(() => []);
const fusion06 = await readRows("sweep_qwen3_0_6b_fusion.csv").catch(() => []);
const fusion17 = await readRows("sweep_qwen3_1_7b_fusion.csv").catch(() => []);
const deltaProb06 = await readRows("sweep_qwen3_0_6b_delta_prob_fusion.csv").catch(() => []);
const deltaProb17 = await readRows("sweep_qwen3_1_7b_delta_prob_fusion.csv").catch(() => []);
const exactRows = [
  ...(await readRows("exact_qwen3_0_6b_base.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_0_6b_hrm.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_base.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_base_chat_boxed.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_blend01.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_blend005.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_agreement.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_confidence.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_final_blend.csv").catch(() => [])),
  ...(await readRows("exact_qwen3_1_7b_hrm_chat_boxed_final_confidence.csv").catch(() => [])),
  ...(await readRows("exact_hrm_text_1b_4bit_corrected.csv").catch(() => [])),
  ...(await readRows("exact_hrm_text_1b_bf16_corrected.csv").catch(() => [])),
];
const rerankRows = [
  ...(await readRows("rerank_qwen3_1_7b_raw_base_verifier.csv").catch(() => [])),
  ...(await readRows("rerank_qwen3_1_7b_raw_hrm_agreement_verifier.csv").catch(() => [])),
  ...(await readRows("rerank_qwen3_1_7b_raw_base_answer_likelihood.csv").catch(() => [])),
  ...(await readRows("rerank_qwen3_1_7b_mixed_base_answer_likelihood.csv").catch(() => [])),
];
const arcRows = [
  ...(await readRows("arc_qwen3_0_6b_base_auto_validation10.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_calibrated_options_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_choice_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_label_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_refined_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibrated_options_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_choice_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_label_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_h2_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_split10_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_calibrated_options_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibrated_options_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_calibrated_options_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibrated_options_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_auto_validation10.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibrated_answer_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibrated_options_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_choice_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_label_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_refined_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibrated_answer_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_choice_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_label_text_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibrated_answer_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_label_text_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibrated_answer_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibrated_answer_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibrated_answer_validation299.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_base_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_base_calibrated_options_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_hrm_agreement_calibrated_options_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_base_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_base_calibrated_options_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_hrm_agreement_calibrated_options_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_calibrated_answer_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_calibrated_answer_w17_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_hrm_agreement_calibrated_answer_validation200.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_calibrated_answer_validation570.csv").catch(() => [])),
];
const arcEnsembleRows = [
  ...(await readRows("arc_qwen3_0_6b_margin_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_margin_ensemble_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_margin_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_label_text_margin_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_label_text_margin_ensemble_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_label_text_margin_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_label_text_margin_ensemble_t050_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_label_text_margin_ensemble_t050_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_label_text_margin_ensemble_t050_validation50.csv").catch(() => [])),
];
const arcScoreEnsembleRows = [
  ...(await readRows("arc_qwen3_0_6b_base_score_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_score_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_score_ensemble_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_score_ensemble_validation50.csv").catch(() => [])),
];
const arcCalibrationSweepRows = [
  ...(await readRows("arc_qwen3_0_6b_base_calibration_weight_sweep_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibration_weight_sweep_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_calibration_weight_sweep_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibration_weight_sweep_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_base_calibration_weight_sweep_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_hrm_agreement_calibration_weight_sweep_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibration_weight_sweep_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibration_weight_sweep_validation50.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibration_weight_sweep_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibration_weight_sweep_validation50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_base_calibration_weight_sweep_validation299.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_1_7b_hrm_agreement_calibration_weight_sweep_validation299.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_base_calibration_weight_sweep_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_0_6b_hrm_agreement_calibration_weight_sweep_validation570.csv").catch(() => [])),
  ...(await readRows("arc_easy_qwen3_1_7b_base_calibration_weight_sweep_validation570.csv").catch(() => [])),
];
const arcCalibrationTransferRows = await readRows("arc_calibration_transfer.csv").catch(() => []);
const arcAdapterRows = [
  ...(await readRows("arc_qwen3_1_7b_score_adapter_recurrent_validation50_to_50_100.csv").catch(() => [])),
  ...(await readRows("arc_qwen3_0_6b_score_adapter_recurrent_validation50_to_50_100.csv").catch(() => [])),
];
const finalRows = [...final06, ...final17];
const finalRuns = uniqueRuns(finalRows);
const sweepRuns = uniqueRuns([
  ...sweep06,
  ...split06,
  ...split17,
  ...tight17,
  ...fusion06,
  ...fusion17,
  ...deltaProb06,
  ...deltaProb17,
]);
const exactRuns = exactRunSummary(exactRows);
const rerankRuns = rerankRunSummary(rerankRows);
const arcRuns = simpleRunSummary(arcRows, ["source_file", "model", "mode", "logit_fusion"]);
const arcEnsembleRuns = simpleRunSummary(arcEnsembleRows, ["source_file", "model", "threshold"]);
const arcScoreEnsembleRuns = simpleRunSummary(arcScoreEnsembleRows, ["source_file", "model", "normalize", "weights"]);
const arcCalibrationSweepRuns = simpleRunSummary(arcCalibrationSweepRows, [
  "source_file",
  "model",
  "mode",
  "calibration",
  "sweep_weight",
]);
const arcCalibrationTransferRuns = simpleRunSummary(arcCalibrationTransferRows, [
  "source_file",
  "fit_source_file",
  "eval_source_file",
]);
const arcAdapterRuns = simpleRunSummary(arcAdapterRows, ["source_file", "model", "adapter", "seed"]);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const finalSheet = workbook.worksheets.add("Final Runs");
const casesSheet = workbook.worksheets.add("Case Outcomes");
const sweepSheet = workbook.worksheets.add("Sweep Runs");
const exactSheet = workbook.worksheets.add("Exact Runs");
const exactCasesSheet = workbook.worksheets.add("Exact Outcomes");
const rerankSheet = workbook.worksheets.add("Rerank Runs");
const arcSheet = workbook.worksheets.add("ARC Runs");
const arcEnsembleSheet = workbook.worksheets.add("ARC Ensembles");
const arcScoreEnsembleSheet = workbook.worksheets.add("ARC Score Ensembles");
const arcCalibrationSweepSheet = workbook.worksheets.add("ARC Calib Sweeps");
const arcCalibrationTransferSheet = workbook.worksheets.add("ARC Calib Transfer");
const arcAdapterSheet = workbook.worksheets.add("ARC Adapters");
const notes = workbook.worksheets.add("Notes");

summary.getRange("A1").values = [["Qwen HRM Conversion Metrics"]];
summary.getRange("A1:F1").format = {
  fill: { type: "solid", color: "#17365D" },
  font: { color: "#FFFFFF", bold: true, size: 16 },
};

const summaryRows = [];
for (const model of [...new Set(finalRuns.map((r) => r.model))]) {
  const base = finalRuns.find((r) => r.model === model && r.label === "base");
  const hrm = finalRuns.find((r) => r.model === model && r.label === "hrm");
  if (!base || !hrm) continue;
  summaryRows.push([
    modelLabel(model),
    Number(base.correct),
    Number(hrm.correct),
    Number(base.total),
    Number(hrm.correct) - Number(base.correct),
    Number(hrm.accuracy) - Number(base.accuracy),
    Number(base.elapsed_s),
    Number(hrm.elapsed_s),
    Number(hrm.elapsed_s) / Number(base.elapsed_s),
  ]);
}
writeTable(
  summary,
  "A3",
  ["Model", "Base Correct", "HRM Correct", "Total", "Delta Correct", "Delta Accuracy", "Base s", "HRM s", "Slowdown"],
  summaryRows,
);
summary.getRange("F4:F10").format.numberFormat = "0.0%";
summary.getRange("G4:I10").format.numberFormat = "0.00";

writeTable(
  summary,
  "A9",
  ["Exact Model", "Mode", "Correct", "Total", "Accuracy", "Delta vs Base", "Elapsed s", "Avg Case s", "Source"],
  exactRuns.map((r) => {
    const base = exactRuns.find(
      (candidate) =>
        candidate.model === r.model &&
        candidate.mode === "base" &&
        (candidate.prompt_style || "") === (r.prompt_style || "") &&
        (candidate.chat_template || "") === (r.chat_template || "") &&
        (candidate.enable_thinking || "") === (r.enable_thinking || ""),
    );
    const delta = base ? Number(r.correct) - Number(base.correct) : "";
    return [
      modelLabel(r.model),
      exactModeLabel(r),
      asNumber(r.correct),
      asNumber(r.total),
      asNumber(r.accuracy),
      delta,
      asNumber(r.elapsed_total_s),
      asNumber(r.avg_case_s),
      r.source_file,
    ];
  }),
);
summary.getRange("E10:E25").format.numberFormat = "0.0%";
summary.getRange("G10:H25").format.numberFormat = "0.00";

writeTable(
  finalSheet,
  "A1",
  [
    "Model",
    "Mode",
    "Correct",
    "Total",
    "Accuracy",
    "Elapsed s",
    "H",
    "L",
    "Blend",
    "Alpha L",
    "Alpha H",
    "Beta",
    "Update Mix",
    "Delta Scale",
    "Split",
    "Fusion",
    "Threshold",
  ],
  finalRuns.map((r) => [
    modelLabel(r.model),
    modeLabel(r),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    asNumber(r.elapsed_s),
    asNumber(r.h_cycles),
    asNumber(r.l_cycles),
    asNumber(r.logit_blend),
    asNumber(r.alpha_l),
    asNumber(r.alpha_h),
    asNumber(r.beta_l),
    asNumber(r.update_mix_l),
    asNumber(r.refined_delta_scale),
    asNumber(r.split_index),
    r.logit_fusion || "blend",
    asNumber(r.fusion_threshold || 0),
  ]),
);
finalSheet.getRange("E2:E20").format.numberFormat = "0.0%";
finalSheet.getRange("F2:F20").format.numberFormat = "0.00";

writeTable(
  casesSheet,
  "A1",
  ["Model", "Mode", "Case", "Expected", "Prediction", "Correct", "A", "B", "C", "D"],
  finalRows.map((r) => [
    modelLabel(r.model),
    modeLabel(r),
    r.case,
    r.expected,
    r.prediction,
    r.is_correct,
    asNumber(r.score_A),
    asNumber(r.score_B),
    asNumber(r.score_C),
    asNumber(r.score_D),
  ]),
);
casesSheet.getRange("G2:J200").format.numberFormat = "0.00";

writeTable(
  sweepSheet,
  "A1",
  [
    "Model",
    "Run",
    "Mode",
    "Correct",
    "Total",
    "Accuracy",
    "Elapsed s",
    "H",
    "L",
    "Blend",
    "Alpha L",
    "Alpha H",
    "Beta",
    "Update Mix",
    "Delta Scale",
    "Split",
    "Fusion",
    "Threshold",
    "Source",
  ],
  sweepRuns.map((r) => [
    modelLabel(r.model),
    asNumber(r.run_id),
    modeLabel(r),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    asNumber(r.elapsed_s),
    asNumber(r.h_cycles),
    asNumber(r.l_cycles),
    asNumber(r.logit_blend),
    asNumber(r.alpha_l),
    asNumber(r.alpha_h),
    asNumber(r.beta_l),
    asNumber(r.update_mix_l),
    asNumber(r.refined_delta_scale),
    asNumber(r.split_index),
    r.logit_fusion || "blend",
    asNumber(r.fusion_threshold || 0),
    r.source_file,
  ]),
);
sweepSheet.getRange("F2:F500").format.numberFormat = "0.0%";
sweepSheet.getRange("G2:G500").format.numberFormat = "0.00";

writeTable(
  exactSheet,
  "A1",
  [
    "Model",
    "Mode",
    "Correct",
    "Total",
    "Accuracy",
    "Elapsed s",
    "Avg Case s",
    "Prompt Style",
    "Chat Template",
    "Thinking",
    "Blend",
    "Fusion",
    "Threshold",
    "Fusion After",
    "Correct Cases",
    "Source",
  ],
  exactRuns.map((r) => [
    modelLabel(r.model),
    exactModeLabel(r),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    asNumber(r.elapsed_total_s),
    asNumber(r.avg_case_s),
    r.prompt_style || "",
    r.chat_template || "",
    r.enable_thinking || "",
    r.logit_blend || "",
    r.logit_fusion || "",
    r.fusion_threshold || "",
    r.fusion_after || "",
    r.correct_cases,
    r.source_file,
  ]),
);
exactSheet.getRange("E2:E50").format.numberFormat = "0.0%";
exactSheet.getRange("F2:G50").format.numberFormat = "0.00";

writeTable(
  exactCasesSheet,
  "A1",
  [
    "Model",
    "Mode",
    "Case",
    "Expected",
    "Extracted",
    "Correct",
    "Elapsed s",
    "Prompt Style",
    "Chat Template",
    "Thinking",
    "Blend",
    "Fusion",
    "Threshold",
    "Fusion After",
    "Source",
  ],
  exactRows.map((r) => [
    modelLabel(r.model),
    exactModeLabel(r),
    r.case,
    r.expected,
    r.extracted,
    r.is_correct,
    asNumber(r.elapsed_s),
    r.prompt_style || "",
    r.chat_template || "",
    r.enable_thinking || "",
    r.logit_blend || "",
    r.logit_fusion || "",
    r.fusion_threshold || "",
    r.fusion_after || "",
    r.source_file,
  ]),
);
exactCasesSheet.getRange("G2:G300").format.numberFormat = "0.00";

writeTable(
  rerankSheet,
  "A1",
  [
    "Model",
    "Verifier",
    "Scoring",
    "Fusion",
    "Blend",
    "Correct",
    "Total",
    "Accuracy",
    "Chat Template",
    "Correct Cases",
    "Source",
  ],
  rerankRuns.map((r) => [
    modelLabel(r.model),
    r.verifier_mode,
    r.scoring || "yesno",
    r.logit_fusion,
    asNumber(r.logit_blend),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    r.chat_template,
    r.correct_cases,
    r.source_file,
  ]),
);
rerankSheet.getRange("H2:H50").format.numberFormat = "0.0%";

writeTable(
  arcSheet,
  "A1",
  [
    "Model",
    "ARC Config",
    "Split",
    "Limit",
    "Mode",
    "Answer Scoring",
    "Calibration",
    "Calib Weight",
    "Fusion",
    "Blend",
    "H",
    "L",
    "Split Index",
    "Correct",
    "Total",
    "Accuracy",
    "Elapsed s",
    "Source",
  ],
  arcRuns.map((r) => [
    modelLabel(r.model),
    r.arc_config || "ARC-Challenge",
    r.split,
    asNumber(r.limit),
    r.mode,
    r.answer_scoring || "label",
    r.calibration || "none",
    asNumber(r.calibration_weight || 0),
    r.logit_fusion,
    asNumber(r.logit_blend),
    asNumber(r.h_cycles),
    asNumber(r.l_cycles),
    asNumber(r.split_index),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    asNumber(r.elapsed_s),
    r.source_file,
  ]),
);
arcSheet.getRange("P2:P150").format.numberFormat = "0.0%";
arcSheet.getRange("Q2:Q150").format.numberFormat = "0.00";

writeTable(
  arcEnsembleSheet,
  "A1",
  ["Model", "Split", "Limit", "Threshold", "Correct", "Total", "Accuracy", "Source"],
  arcEnsembleRuns.map((r) => [
    modelLabel(r.model),
    r.split,
    asNumber(r.limit),
    asNumber(r.threshold),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    r.source_file,
  ]),
);
arcEnsembleSheet.getRange("G2:G50").format.numberFormat = "0.0%";

writeTable(
  arcScoreEnsembleSheet,
  "A1",
  ["Model", "Split", "Limit", "Normalize", "Weights", "Correct", "Total", "Accuracy", "Source Files", "Source"],
  arcScoreEnsembleRuns.map((r) => [
    modelLabel(r.model),
    r.split,
    asNumber(r.limit),
    r.normalize,
    r.weights,
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    r.source_files,
    r.source_file,
  ]),
);
arcScoreEnsembleSheet.getRange("H2:H50").format.numberFormat = "0.0%";

writeTable(
  arcCalibrationSweepSheet,
  "A1",
  [
    "Model",
    "Split",
    "Limit",
    "Mode",
    "Calibration",
    "Source Weight",
    "Sweep Weight",
    "Correct",
    "Total",
    "Accuracy",
    "Source",
  ],
  arcCalibrationSweepRuns.map((r) => [
    modelLabel(r.model),
    r.split,
    asNumber(r.limit),
    r.mode,
    r.calibration,
    asNumber(r.source_weight),
    asNumber(r.sweep_weight),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    r.source_file,
  ]),
);
arcCalibrationSweepSheet.getRange("J2:J320").format.numberFormat = "0.0%";

writeTable(
  arcCalibrationTransferSheet,
  "A1",
  [
    "Model",
    "Mode",
    "Calibration",
    "Fit Dataset",
    "Fit Split",
    "Fit Limit",
    "Fit Weight",
    "Fit Correct",
    "Fit Total",
    "Fit Accuracy",
    "Eval Dataset",
    "Eval Split",
    "Eval Limit",
    "Eval Correct",
    "Eval Total",
    "Eval Accuracy",
    "No-Cal Correct",
    "Weight 1 Correct",
    "Oracle Weight",
    "Oracle Correct",
    "Delta vs No-Cal",
    "Delta vs Weight 1",
    "Fit Source",
    "Eval Source",
    "Source",
  ],
  arcCalibrationTransferRuns.map((r) => [
    modelLabel(r.model),
    r.mode,
    r.calibration,
    r.fit_arc_config,
    r.fit_split,
    asNumber(r.fit_limit),
    asNumber(r.fit_weight),
    asNumber(r.fit_correct),
    asNumber(r.fit_total),
    asNumber(r.fit_accuracy),
    r.eval_arc_config,
    r.eval_split,
    asNumber(r.eval_limit),
    asNumber(r.eval_correct),
    asNumber(r.eval_total),
    asNumber(r.eval_accuracy),
    asNumber(r.eval_no_cal_correct),
    asNumber(r.eval_weight_1_correct),
    asNumber(r.eval_oracle_weight),
    asNumber(r.eval_oracle_correct),
    asNumber(r.eval_delta_vs_no_cal),
    asNumber(r.eval_delta_vs_weight_1),
    r.fit_source_file,
    r.eval_source_file,
    r.source_file,
  ]),
);
arcCalibrationTransferSheet.getRange("J2:J50").format.numberFormat = "0.0%";
arcCalibrationTransferSheet.getRange("P2:P50").format.numberFormat = "0.0%";

writeTable(
  arcAdapterSheet,
  "A1",
  [
    "Model",
    "Adapter",
    "Seed",
    "Hidden",
    "Train Split",
    "Train Limit",
    "Train Correct",
    "Train Total",
    "Eval Split",
    "Eval Limit",
    "Eval Correct",
    "Eval Total",
    "Eval Accuracy",
    "Raw Correct",
    "Weight 1 Correct",
    "Delta vs Raw",
    "Delta vs Weight 1",
    "Calibration",
    "Train Source",
    "Eval Source",
    "Source",
  ],
  arcAdapterRuns.map((r) => [
    modelLabel(r.model),
    r.adapter,
    asNumber(r.seed),
    asNumber(r.hidden_size),
    r.train_split,
    asNumber(r.train_limit),
    asNumber(r.train_correct),
    asNumber(r.train_total),
    r.eval_split,
    asNumber(r.eval_limit),
    asNumber(r.eval_correct),
    asNumber(r.eval_total),
    asNumber(r.eval_accuracy),
    asNumber(r.eval_raw_correct),
    asNumber(r.eval_weight1_correct),
    asNumber(r.eval_correct) - asNumber(r.eval_raw_correct),
    asNumber(r.eval_correct) - asNumber(r.eval_weight1_correct),
    r.calibration,
    r.train_source_file,
    r.eval_source_file,
    r.source_file,
  ]),
);
arcAdapterSheet.getRange("M2:M20").format.numberFormat = "0.0%";

notes.getRange("A1:B27").values = [
  ["Item", "Note"],
  ["Benchmark", "Closed-form multiple-choice probe scored by candidate letter log-probability."],
  ["Current wins", "Qwen3-0.6B-4bit improves from 5/17 to 7/17 with split 14; Qwen3-1.7B-4bit improves from 10/17 to 11/17 with split 10."],
  ["Split finding", "The symmetric 14/14 split is best for 0.6B; an earlier split at 10 lower layers is best for 1.7B on the corrected probe."],
  ["Exact-answer probe", "Added a stricter generative probe with exact extraction. Qwen3-0.6B moves from 1/17 base to 2/17 HRM; Qwen3-1.7B is 6/17 for both base and HRM after corrected text-answer scoring."],
  ["Qwen chat exact", "On Qwen3-1.7B chat-template boxed runs, fixed blend scores 5/17 while agreement_blend and confidence_gate recover the 6/17 base score. Use gated fusion for open-ended generation."],
  ["Delayed fusion", "Final-answer-triggered HRM fusion also recovers the 6/17 base score on Qwen3-1.7B chat-template boxed runs. It is safer than full-sequence fixed blending but still not above base."],
  ["Candidate rerank", "Saved-output candidate reranking found an 11/17 oracle union across raw/chat candidates, but naive yes/no and answer-likelihood rerankers selected only 6/17. Need a stronger verifier."],
  ["ARC-Challenge slice", "On ARC-Challenge validation[:50], Qwen3-0.6B is flat at 21/50 for base, fixed blend, and agreement_blend; Qwen3-1.7B base is 37/50, fixed blend is 35/50, and agreement_blend recovers 37/50."],
  ["ARC pure refined", "Pure refined-output scoring collapses on ARC validation[:50]: Qwen3-0.6B drops to 14/50 and Qwen3-1.7B drops to 9/50. This conversion needs conservative fusion; the recurrent path is not a standalone no-training HRM replacement."],
  ["ARC margin ensemble", "A base-margin switch to HRM when base top-two margin <= 0.25 improves Qwen3-0.6B validation[:50] from 21/50 to 23/50, but drops validation[50:100] from 17/50 to 16/50 and Qwen3-1.7B from 37/50 to 36/50. Treat as overfit."],
  ["ARC architecture probes", "Qwen3-0.6B agreement_blend with H=2 and with split_index=10 both remained 21/50 on ARC validation[:50]. Deeper recurrence or a smaller L split did not improve the dataset-backed slice."],
  ["ARC full validation", "On full ARC-Challenge validation, uncalibrated Qwen3-0.6B base and agreement_blend both score 112/299; uncalibrated Qwen3-1.7B base and agreement_blend both score 207/299. The safer gated fusion preserves base but does not improve the full validation split."],
  ["ARC calibrated full validation", "No-training calibration is the first robust ARC gain: Qwen3-0.6B option-prior calibration improves base/agreement_blend to 121/299, and Qwen3-1.7B answer-prior calibration improves base/agreement_blend to 211/299."],
  ["ARC calibration sweep", "Post-hoc calibration-weight sweeps from saved raw/prior scores show Qwen3-0.6B is best at weight 1.0 on full validation, while Qwen3-1.7B answer-prior weight 1.7 reaches 216/299 for both base and agreement_blend. Treat 1.7 as tuned on validation until tested elsewhere."],
  ["ARC calibration transfer", "Transfer checks fit the calibration weight on one saved score file and evaluate a separate file. ARC-Challenge validation[:50] to validation[50:100] gives +3 for Qwen3-0.6B and +2 for Qwen3-1.7B. ARC-Challenge full to ARC-Easy full gives +7 for Qwen3-0.6B but +0 for Qwen3-1.7B."],
  ["ARC-Easy transfer", "On full ARC-Easy validation, Qwen3-0.6B option-prior calibration transfers from 348/570 to 355/570 at weight 1.0; a post-hoc weight sweep reaches 365/570 at weight 0.8 for both base and agreement_blend. Qwen3-1.7B answer-prior calibration is not robust: weight 1.0 is 488/570 and the best swept weight 0.4 reaches only 490/570."],
  ["ARC auto policy", "benchmarks/qwen_arc_probe.py now supports --calibration auto. It applies the conservative transferred policy: Qwen3-0.6B uses options_prior at weight 1.0; Qwen3-1.7B uses answer_prior at weight 1.0 and avoids the ARC-Challenge-tuned 1.7 weight."],
  ["ARC auto smoke", "Tiny MLX smoke runs verified --calibration auto resolves to options_prior@1.0 for Qwen3-0.6B and answer_prior@1.0 for Qwen3-1.7B on ARC-Challenge validation[:10]."],
  ["ARC score adapter", "A tiny recurrent adapter trained on saved Qwen option score surfaces is a partial positive signal: Qwen3-1.7B improves validation[50:100] from 33/50 raw and 35/50 fixed calibration to 37/50, while Qwen3-0.6B underperforms fixed calibration at 20/50 vs 22/50."],
  ["ARC scoring surfaces", "Choice-text scoring is much worse than answer-letter scoring on ARC slices. Qwen3-1.7B label+text scoring ties the 37/50 first-slice base score, and a margin switch reaches 38/50 on validation[:50] but only ties base at 33/50 on validation[50:100]. Treat as exploratory, not solid improvement."],
  ["HRM-Text comparison", "The original HRM-Text exact run used the wrong final-answer prompt/extraction and too small a token cap. Corrected boxed-prompt runs score 16/17 for both 4-bit and BF16."],
  ["Cost", "HRM is slower because it adds warm split and recurrent passes per scored candidate/token."],
  ["Correction", "Earlier sequence answer was corrected from 80 to 67 before final runs."],
  ["Interpretation", "The MC probe is a fast regression test, not an optimal reasoning benchmark. The exact probe is better but still too small for broad HRM-level claims."],
  ["Fusion finding", "Confidence-gated, agreement-gated, probability-space, and extrapolated-delta fusion did not beat fixed logit blending on the current probe."],
  ["Next", "Need larger held-out probe and a stronger no-training verifier/reranker strategy, or light adapter calibration, for stronger claims."],
];
notes.getRange("A1:B1").format = {
  fill: { type: "solid", color: "#1F4E78" },
  font: { color: "#FFFFFF", bold: true },
};
notes.getRange("A1:B27").format.wrapText = true;
notes.getRange("A1:B27").format.autofitColumns();

for (const sheet of [
  summary,
  finalSheet,
  casesSheet,
  sweepSheet,
  exactSheet,
  exactCasesSheet,
  rerankSheet,
  arcSheet,
  arcEnsembleSheet,
  arcScoreEnsembleSheet,
  arcCalibrationSweepSheet,
  arcCalibrationTransferSheet,
  arcAdapterSheet,
  notes,
]) {
  sheet.getRange("A1:Z300").format.verticalAlignment = "top";
}

await fs.mkdir(OUT_DIR, { recursive: true });
await workbook.inspect({ kind: "table", range: "Summary!A1:I6", include: "values,formulas", tableMaxRows: 10, tableMaxCols: 10 });
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);
await workbook.render({ sheetName: "Summary", range: "A1:I16", scale: 2 });
await workbook.render({ sheetName: "Exact Runs", range: "A1:N14", scale: 2 });
await workbook.render({ sheetName: "Rerank Runs", range: "A1:K8", scale: 2 });
await workbook.render({ sheetName: "ARC Runs", range: "A1:R60", scale: 2 });
await workbook.render({ sheetName: "ARC Ensembles", range: "A1:H11", scale: 2 });
await workbook.render({ sheetName: "ARC Score Ensembles", range: "A1:J5", scale: 2 });
await workbook.render({ sheetName: "ARC Calib Sweeps", range: "A1:K316", scale: 2 });
await workbook.render({ sheetName: "ARC Calib Transfer", range: "A1:Y9", scale: 2 });
await workbook.render({ sheetName: "ARC Adapters", range: "A1:U4", scale: 2 });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUT_FILE);
console.log(OUT_FILE);
