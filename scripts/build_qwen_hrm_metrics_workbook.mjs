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
  ...(await readRows("exact_hrm_text_1b_4bit_corrected.csv").catch(() => [])),
  ...(await readRows("exact_hrm_text_1b_bf16_corrected.csv").catch(() => [])),
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

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const finalSheet = workbook.worksheets.add("Final Runs");
const casesSheet = workbook.worksheets.add("Case Outcomes");
const sweepSheet = workbook.worksheets.add("Sweep Runs");
const exactSheet = workbook.worksheets.add("Exact Runs");
const exactCasesSheet = workbook.worksheets.add("Exact Outcomes");
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
  ["Exact Model", "Mode", "Correct", "Total", "Accuracy", "Delta vs Base", "Elapsed s", "Avg Case s"],
  exactRuns.map((r) => {
    const base = exactRuns.find((candidate) => candidate.model === r.model && candidate.mode === "base");
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
  ["Model", "Mode", "Correct", "Total", "Accuracy", "Elapsed s", "Avg Case s", "Correct Cases", "Source"],
  exactRuns.map((r) => [
    modelLabel(r.model),
    exactModeLabel(r),
    asNumber(r.correct),
    asNumber(r.total),
    asNumber(r.accuracy),
    asNumber(r.elapsed_total_s),
    asNumber(r.avg_case_s),
    r.correct_cases,
    r.source_file,
  ]),
);
exactSheet.getRange("E2:E50").format.numberFormat = "0.0%";
exactSheet.getRange("F2:G50").format.numberFormat = "0.00";

writeTable(
  exactCasesSheet,
  "A1",
  ["Model", "Mode", "Case", "Expected", "Extracted", "Correct", "Elapsed s", "Source"],
  exactRows.map((r) => [
    modelLabel(r.model),
    exactModeLabel(r),
    r.case,
    r.expected,
    r.extracted,
    r.is_correct,
    asNumber(r.elapsed_s),
    r.source_file,
  ]),
);
exactCasesSheet.getRange("G2:G300").format.numberFormat = "0.00";

notes.getRange("A1:B11").values = [
  ["Item", "Note"],
  ["Benchmark", "Closed-form multiple-choice probe scored by candidate letter log-probability."],
  ["Current wins", "Qwen3-0.6B-4bit improves from 5/17 to 7/17 with split 14; Qwen3-1.7B-4bit improves from 10/17 to 11/17 with split 10."],
  ["Split finding", "The symmetric 14/14 split is best for 0.6B; an earlier split at 10 lower layers is best for 1.7B on the corrected probe."],
  ["Exact-answer probe", "Added a stricter generative probe with exact extraction. Qwen3-0.6B moves from 1/17 base to 2/17 HRM; Qwen3-1.7B is 6/17 for both base and HRM after corrected text-answer scoring."],
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
notes.getRange("A1:B11").format.wrapText = true;
notes.getRange("A1:B11").format.autofitColumns();

for (const sheet of [summary, finalSheet, casesSheet, sweepSheet, exactSheet, exactCasesSheet, notes]) {
  sheet.getRange("A1:Q300").format.verticalAlignment = "top";
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
await workbook.render({ sheetName: "Exact Runs", range: "A1:I9", scale: 2 });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUT_FILE);
console.log(OUT_FILE);
