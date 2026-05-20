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
    const key = `${row.source_file}|${row.model}|${row.run_id}`;
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

function modelLabel(model) {
  if (model.includes("0.6B")) return "Qwen3-0.6B-4bit";
  if (model.includes("1.7B")) return "Qwen3-1.7B-4bit";
  return model;
}

function modeLabel(row) {
  return row.label === "base" ? "Base" : `HRM H=${row.h_cycles} blend=${row.logit_blend}`;
}

function writeTable(sheet, startCell, headers, rows) {
  const matrix = [headers, ...rows];
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
const finalRows = [...final06, ...final17];
const finalRuns = uniqueRuns(finalRows);
const sweepRuns = uniqueRuns([...sweep06, ...split06, ...split17, ...tight17, ...fusion06, ...fusion17]);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const finalSheet = workbook.worksheets.add("Final Runs");
const casesSheet = workbook.worksheets.add("Case Outcomes");
const sweepSheet = workbook.worksheets.add("Sweep Runs");
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

notes.getRange("A1:B9").values = [
  ["Item", "Note"],
  ["Benchmark", "Closed-form multiple-choice probe scored by candidate letter log-probability."],
  ["Current wins", "Qwen3-0.6B-4bit improves from 5/17 to 7/17 with split 14; Qwen3-1.7B-4bit improves from 10/17 to 11/17 with split 10."],
  ["Split finding", "The symmetric 14/14 split is best for 0.6B; an earlier split at 10 lower layers is best for 1.7B on the corrected probe."],
  ["Cost", "HRM is slower because it adds warm split and recurrent passes per scored candidate/token."],
  ["Correction", "Earlier sequence answer was corrected from 80 to 67 before final runs."],
  ["Interpretation", "No-training recurrence helps the smaller model on this probe, but this is not yet broad HRM-level performance."],
  ["Fusion finding", "Confidence-gated and agreement-gated logit fusion did not preserve the gains; fixed logit blending remains best on the current probe."],
  ["Next", "Need larger held-out probe and a stronger no-training verifier/reranker strategy, or light adapter calibration, for stronger claims."],
];
notes.getRange("A1:B1").format = {
  fill: { type: "solid", color: "#1F4E78" },
  font: { color: "#FFFFFF", bold: true },
};
notes.getRange("A1:B9").format.wrapText = true;
notes.getRange("A1:B9").format.autofitColumns();

for (const sheet of [summary, finalSheet, casesSheet, sweepSheet, notes]) {
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
await workbook.render({ sheetName: "Summary", range: "A1:I8", scale: 2 });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(OUT_FILE);
console.log(OUT_FILE);
