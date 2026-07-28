"""
Generate a browser-based annotation tool for the gold set.

REVISION NOTE -- why this was rebuilt:
  The first version pre-filled every field from the rules extractor and put a
  large "Save & next" button beside them. The result was 22 postings recorded
  in 22 seconds: the extractor's own output, saved under a human's name. That
  is worse than no gold set, because every downstream metric would inherit it
  and the rules would score near 100% by construction.

  Four guardrails follow from that failure:

    NO PRE-FILLS      You start blank. Anchoring to the extractor's answer is
                      exactly what a gold set must not do. (A toggle exists
                      for comparison, off by default, and its use is recorded
                      per posting so the labels stay auditable.)

    ARRANGEMENT GATE  You cannot advance without choosing one. It is the field
                      with no reference labels at all, so hand-labelling is
                      the only way it will ever be measured -- and it was
                      null on all 22 records of the first attempt.

    TIME RECORDED     Seconds-on-posting is stored with every label. If the
                      median is 15 seconds, the annotation is not credible,
                      and that should be visible rather than hidden.

    SECTION HEADINGS  Qualification headings are detected and made prominent.
                      Most of a 5,800-character posting is company blurb; the
                      requirements live in two lists near the bottom. Making
                      them findable is the difference between reading and
                      skimming.

Run:  python src/build_annotator.py
Then: open docs/annotate.html
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (  # noqa: E402
    _SKILL_PATTERNS,
    extract_skills,
    extract_years_experience,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# Kept to technologies that actually appear in data-role postings. A longer
# list slows every annotation down for entries that will never be ticked.
SKILL_VOCAB = [
    "SQL", "Python", "R", "Excel", "Tableau", "Power BI", "Looker",
    "Snowflake", "Redshift", "BigQuery", "Databricks", "dbt", "Airflow",
    "Spark", "Hadoop", "Kafka", "AWS", "Azure", "GCP", "Java", "Scala",
    "SAS", "SPSS", "Git", "Docker", "Kubernetes", "ETL", "Machine Learning",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
    "PostgreSQL", "MySQL", "MongoDB", "Oracle", "Alteryx", "VBA",
    "JavaScript", "Salesforce", "SSIS", "SSRS", "Sigma", "Mode", "Hex",
]

# Headings that introduce a requirements list. Detected so they can be made
# visually prominent -- these two or three lines are where the answer lives.
_HEADING = re.compile(
    r"(?:^|(?<=[.!?\s]))("
    r"(?:minimum |basic |required |preferred |desired |additional |key )?"
    r"(?:qualifications?|requirements?|skills?(?: and experience)?|"
    r"what (?:you|we)[^.\n]{0,40}|"
    r"nice[- ]to[- ]haves?|"
    r"who you are|about you|the ideal candidate"
    r")\b:?)",
    re.I,
)

_HIGHLIGHT = [
    ("skill", re.compile("|".join(_SKILL_PATTERNS.values()), re.I)),
    ("hedge-req", re.compile(
        r"\brequired\b|\bmust have\b|\bmust possess\b|\bessential\b|\bmandatory\b", re.I)),
    ("hedge-pref", re.compile(
        r"\bpreferred\b|\ba plus\b|\bnice to have\b|\bdesirable\b|\bbonus\b|"
        r"\bfamiliarity with\b|\bideally\b", re.I)),
    ("arrangement", re.compile(r"\bremote\b|\bhybrid\b|\bon-?site\b|\bin-?office\b", re.I)),
    ("years", re.compile(r"\d{1,2}\+?\s*(?:-|–|to)?\s*\d{0,2}\s*\+?\s*(?:year|yr)s?", re.I)),
]


def render_description(text: str) -> str:
    """Escape, mark headings, highlight signals, preserve line breaks."""
    out = html.escape(text)

    # Headings first, with a sentinel so later patterns cannot match inside
    # the tag attributes they introduce.
    out = _HEADING.sub(lambda m: f"\x00HEAD\x01{m.group(1)}\x02", out)

    for css_class, pattern in _HIGHLIGHT:
        out = pattern.sub(lambda m: f'<mark class="{css_class}">{m.group(0)}</mark>', out)

    out = (out.replace("\x00HEAD\x01", '<span class="heading">')
              .replace("\x02", "</span>"))
    return out.replace("\n", "<br>")


def build_records() -> list[dict]:
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT job_id, title, company_name, location, role_family,
               description, difficulty, desc_len
        FROM '{PROCESSED / "gold_seed.parquet"}'
        ORDER BY role_family, job_id
    """).fetchdf()

    records = []
    for row in df.itertuples(index=False):
        required, preferred, _ = extract_skills(row.description)
        years, years_evidence = extract_years_experience(row.description)

        records.append({
            "job_id": int(row.job_id),
            "title": row.title,
            "company": row.company_name,
            "location": row.location,
            "role_family": row.role_family,
            "difficulty": round(float(row.difficulty), 1),
            "chars": int(row.desc_len),
            "html": render_description(row.description),
            # Hidden by default. Available behind a toggle purely so you can
            # see what the extractor thought AFTER deciding for yourself.
            "guess_required": required,
            "guess_preferred": preferred,
            "guess_years": int(years) if years is not None else None,
            "guess_years_evidence": years_evidence or "",
        })
    return records


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gold set annotation</title>
<style>
  :root {
    --bg:#faf9f5; --panel:#fff; --line:#e3e0d8; --ink:#2b2926;
    --muted:#6b6760; --accent:#7a6a52; --warn:#b4472e;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
  header { position:sticky; top:0; z-index:10; background:var(--panel);
           border-bottom:1px solid var(--line); padding:10px 18px;
           display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  .bar { flex:1; height:8px; background:var(--line); border-radius:4px; min-width:160px; }
  .bar>div { height:100%; background:var(--accent); border-radius:4px; width:0; transition:width .2s; }
  button { font:inherit; padding:7px 14px; border:1px solid var(--line);
           background:var(--panel); border-radius:7px; cursor:pointer; }
  button:hover { background:#f0ede5; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .stat { font-size:13px; color:var(--muted); }
  .stat b { color:var(--ink); }
  main { display:grid; grid-template-columns:1fr 380px; gap:18px; padding:18px; align-items:start; }
  @media (max-width:1050px){ main{grid-template-columns:1fr;} }
  .posting,.form { background:var(--panel); border:1px solid var(--line);
                   border-radius:10px; padding:18px; }
  .posting { max-height:80vh; overflow-y:auto; }
  .form { position:sticky; top:72px; max-height:80vh; overflow-y:auto; }
  h2 { margin:0 0 4px; font-size:17px; }
  .meta { color:var(--muted); font-size:13px; margin-bottom:10px; }
  .desc { font-size:14px; line-height:1.7; }
  .heading { display:inline-block; margin-top:14px; font-weight:700;
             font-size:15px; border-bottom:2px solid var(--accent); padding-bottom:1px; }
  mark { padding:1px 2px; border-radius:3px; }
  mark.skill{background:#cfe3f7;} mark.hedge-req{background:#c8ebd0;font-weight:600;}
  mark.hedge-pref{background:#fbe3b8;font-weight:600;} mark.arrangement{background:#e2d5f2;}
  mark.years{background:#f7d4d4;}
  fieldset { border:1px solid var(--line); border-radius:8px; margin:0 0 12px; padding:9px 11px; }
  fieldset.needed { border-color:var(--warn); }
  legend { font-size:11px; text-transform:uppercase; letter-spacing:.5px;
           color:var(--muted); padding:0 4px; }
  .skills { display:flex; flex-wrap:wrap; gap:4px; }
  .chip { font-size:12px; padding:3px 9px; border:1px solid var(--line);
          border-radius:20px; cursor:pointer; user-select:none; background:#fff; }
  .chip.req{background:#c8ebd0;border-color:#8fce9f;font-weight:600;}
  .chip.pref{background:#fbe3b8;border-color:#e0b96a;font-weight:600;}
  .hint { font-size:12px; color:var(--muted); margin-top:6px; }
  .hint.warn { color:var(--warn); font-weight:600; }
  input[type=number],input[type=text]{ font:inherit; padding:5px 8px;
    border:1px solid var(--line); border-radius:6px; width:100%; }
  .radios { display:flex; gap:5px; flex-wrap:wrap; }
  .radios label { font-size:13px; padding:5px 11px; border:1px solid var(--line);
                  border-radius:7px; cursor:pointer; }
  .radios input { display:none; }
  .radios label:has(input:checked){ background:#e2d5f2; border-color:#b79fd8; font-weight:700; }
  .key { font-size:12px; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
  .jump { font-size:12px; padding:3px 9px; }
</style>
</head>
<body>

<header>
  <strong>Gold set</strong>
  <span class="stat" id="counter"></span>
  <div class="bar"><div id="progress"></div></div>
  <span class="stat" id="timer"></span>
  <span class="stat" id="pace"></span>
  <button id="prev">← Prev</button>
  <button id="next" class="primary">Save &amp; next →</button>
  <button id="export">Export</button>
  <button id="reset" title="Delete all saved labels">Reset</button>
</header>

<main>
  <div class="posting" id="posting">
    <h2 id="title"></h2>
    <div class="meta" id="meta"></div>
    <div class="key">
      <span><mark class="skill">tool</mark></span>
      <span><mark class="hedge-req">required</mark></span>
      <span><mark class="hedge-pref">preferred</mark></span>
      <span><mark class="arrangement">location</mark></span>
      <span><mark class="years">years</mark></span>
      <button class="jump" id="jump">Jump to qualifications ↓</button>
    </div>
    <hr style="border:none;border-top:1px solid var(--line);margin:10px 0">
    <div class="desc" id="desc"></div>
  </div>

  <div class="form">
    <fieldset>
      <legend>Tools — click once = required, twice = preferred</legend>
      <div class="skills" id="skills"></div>
      <div class="hint">Only what the CANDIDATE must know. "Our platform runs
        on Spark" is company background, not a requirement.</div>
    </fieldset>

    <fieldset id="fs-arr" class="needed">
      <legend>Work arrangement — required</legend>
      <div class="radios" id="arrangement">
        <label><input type="radio" name="arr" value="remote">Remote</label>
        <label><input type="radio" name="arr" value="hybrid">Hybrid</label>
        <label><input type="radio" name="arr" value="onsite">Onsite</label>
        <label><input type="radio" name="arr" value="unclear">Not stated</label>
      </div>
      <div class="hint">"Remote within California" is still Remote. A city
        name with no other signal is Not stated, not Onsite.</div>
    </fieldset>

    <fieldset>
      <legend>Minimum years of experience</legend>
      <input type="number" id="years" min="0" max="30" placeholder="blank = not stated">
      <div class="hint">The floor to be considered. "3-5 years" → 3.
        "5+ years" → 5. Ignore company age.</div>
    </fieldset>

    <fieldset>
      <legend>Notes</legend>
      <label style="font-size:14px"><input type="checkbox" id="ambiguous">
        Genuinely ambiguous</label>
      <input type="text" id="note" placeholder="optional — why it was hard"
             style="margin-top:6px">
    </fieldset>

    <fieldset>
      <legend>Compare</legend>
      <label style="font-size:13px"><input type="checkbox" id="showguess">
        Show what the rules extractor guessed</label>
      <div class="hint" id="guessbox"></div>
    </fieldset>

    <div class="hint">Keys: <b>1</b> remote · <b>2</b> hybrid · <b>3</b> onsite
      · <b>4</b> not stated · <b>Cmd/Ctrl+Enter</b> next</div>
  </div>
</main>

<script>
const DATA = __DATA__;
const VOCAB = __VOCAB__;
const KEY = "jmi_gold_v2";

let idx = 0;
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
let startedAt = Date.now();
let peeked = false;

const $ = id => document.getElementById(id);
const done = () => Object.keys(labels).length;

function medianSeconds() {
  const xs = Object.values(labels).map(r => r.seconds_spent)
                   .filter(Number.isFinite).sort((a,b) => a-b);
  return xs.length ? xs[Math.floor(xs.length/2)] : 0;
}

function render() {
  const d = DATA[idx];
  startedAt = Date.now();
  peeked = false;
  $("showguess").checked = false;
  $("guessbox").textContent = "";

  $("title").textContent = d.title;
  $("meta").textContent =
    `${d.company} · ${d.location} · ${d.role_family} · ${d.chars.toLocaleString()} chars`;
  $("desc").innerHTML = d.html;
  $("counter").innerHTML = `<b>${idx+1}</b>/${DATA.length} · <b>${done()}</b> done`;
  $("progress").style.width = (100*done()/DATA.length) + "%";

  const med = medianSeconds();
  $("pace").innerHTML = med
    ? `median <b>${med}s</b>` + (med < 60 ? ' <span style="color:var(--warn)">— too fast</span>' : '')
    : "";

  const box = $("skills");
  box.innerHTML = "";
  const saved = labels[d.job_id];
  VOCAB.forEach(skill => {
    const chip = document.createElement("span");
    let state = "";
    if (saved) {
      if (saved.required_skills.includes(skill)) state = "req";
      else if (saved.preferred_skills.includes(skill)) state = "pref";
    }
    chip.className = "chip " + state;
    chip.textContent = skill;
    chip.onclick = () => {
      chip.className = "chip " + (chip.classList.contains("req") ? "pref"
                                : chip.classList.contains("pref") ? "" : "req");
    };
    box.appendChild(chip);
  });

  document.querySelectorAll("input[name=arr]").forEach(r => {
    r.checked = saved ? r.value === saved.work_arrangement : false;
  });
  $("years").value = saved ? (saved.years_experience_min ?? "") : "";
  $("ambiguous").checked = saved ? saved.ambiguous : false;
  $("note").value = saved ? saved.note : "";
  updateGate();
  $("posting").scrollTop = 0;
}

function updateGate() {
  const chosen = !!document.querySelector("input[name=arr]:checked");
  $("next").disabled = !chosen;
  $("fs-arr").className = chosen ? "" : "needed";
}

function tick() {
  const s = Math.round((Date.now() - startedAt)/1000);
  $("timer").innerHTML = `<b>${s}s</b> on this one`;
}
setInterval(tick, 1000);

document.getElementById("arrangement").addEventListener("change", updateGate);

function save() {
  const d = DATA[idx];
  const req = [], pref = [];
  document.querySelectorAll(".chip").forEach(c => {
    if (c.classList.contains("req")) req.push(c.textContent);
    else if (c.classList.contains("pref")) pref.push(c.textContent);
  });
  const arr = document.querySelector("input[name=arr]:checked");
  if (!arr) return false;
  const yrs = $("years").value;

  labels[d.job_id] = {
    job_id: d.job_id, title: d.title, role_family: d.role_family,
    required_skills: req, preferred_skills: pref,
    work_arrangement: arr.value,
    years_experience_min: yrs === "" ? null : Number(yrs),
    ambiguous: $("ambiguous").checked,
    note: $("note").value,
    seconds_spent: Math.max(
      labels[d.job_id]?.seconds_spent ?? 0,
      Math.round((Date.now() - startedAt) / 1000)
    ),
    saw_rules_guess: peeked,
    annotated_at: new Date().toISOString(),
  };
  localStorage.setItem(KEY, JSON.stringify(labels));
  return true;
}

$("next").onclick = () => {
  if (!save()) return;
  if (idx < DATA.length - 1) idx++;
  render();
};
$("prev").onclick = () => {
  const d = DATA[idx];
  if (labels[d.job_id]) {
    labels[d.job_id].seconds_spent = Math.max(
      labels[d.job_id].seconds_spent,
      Math.round((Date.now() - startedAt) / 1000)
    );
    localStorage.setItem(KEY, JSON.stringify(labels));
  }
  if (idx > 0) idx--;
  render();
};

$("jump").onclick = () => {
  const h = document.querySelectorAll("#desc .heading");
  if (!h.length) return;
  // Requirements usually sit in the back half of the posting.
  h[Math.min(h.length - 1, Math.floor(h.length/2))]
    .scrollIntoView({behavior:"smooth", block:"start"});
};

$("showguess").onchange = e => {
  const d = DATA[idx];
  if (e.target.checked) {
    peeked = true;
    $("guessbox").innerHTML =
      `required: ${d.guess_required.join(", ") || "—"}<br>` +
      `preferred: ${d.guess_preferred.join(", ") || "—"}<br>` +
      `years: ${d.guess_years ?? "—"}` +
      `<br><i>recorded that you looked</i>`;
  } else {
    $("guessbox").textContent = "";
  }
};

$("export").onclick = () => {
  const rows = Object.values(labels);
  const med = medianSeconds();
  const blob = new Blob([JSON.stringify({
    n: rows.length, median_seconds: med,
    exported_at: new Date().toISOString(), labels: rows
  }, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "gold_labels.json";
  a.click();
};

$("reset").onclick = () => {
  if (!confirm("Delete all saved labels? This cannot be undone.")) return;
  labels = {}; localStorage.removeItem(KEY); idx = 0; render();
};

document.addEventListener("keydown", e => {
  const typing = ["INPUT","TEXTAREA"].includes(e.target.tagName);
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { $("next").click(); return; }
  if (typing) return;
  const map = {"1":"remote","2":"hybrid","3":"onsite","4":"unclear"};
  if (map[e.key]) {
    document.querySelector(`[value=${map[e.key]}]`).checked = true;
    updateGate();
  }
});

render();
</script>
</body>
</html>
"""


def main() -> None:
    records = build_records()
    page = (TEMPLATE
            .replace("__DATA__", json.dumps(records))
            .replace("__VOCAB__", json.dumps(SKILL_VOCAB)))

    out = DOCS / "annotate.html"
    out.write_text(page, encoding="utf-8")

    print(f"Wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(records)} postings)")
    print(f"\nOpen it:  open {out}")
    print("""
FIRST: click Reset. The previous attempt stored 22 click-through records
that must not reach the gold set.

How to annotate one posting:

  1. Click "Jump to qualifications" -- it scrolls to the requirements list.
     That is where the answer lives; the rest is company blurb.
  2. Find the two lists: what they NEED (required) and what they'd LIKE
     (preferred). Headings are bold and underlined.
  3. Click each tool's chip: once = required (green), twice = preferred
     (amber), three times = clear.
  4. Choose the work arrangement. You cannot advance without it.
  5. Type the minimum years, or leave blank if not stated.
  6. Cmd+Enter for the next one.

Judgement calls:
  * Tools: only what the CANDIDATE needs. "We use Spark" is background.
  * Required vs preferred: hedged ("a plus", "familiarity with") = preferred.
  * Years: the floor. "3-5 years" -> 3. "5+ years" -> 5.
  * Arrangement: "Remote within California" is Remote. A bare city name is
    Not stated.
  * Tick ambiguous freely. How many were genuinely unclear is itself a result.

The header shows seconds on the current posting and your running median.
If the median drops below 60s the pace indicator turns red. These postings
average 5,800 characters -- under a minute means the reading did not happen.
Work in 20-30 minute sittings; fatigue errors are the failure mode here.
""")


if __name__ == "__main__":
    main()
