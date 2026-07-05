import { useState, useRef, useEffect } from "react";

// ---------- Design tokens ----------
const T = {
  paper: "#F1F4F8", ink: "#0F1B2D", sub: "#5B6B7F", line: "#DDE4EC",
  card: "#FFFFFF", accent: "#0E7490", accentSoft: "#E0F2F7",
  amber: "#B45309", amberSoft: "#FEF3E2", green: "#047857", greenSoft: "#E7F6EF",
  red: "#B42318", redSoft: "#FEF0EE",
  mono: "'JetBrains Mono','SF Mono',Menlo,monospace",
  disp: "'Space Grotesk','Avenir Next',system-ui,sans-serif",
  body: "system-ui,-apple-system,'Segoe UI',sans-serif",
};

// ---------- Capability tags: which options work on text vs text+images ----------
const CAPABILITY = {
  parsing: { "Assorted (auto-detect)": "both", "PDF – PyMuPDF": "both", "DOCX – python-docx": "both",
             "PPTX – python-pptx": "both", "Markdown": "both" },
  chunking: { "Recursive": "text", "Fixed": "text", "Semantic": "text", "Sentence": "text", "Markdown structure": "text" },
  embedding: { "MiniLM-L6 (local)": "text", "BGE-small (local)": "text", "Gemini text-embedding": "text",
               "OpenAI text-embedding-3-small": "text", "OpenAI text-embedding-3-large": "text" },
  retrieval: { "Semantic": "both", "Keyword (BM25)": "text", "Hybrid-RRF": "both" },
  reranking: { "None": "both", "Cross-encoder (local)": "text" },
  generation: { "Claude Sonnet": "both", "Gemini 2.5 Flash": "both", "Gemini 2.5 Pro": "both", "GPT-4o mini": "both" },
};
const CapTag = ({ cap }) => (
  <span style={{
    fontFamily: T.mono, fontSize: 9.5, padding: "1px 6px", borderRadius: 99, marginLeft: 6,
    background: cap === "both" ? T.greenSoft : "#EEF1F6", color: cap === "both" ? T.green : T.sub,
  }}>{cap === "both" ? "text+image" : "text only"}</span>
);

// ---------- Mock pipeline data ----------
const INITIAL_STAGES = [
  { id: "parsing", label: "Parsing", options: Object.keys(CAPABILITY.parsing), sel: 0,
    out: "Parsed 5 docs (pdf, docx, pptx, md) → 6 pages, 54 words. Images: 5. OCR ran on 1 scanned page.",
    lat: 35, tok: 0, cost: 0, extractImages: true, ocr: true,
    evalFree: { n_docs: 5, total_words: 54, total_images: 5, n_ocr_blocks: 1 },
    rec: "1 doc produced no native text — OCR correctly recovered it. Spot-check OCR quality on scans." },
  { id: "chunking", label: "Chunking", options: ["Recursive", "Fixed", "Semantic", "Sentence", "Markdown structure"], sel: 0,
    chunkSize: 512, overlap: 64,
    out: "6 chunks. Mean 12.8 tokens, σ=9.3 (tiny sample corpus — real corpora will track chunk_size far more closely).",
    lat: 0.1, tok: 0, cost: 0,
    evalFree: { n_chunks: 6, mean_tokens: 12.8, std_tokens: 9.3 },
    rec: "Chunks are much smaller than target — expected on this 5-file demo corpus; on your real corpus this flags fragmentation." },
  { id: "embedding", label: "Embedding", options: ["MiniLM-L6 (local)", "BGE-small (local)", "Gemini text-embedding",
      "OpenAI text-embedding-3-small", "OpenAI text-embedding-3-large"], sel: 0,
    out: "6 text vectors, dim 384. 5 image vectors via captioning. Indexed in ChromaDB.",
    lat: 1340, tok: 0, cost: 0,
    evalFree: { n_text_vectors: 6, dim: 384, n_image_vectors: 5 },
    rec: "Vectors indexed and ready for retrieval." },
  { id: "retrieval", label: "Retrieval", options: ["Semantic", "Keyword (BM25)", "Hybrid-RRF"], sel: 2,
    topK: 10, resultMode: 1,
    out: "6 text chunks + 2 images retrieved. Top score 0.81. Image #3 (Fig. 2, attention heatmap) score 0.77.",
    lat: 96, tok: 0, cost: 0,
    evalFree: { n_results: 8, top_score: 0.81, score_floor: 0.42, n_images: 2 },
    rec: "Wide score spread (0.81→0.42) — the lowest-ranked results may be irrelevant; reranking should help." },
  { id: "reranking", label: "Re-ranking", options: ["None", "Cross-encoder (local)"], sel: 1, keepTop: 4,
    out: "8 → 4 kept (3 text + 1 image). Order changed for 3. Top chunk stays rank 1 (score 0.93).",
    lat: 640, tok: 0, cost: 0,
    evalFree: { n_results: 4, top_score: 0.93, score_floor: 0.71, n_images: 1 },
    rec: "Scores tightened after reranking — healthy, no further action needed." },
  { id: "generation", label: "Generation", options: ["Claude Sonnet", "Gemini 2.5 Flash", "Gemini 2.5 Pro", "GPT-4o mini"], sel: 0,
    visionGrounded: false,
    out: "\"Self-attention lets each token weigh every other token when building its representation, as shown in Fig. 2 [img#3]…\" (312 tokens, grounded in 3 sources).",
    lat: 2380, tok: 1840, cost: 0.011,
    evalFree: { faithfulness: 0.91, answer_relevancy: 0.87, context_precision_without_reference: 0.74 },
    rec: "All reference-free scores healthy. Context precision (0.74) is the weakest — try tightening top_k or reranking further." },
];

const IMAGE_EMBED_OPTIONS = ["Caption + text-embed (vision LLM)", "CLIP (local)"];
const RESULT_MODES = ["Text only", "Text + Images (joint rank)", "Text + Images (separate, merge top-k)"];
const PROVIDERS = ["claude", "gemini", "openai"];
const REWRITE_GUIDELINES = {
  claude: "Best for precise, structure-preserving rewrites — use when the prompt already has a shape you want kept.",
  gemini: "Fastest and cheapest — use for quick iteration when you'll throw away most drafts.",
  openai: "Good second opinion with a different house style from Claude/Gemini.",
};

const LEARN = {
  parsing: {
    concept: "Every format becomes one shape: text blocks + images + tables, each tagged with page/section. 'Assorted' just looks at the file extension and routes to the right parser — same output schema either way.",
    table: [["Assorted", "Mixed corpus (pdf+docx+pptx+md together) — the default for real use", "both"], ["Format-specific", "Single-format corpus, slightly simpler traces", "both"], ["OCR toggle", "Turn on when pages are scans/photos of text, not real text layers", "both"]],
    demo: "Toy: a 1-page scanned PDF with no text layer → OCR off gives 0 words; OCR on recovers the sentence via Tesseract, tagged source_ocr:true so you always know which text came from where.",
  },
  chunking: {
    concept: "Chunk size trades off two errors: too large dilutes the embedding (low precision); too small splits a fact across chunks (low recall). Markdown structure groups by heading instead of a token window.",
    table: [["Recursive", "Safe general default; overlap protects boundary splits", "text"], ["Fixed", "Short, uniform, fact-dense text (FAQs)", "text"], ["Semantic", "Long, structurally loose prose (essays, transcripts)", "text"], ["Markdown structure", "Docs where headings carry real structure (specs, wikis)", "text"]],
    demo: "chunk_size/overlap are manual inputs (default 512/64 tokens) — toy: one paragraph split at 256 vs 512 shows a mid-sentence cut appear and disappear.",
  },
  embedding: {
    concept: "Embeddings map text to a vector so cosine similarity ≈ semantic similarity: sim(a,b) = (a·b)/(‖a‖‖b‖). Local models are free; API embeddings cost per call.",
    table: [["MiniLM-L6 (local)", "Default for ≤20 files — free, fast, private", "text"], ["Gemini / OpenAI-3-*", "Demonstrating API-embedding cost/quality trade-off — needs that provider's key", "text"], ["Caption+text-embed", "Images: caption once via vision LLM, embed in the SAME space as text", "both"], ["CLIP (local)", "Pure visual similarity ('find charts like this') — its own space, needs its own query encoder", "both"]],
    demo: "Toy: 'car' vs 'automobile' vs 'banana' → cosine ≈0.82, 0.82, 0.11. The vector space encodes meaning, not just words.",
  },
  retrieval: {
    concept: "Semantic ranks by embedding similarity; Keyword (BM25) ranks by weighted term overlap. Hybrid-RRF fuses both by RANK, not raw score: RRF(d)=Σ 1/(k+rank_i(d)), k=60 — this sidesteps the problem that cosine and BM25 scores live on different scales.",
    table: [["Semantic", "Conceptual/paraphrased queries", "both (with caption/CLIP images)"], ["Keyword (BM25)", "Exact terms, codes, acronyms embeddings blur", "text"], ["Hybrid-RRF", "Real queries mix both — usually the best default", "both"], ["Result mode", "text-only / joint (needs shared embedding space) / separate-merge (always safe)", "both"]],
    demo: "Toy query 'BLEU score formula' — Semantic alone ranks a paraphrase above the exact formula chunk; Hybrid-RRF fixes it.",
  },
  reranking: {
    concept: "Retrieval optimizes for recall (cast a wide net cheaply); reranking optimizes precision on that smaller set. Cross-encoders score (query, chunk) jointly instead of comparing two frozen vectors.",
    table: [["None", "Retrieval is already precise, or teaching the baseline", "both"], ["Cross-encoder (local)", "Best precision-per-rupee, zero API cost — text pairs only; images pass through unscored", "text"]],
    demo: "Toy: 8 retrieved chunks, cross-encoder flips rank 5 to rank 1 because it reads query+chunk together.",
  },
  generation: {
    concept: "Grounding forces the model to answer from retrieved context, not memory. Vision-grounded generation sends actual image bytes instead of just the caption — costs more, lets the model read the figure itself.",
    table: [["Claude Sonnet", "Strongest grounded, citation-style answers", "both"], ["Gemini 2.5 Flash", "~8× cheaper, good for high-volume/demo use", "both"], ["Vision-grounded off (default)", "Caption text is enough — cheaper, faster", "both"], ["Vision-grounded on", "Answer genuinely depends on reading the image", "both"]],
    demo: "Toy: caption-only gets a chart's trend right but invents an exact number; vision-grounded reads it correctly.",
  },
};

const REF_FREE_LABELS = { faithfulness: "Faithfulness", answer_relevancy: "Answer relevancy", context_precision_without_reference: "Context precision" };

const DEFAULT_PROMPT = "You are a helpful assistant. Answer using ONLY the provided context. Cite chunk ids like [#47]. If the context is insufficient, say so.";
const REWRITTEN_PROMPT = "You are a precise technical tutor. Ground every claim in the provided context and cite chunk ids inline like [#47]. Structure: 1-line direct answer → supporting explanation → cited evidence. If context is insufficient, state exactly what is missing instead of guessing. Never use outside knowledge.";

// ---------- Small primitives ----------
const Chip = ({ children, color = T.accent, bg = T.accentSoft }) => (
  <span style={{ background: bg, color, fontFamily: T.mono, fontSize: 11, padding: "2px 8px", borderRadius: 99, whiteSpace: "nowrap" }}>{children}</span>
);
const Bar = ({ v, color }) => (
  <div style={{ background: "#EDF1F6", borderRadius: 99, height: 6, width: "100%" }}>
    <div style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%`, background: color, height: 6, borderRadius: 99, transition: "width .6s ease" }} />
  </div>
);
const scoreColor = v => v >= 0.8 ? T.green : v >= 0.6 ? T.amber : T.red;

const Toggle = ({ label, checked, onChange }) => (
  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: T.sub, cursor: "pointer" }}>
    <input type="checkbox" checked={checked} onChange={onChange} style={{ accentColor: T.accent }} />
    {label}
  </label>
);
const NumberField = ({ label, value, onChange, min = 1, max = 4096 }) => (
  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: T.sub }}>
    {label}
    <input type="number" value={value} min={min} max={max} onChange={e => onChange(+e.target.value)}
      style={{ width: 68, padding: "4px 6px", borderRadius: 6, border: `1px solid ${T.line}`, fontFamily: T.mono, fontSize: 12 }} />
  </label>
);

// ---------- Eval report block (shown after every stage) ----------
function EvalReport({ evalFree, rec, needsAck, acknowledged, onAck }) {
  if (!evalFree) return null;
  return (
    <div style={{ marginTop: 10, background: "#F6F8FB", border: `1px solid ${T.line}`, borderRadius: 10, padding: 12 }}>
      <div style={{ fontFamily: T.disp, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: T.sub, marginBottom: 6 }}>Reference-free report</div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12 }}>
        {Object.entries(evalFree).map(([k, v]) => (
          <span key={k} style={{ fontFamily: T.mono }}>
            {k}: <b style={{ color: typeof v === "number" && v <= 1 ? scoreColor(v) : T.ink }}>{typeof v === "number" ? v : v}</b>
          </span>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 12.5, color: "#155E6E", lineHeight: 1.5 }}>
        <b style={{ fontFamily: T.disp }}>Recommendation:</b> {rec}
      </div>
      {needsAck && (
        <button onClick={onAck} disabled={acknowledged}
          style={{ marginTop: 10, padding: "7px 14px", borderRadius: 8, border: "none", cursor: acknowledged ? "default" : "pointer",
                   background: acknowledged ? T.greenSoft : T.ink, color: acknowledged ? T.green : "#fff",
                   fontFamily: T.disp, fontSize: 12.5, fontWeight: 600 }}>
          {acknowledged ? "✓ Acknowledged — advancing" : "Acknowledge & continue"}
        </button>
      )}
    </div>
  );
}

// ---------- Learn drawer ----------
function LearnDrawer({ content }) {
  if (!content) return null;
  return (
    <div style={{ marginTop: 10, background: "#F6F8FB", border: `1px solid ${T.line}`, borderRadius: 10, padding: 14 }}>
      <div style={{ fontFamily: T.disp, fontSize: 11.5, letterSpacing: 1, textTransform: "uppercase", color: T.accent, marginBottom: 6 }}>Concept</div>
      <div style={{ fontSize: 13, lineHeight: 1.6, color: T.ink }}>{content.concept}</div>
      <div style={{ fontFamily: T.disp, fontSize: 11.5, letterSpacing: 1, textTransform: "uppercase", color: T.accent, margin: "12px 0 6px" }}>When to use which</div>
      {content.table.map(([opt, when, cap], i) => (
        <div key={i} style={{ display: "flex", gap: 10, fontSize: 12.5, padding: "5px 0", borderBottom: i < content.table.length - 1 ? `1px dashed ${T.line}` : "none", alignItems: "baseline" }}>
          <span style={{ fontFamily: T.mono, fontWeight: 700, minWidth: 140, color: T.ink }}>{opt}{cap && <CapTag cap={cap === "text" ? "text" : "both"} />}</span>
          <span style={{ color: T.sub }}>{when}</span>
        </div>
      ))}
      <div style={{ fontFamily: T.disp, fontSize: 11.5, letterSpacing: 1, textTransform: "uppercase", color: T.accent, margin: "12px 0 6px" }}>Toy demo</div>
      <div style={{ fontSize: 12.5, lineHeight: 1.6, color: T.sub, fontStyle: "italic" }}>{content.demo}</div>
    </div>
  );
}

// ---------- Stage card ----------
function StageCard({ stage, idx, status, onSelect, onSuggest, suggestion, onManage, trace, expanded, onToggle,
                     learnOpen, onLearnToggle, onFlag, onNum, runMode, acknowledged, onAck }) {
  const live = status === "running", done = status === "done";
  const needsAck = runMode === "step" && done;

  return (
    <div style={{ display: "flex", gap: 14 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 26 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 99, display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: T.mono, fontSize: 10, fontWeight: 700, flexShrink: 0, transition: "all .3s",
          background: done ? T.accent : live ? T.ink : "#fff",
          color: done || live ? "#fff" : T.sub,
          border: `2px solid ${done || live ? T.accent : T.line}`,
          boxShadow: live ? `0 0 0 5px ${T.accentSoft}` : "none",
        }}>{done ? "✓" : idx + 1}</div>
        <div style={{ flex: 1, width: 2, background: done ? T.accent : T.line, transition: "background .4s", minHeight: 18 }} />
      </div>

      <div style={{ flex: 1, background: T.card, border: `1px solid ${live ? T.accent : T.line}`, borderRadius: 12, padding: 14, marginBottom: 14, transition: "border .3s" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15, color: T.ink }}>{stage.label}</div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {done && <Chip>{stage.lat} ms</Chip>}
            {done && stage.cost > 0 && <Chip color={T.amber} bg={T.amberSoft}>${stage.cost.toFixed(3)}</Chip>}
            <button onClick={() => onLearnToggle(idx)} title="Learn: theory, guidelines, demo"
              style={{ width: 24, height: 24, borderRadius: 99, border: `1px solid ${learnOpen ? T.accent : T.line}`, background: learnOpen ? T.accent : "#fff", color: learnOpen ? "#fff" : T.sub, cursor: "pointer", fontSize: 12, fontFamily: T.disp, lineHeight: 1 }}>?</button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
          <select value={stage.sel} onChange={e => onSelect(idx, +e.target.value)}
            style={{ flex: 1, minWidth: 160, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.body, fontSize: 13, color: T.ink, background: "#FAFCFE" }}>
            {stage.options.map((o, i) => <option key={i} value={i}>{o}</option>)}
          </select>
          <button onClick={() => onManage(idx)} title="Add / remove options"
            style={{ padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, background: "#fff", cursor: "pointer", fontSize: 13, color: T.sub }}>⚙︎</button>
          <button onClick={() => onSuggest(idx)}
            style={{ padding: "8px 12px", borderRadius: 8, border: "none", background: T.ink, color: "#fff", cursor: "pointer", fontSize: 12, fontFamily: T.disp }}>✦ Suggest</button>
        </div>
        <div style={{ marginTop: 4 }}><CapTag cap={CAPABILITY[stage.id]?.[stage.options[stage.sel]] || "text"} /></div>

        {/* Stage-specific manual controls */}
        {stage.id === "parsing" && (
          <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Toggle label="Extract images" checked={stage.extractImages} onChange={() => onFlag(idx, "extractImages")} />
            <Toggle label="OCR scanned pages (Tesseract)" checked={stage.ocr} onChange={() => onFlag(idx, "ocr")} />
          </div>
        )}
        {stage.id === "parsing" && stage.extractImages && (
          <div style={{ marginTop: 8, background: T.greenSoft, borderRadius: 8, padding: "8px 10px", fontSize: 12, color: "#065F46" }}>
            Image embedding method: <select defaultValue={0} style={{ marginLeft: 6, fontSize: 12, padding: "3px 6px", borderRadius: 6, border: `1px solid ${T.line}` }}>
              {IMAGE_EMBED_OPTIONS.map((o, i) => <option key={i} value={i}>{o}</option>)}
            </select>
          </div>
        )}
        {stage.id === "chunking" && (
          <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
            <NumberField label="Chunk size (tokens)" value={stage.chunkSize} onChange={v => onNum(idx, "chunkSize", v)} min={16} max={4096} />
            <NumberField label="Overlap (tokens)" value={stage.overlap} onChange={v => onNum(idx, "overlap", v)} min={0} max={1024} />
          </div>
        )}
        {stage.id === "retrieval" && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
            <NumberField label="Top K" value={stage.topK} onChange={v => onNum(idx, "topK", v)} min={1} max={50} />
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: T.sub, flexWrap: "wrap" }}>
              Result mode:
              <select value={stage.resultMode} onChange={e => onNum(idx, "resultMode", +e.target.value)}
                style={{ fontSize: 12.5, padding: "5px 8px", borderRadius: 6, border: `1px solid ${T.line}`, flex: 1, minWidth: 200 }}>
                {RESULT_MODES.map((o, i) => <option key={i} value={i}>{o}</option>)}
              </select>
            </div>
          </div>
        )}
        {stage.id === "reranking" && (
          <div style={{ marginTop: 8 }}>
            <NumberField label="Keep top" value={stage.keepTop} onChange={v => onNum(idx, "keepTop", v)} min={1} max={20} />
          </div>
        )}
        {stage.id === "generation" && (
          <div style={{ marginTop: 10 }}>
            <Toggle label="Vision-grounded generation (send image, not just caption)" checked={stage.visionGrounded} onChange={() => onFlag(idx, "visionGrounded")} />
          </div>
        )}

        {suggestion && (
          <div style={{ marginTop: 10, background: T.accentSoft, borderLeft: `3px solid ${T.accent}`, borderRadius: "0 8px 8px 0", padding: "8px 12px", fontSize: 12.5, color: "#155E6E", lineHeight: 1.5 }}>
            <b style={{ fontFamily: T.disp }}>Advisor:</b> {suggestion}
          </div>
        )}

        {learnOpen && <LearnDrawer content={LEARN[stage.id]} />}

        {(live || done) && (
          <div style={{ marginTop: 10 }}>
            <button onClick={() => onToggle(idx)} style={{ background: "none", border: "none", cursor: "pointer", fontFamily: T.mono, fontSize: 11, color: T.accent, padding: 0 }}>
              {live ? "● streaming…" : expanded ? "▾ trace" : "▸ trace"}
            </button>
            {done && expanded && (
              <pre style={{ margin: "6px 0 0", background: "#0F1B2D", color: "#C7E5EE", borderRadius: 8, padding: 12, fontFamily: T.mono, fontSize: 11.5, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{trace}</pre>
            )}
            {done && <EvalReport evalFree={stage.evalFree} rec={stage.rec} needsAck={needsAck} acknowledged={acknowledged} onAck={onAck} />}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------- App ----------
export default function RAGLab() {
  const [stages, setStages] = useState(INITIAL_STAGES);
  const [statuses, setStatuses] = useState(Array(6).fill("idle"));
  const [expanded, setExpanded] = useState(Array(6).fill(false));
  const [acks, setAcks] = useState(Array(6).fill(false));
  const [suggestions, setSuggestions] = useState({});
  const [keysOpen, setKeysOpen] = useState(false);
  const [keys, setKeys] = useState({ claude: "", gemini: "", openai: "" });
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [rewriteProvider, setRewriteProvider] = useState("claude");
  const [rewriting, setRewriting] = useState(false);
  const [query, setQuery] = useState("How does self-attention work in the Transformer paper?");
  const [running, setRunning] = useState(false);
  const [runMode, setRunMode] = useState("all"); // "all" | "step"
  const [pausedAt, setPausedAt] = useState(-1);
  const [tab, setTab] = useState("trace");
  const [hasDataset, setHasDataset] = useState(false);
  const [manageIdx, setManageIdx] = useState(null);
  const [newOpt, setNewOpt] = useState("");
  const [learnOpen, setLearnOpen] = useState(Array(6).fill(false));
  const timers = useRef([]);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const totalLat = stages.reduce((a, s) => a + s.lat, 0);
  const totalCost = stages.reduce((a, s) => a + s.cost, 0);
  const doneAll = statuses.every(s => s === "done");

  const runFrom = (startIdx) => {
    setRunning(true); setTab("trace");
    let t = 100;
    for (let i = startIdx; i < stages.length; i++) {
      const s = stages[i];
      timers.current.push(setTimeout(() => setStatuses(p => p.map((x, j) => j === i ? "running" : x)), t));
      t += Math.max(400, s.lat * 0.4);
      timers.current.push(setTimeout(() => {
        setStatuses(p => p.map((x, j) => j === i ? "done" : x));
        setExpanded(p => p.map((x, j) => j === i ? true : x));
        if (runMode === "step") { setRunning(false); setPausedAt(i); }
        else if (i === stages.length - 1) setRunning(false);
      }, t));
      if (runMode === "step") break; // only run ONE stage, then wait for acknowledgment
    }
  };

  const run = () => {
    if (running) return;
    setStatuses(Array(6).fill("idle")); setExpanded(Array(6).fill(false)); setAcks(Array(6).fill(false));
    setPausedAt(-1);
    runFrom(0);
  };

  const acknowledge = (idx) => {
    setAcks(p => p.map((x, j) => j === idx ? true : x));
    if (idx < stages.length - 1) runFrom(idx + 1);
  };

  const rewrite = () => {
    setRewriting(true);
    timers.current.push(setTimeout(() => { setPrompt(REWRITTEN_PROMPT); setRewriting(false); }, 1100));
  };

  const REF_FREE = stages[5].evalFree ? Object.entries(stages[5].evalFree).map(([k, v]) => ({ m: REF_FREE_LABELS[k] || k, v, d: "" })) : [];
  const WITH_REF = [
    { m: "Context precision (ref)", v: 0.83, d: "Retrieved chunks judged against the gold answer" },
    { m: "Context recall", v: 0.78, d: "Share of gold-answer facts actually retrieved" },
    { m: "Answer correctness", v: 0.72, d: "Weighted blend of factual + semantic match to gold" },
    { m: "Semantic similarity", v: 0.89, d: "Cosine sim of answer vs gold" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: T.paper, fontFamily: T.body, color: T.ink }}>
      {/* Top bar */}
      <div style={{ background: T.ink, color: "#fff", padding: "12px 20px", display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <div style={{ fontFamily: T.disp, fontWeight: 700, fontSize: 17, letterSpacing: .3 }}>
          Glass<span style={{ color: "#67D6EC" }}>Box</span> <span style={{ fontWeight: 400, opacity: .7 }}>· AI Training Lab</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <span style={{ background: "#1D3247", padding: "5px 14px", borderRadius: 99, fontSize: 12.5, fontFamily: T.disp, border: "1px solid #2E4B66" }}>RAG Studio</span>
          <span style={{ padding: "5px 14px", borderRadius: 99, fontSize: 12.5, opacity: .45 }}>ReAct Loop · soon</span>
          <span style={{ padding: "5px 14px", borderRadius: 99, fontSize: 12.5, opacity: .45 }}>Agent Eval · soon</span>
        </div>
        <button onClick={() => setKeysOpen(o => !o)} style={{ marginLeft: "auto", background: "none", border: "1px solid #2E4B66", color: "#9FD9E8", padding: "6px 14px", borderRadius: 8, cursor: "pointer", fontSize: 12.5 }}>
          {keys.claude || keys.gemini || keys.openai ? "🔑 Keys set" : "🔑 API keys"}
        </button>
      </div>

      {keysOpen && (
        <div style={{ background: "#fff", borderBottom: `1px solid ${T.line}`, padding: "14px 20px", display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[["claude", "Anthropic key (Claude Sonnet)"], ["gemini", "Google key (Gemini)"], ["openai", "OpenAI key (GPT-4o / text-embedding-3-*)"]].map(([k, label]) => (
            <label key={k} style={{ fontSize: 12, color: T.sub, display: "flex", flexDirection: "column", gap: 4, flex: 1, minWidth: 220 }}>
              {label}
              <input type="password" placeholder="stored in session only" value={keys[k]}
                onChange={e => setKeys(p => ({ ...p, [k]: e.target.value }))}
                style={{ padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.mono, fontSize: 12 }} />
            </label>
          ))}
        </div>
      )}

      <div style={{ display: "flex", gap: 20, padding: 20, flexWrap: "wrap", alignItems: "flex-start", maxWidth: 1280, margin: "0 auto" }}>
        {/* LEFT: builder */}
        <div style={{ flex: "1 1 460px", minWidth: 340 }}>
          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Pipeline · signal chain</div>

          {/* System prompt block */}
          <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14, marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
              <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15 }}>System prompt</div>
              <div style={{ display: "flex", gap: 6 }}>
                <select value={rewriteProvider} onChange={e => setRewriteProvider(e.target.value)}
                  style={{ padding: "6px 8px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 12 }}>
                  {PROVIDERS.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
                <button onClick={rewrite} disabled={rewriting}
                  style={{ padding: "6px 12px", borderRadius: 8, border: "none", background: rewriting ? T.sub : T.accent, color: "#fff", cursor: "pointer", fontSize: 12, fontFamily: T.disp }}>
                  {rewriting ? "Rewriting…" : "✦ Rewrite with LLM"}
                </button>
              </div>
            </div>
            <div style={{ fontSize: 11.5, color: T.sub, marginTop: 6 }}>{REWRITE_GUIDELINES[rewriteProvider]}</div>
            <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={4}
              style={{ width: "100%", marginTop: 10, padding: 10, borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.mono, fontSize: 12, lineHeight: 1.6, color: T.ink, resize: "vertical", boxSizing: "border-box", background: "#FAFCFE" }} />
          </div>

          {stages.map((s, i) => (
            <StageCard key={s.id} stage={s} idx={i} status={statuses[i]}
              onSelect={(idx, v) => setStages(p => p.map((x, j) => j === idx ? { ...x, sel: v } : x))}
              onSuggest={idx => setSuggestions(p => ({ ...p, [stages[idx].id]: p[stages[idx].id] ? undefined : "Advisor would call your chosen LLM here with a short description of your corpus/query and recommend one option with a 2-sentence justification." }))}
              suggestion={suggestions[s.id]}
              onManage={setManageIdx}
              trace={`in : ${i === 0 ? "sample_corpus/ (5 files)" : `output of ${stages[i - 1].label.toLowerCase()}`}\ncfg: ${s.options[s.sel]}\nout: ${s.out}`}
              expanded={expanded[i]} onToggle={idx => setExpanded(p => p.map((x, j) => j === idx ? !x : x))}
              learnOpen={learnOpen[i]} onLearnToggle={idx => setLearnOpen(p => p.map((x, j) => j === idx ? !x : x))}
              onFlag={(idx, key) => setStages(p => p.map((x, j) => j === idx ? { ...x, [key]: !x[key] } : x))}
              onNum={(idx, key, v) => setStages(p => p.map((x, j) => j === idx ? { ...x, [key]: v } : x))}
              runMode={runMode} acknowledged={acks[i]} onAck={() => acknowledge(i)} />
          ))}
        </div>

        {/* RIGHT: run + results */}
        <div style={{ flex: "1 1 420px", minWidth: 340, position: "sticky", top: 20 }}>
          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Run & inspect</div>
          <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              {[["all", "Run all"], ["step", "Step-by-step"]].map(([m, label]) => (
                <button key={m} onClick={() => setRunMode(m)}
                  style={{ flex: 1, padding: "7px 10px", borderRadius: 8, border: `1px solid ${runMode === m ? T.accent : T.line}`,
                           background: runMode === m ? T.accentSoft : "#fff", color: runMode === m ? T.accent : T.sub,
                           fontFamily: T.disp, fontSize: 12.5, cursor: "pointer" }}>{label}</button>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <input value={query} onChange={e => setQuery(e.target.value)}
                style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13, background: "#FAFCFE" }} />
              <button onClick={run} disabled={running}
                style={{ padding: "10px 18px", borderRadius: 8, border: "none", background: running ? T.sub : T.accent, color: "#fff", fontFamily: T.disp, fontWeight: 600, fontSize: 13, cursor: "pointer" }}>
                {running ? "Running…" : "Run ▸"}
              </button>
            </div>
            {runMode === "step" && <div style={{ fontSize: 11.5, color: T.sub, marginTop: 6 }}>Each stage pauses at "Acknowledge & continue" below it — advance stage by stage on the left.</div>}

            <div style={{ display: "flex", gap: 4, marginTop: 14, borderBottom: `1px solid ${T.line}` }}>
              {["trace", "evaluation", "ragas", "compare"].map(t => (
                <button key={t} onClick={() => setTab(t)}
                  style={{ padding: "8px 14px", border: "none", background: "none", cursor: "pointer", fontFamily: T.disp, fontSize: 13, color: tab === t ? T.accent : T.sub, borderBottom: tab === t ? `2px solid ${T.accent}` : "2px solid transparent" }}>
                  {t === "ragas" ? "RAGAS" : t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {tab === "trace" && (
              <div style={{ paddingTop: 12 }}>
                {!doneAll && !running && <div style={{ color: T.sub, fontSize: 13, padding: "24px 0", textAlign: "center" }}>Press <b>Run</b> — the signal rail on the left lights up stage by stage, with an eval report after each one.</div>}
                {(running || doneAll || pausedAt >= 0) && (
                  <div style={{ fontSize: 13 }}>
                    {stages.map((s, i) => (
                      <div key={s.id} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px dashed ${T.line}`, opacity: statuses[i] === "idle" ? .35 : 1 }}>
                        <span style={{ fontFamily: T.mono, fontSize: 12 }}>{statuses[i] === "done" ? "✓" : statuses[i] === "running" ? "●" : "○"} {s.label}</span>
                        <span style={{ fontFamily: T.mono, fontSize: 12, color: T.sub }}>{statuses[i] === "done" ? `${s.lat} ms${s.tok ? ` · ${s.tok} tok` : ""}` : ""}</span>
                      </div>
                    ))}
                    {doneAll && (
                      <div style={{ marginTop: 12, background: T.greenSoft, borderRadius: 8, padding: 12, fontSize: 13, lineHeight: 1.6 }}>
                        <b style={{ fontFamily: T.disp, color: T.green }}>Answer</b><br />
                        Self-attention lets each token weigh every other token when building its representation [#47]. Queries, keys and values are linear projections of the input; attention = softmax(QKᵀ/√d)V [#12]…
                      </div>
                    )}
                    <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                      <Chip>Σ latency {doneAll ? (totalLat / 1000).toFixed(2) : "…"} s</Chip>
                      <Chip color={T.amber} bg={T.amberSoft}>Σ cost ${doneAll ? totalCost.toFixed(3) : "…"}</Chip>
                      <Chip color={T.green} bg={T.greenSoft}>{doneAll ? "1,840 tokens" : "…"}</Chip>
                    </div>
                  </div>
                )}
              </div>
            )}

            {tab === "evaluation" && (
              <div style={{ paddingTop: 12 }}>
                <div style={{ fontSize: 12, color: T.sub, marginBottom: 10, lineHeight: 1.5 }}>
                  Every stage gets its own report (see the left panel) — parsing/chunking/embedding/retrieval use fast, free, rule-based heuristics; this tab is the final generation-stage RAGAS rollup.
                </div>
                <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13, color: T.sub, marginBottom: 12, cursor: "pointer" }}>
                  <input type="checkbox" checked={hasDataset} onChange={e => setHasDataset(e.target.checked)} />
                  Gold dataset attached (enables reference-based metrics)
                </label>
                <div style={{ fontFamily: T.disp, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: T.sub, margin: "6px 0" }}>Reference-free</div>
                {REF_FREE.map(r => (
                  <div key={r.m} style={{ padding: "8px 0" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                      <span>{r.m}</span><b style={{ fontFamily: T.mono, color: scoreColor(r.v) }}>{doneAll ? r.v.toFixed(2) : "–"}</b>
                    </div>
                    <Bar v={doneAll ? r.v : 0} color={scoreColor(r.v)} />
                  </div>
                ))}
                <div style={{ fontFamily: T.disp, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: T.sub, margin: "14px 0 6px" }}>With reference {hasDataset ? "" : "· attach dataset to unlock"}</div>
                <div style={{ opacity: hasDataset ? 1 : .35, pointerEvents: "none" }}>
                  {WITH_REF.map(r => (
                    <div key={r.m} style={{ padding: "8px 0" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                        <span>{r.m}</span><b style={{ fontFamily: T.mono, color: scoreColor(r.v) }}>{hasDataset && doneAll ? r.v.toFixed(2) : "–"}</b>
                      </div>
                      <Bar v={hasDataset && doneAll ? r.v : 0} color={scoreColor(r.v)} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {tab === "ragas" && (
              <div style={{ paddingTop: 12, fontSize: 13, lineHeight: 1.7 }}>
                <div style={{ background: "#F6F8FB", borderRadius: 8, padding: 12, marginBottom: 10 }}>
                  <b style={{ fontFamily: T.disp }}>Final RAGAS report</b> — computed once, at generation, using the real answer. Built on the actual <span style={{ fontFamily: T.mono }}>ragas</span> package.
                </div>
                {doneAll ? (
                  <>
                    <div style={{ fontSize: 12.5, color: T.sub, marginBottom: 8 }}>Reference-free scores (this run):</div>
                    {REF_FREE.map(r => (
                      <div key={r.m} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px dashed ${T.line}` }}>
                        <span>{r.m}</span><b style={{ fontFamily: T.mono, color: scoreColor(r.v) }}>{r.v.toFixed(2)}</b>
                      </div>
                    ))}
                    <div style={{ marginTop: 10, background: T.accentSoft, borderLeft: `3px solid ${T.accent}`, borderRadius: "0 8px 8px 0", padding: "10px 12px", fontSize: 12.5, color: "#155E6E" }}>
                      <b style={{ fontFamily: T.disp }}>Recommendation:</b> {stages[5].rec}
                    </div>
                  </>
                ) : <div style={{ color: T.sub, textAlign: "center", padding: "16px 0" }}>Run the pipeline to see the RAGAS rollup here.</div>}
              </div>
            )}

            {tab === "compare" && (
              <div style={{ paddingTop: 12, fontSize: 13, lineHeight: 1.7 }}>
                <div style={{ background: "#F6F8FB", borderRadius: 8, padding: 12 }}>
                  <b style={{ fontFamily: T.disp }}>A/B pipeline compare</b> — run two configs on the same query and diff traces, scores, cost and latency side-by-side.
                </div>
                <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
                  {[["A · Semantic only", "0.74 faithful · $0.011 · 5.4s"], ["B · Hybrid-RRF + rerank", "0.91 faithful · $0.013 · 5.9s"]].map(([n, s]) => (
                    <div key={n} style={{ flex: 1, minWidth: 150, border: `1px solid ${T.line}`, borderRadius: 8, padding: 10 }}>
                      <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 13 }}>{n}</div>
                      <div style={{ fontFamily: T.mono, fontSize: 11.5, color: T.sub, marginTop: 4 }}>{s}</div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 10, color: T.green, fontSize: 12.5 }}>▲ +0.17 faithfulness for +$0.002 — the lesson writes itself.</div>
              </div>
            )}
          </div>

          <div style={{ marginTop: 12, fontSize: 11.5, color: T.sub, lineHeight: 1.6 }}>
            Mockup: outputs are simulated. Real backend (<span style={{ fontFamily: T.mono }}>core/</span>) is built and tested — see Milestone 2 README.
          </div>
        </div>
      </div>

      {/* Manage options modal */}
      {manageIdx !== null && (
        <div onClick={() => setManageIdx(null)} style={{ position: "fixed", inset: 0, background: "rgba(15,27,45,.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }}>
          <div onClick={e => e.stopPropagation()} style={{ background: "#fff", borderRadius: 14, padding: 20, width: 340, maxWidth: "90vw" }}>
            <div style={{ fontFamily: T.disp, fontWeight: 700, fontSize: 16 }}>Manage {stages[manageIdx].label} options</div>
            <div style={{ marginTop: 12 }}>
              {stages[manageIdx].options.map((o, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 0", borderBottom: `1px dashed ${T.line}`, fontSize: 13 }}>
                  {o}
                  <button onClick={() => setStages(p => p.map((s, j) => j === manageIdx ? { ...s, options: s.options.filter((_, k) => k !== i), sel: 0 } : s))}
                    disabled={stages[manageIdx].options.length <= 1}
                    style={{ border: "none", background: "none", color: T.red, cursor: "pointer", fontSize: 12 }}>remove</button>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <input value={newOpt} onChange={e => setNewOpt(e.target.value)} placeholder="New option name…"
                style={{ flex: 1, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13 }} />
              <button onClick={() => { if (newOpt.trim()) { setStages(p => p.map((s, j) => j === manageIdx ? { ...s, options: [...s.options, newOpt.trim()] } : s)); setNewOpt(""); } }}
                style={{ padding: "8px 14px", borderRadius: 8, border: "none", background: T.accent, color: "#fff", cursor: "pointer", fontSize: 13 }}>Add</button>
            </div>
            <div style={{ fontSize: 11.5, color: T.sub, marginTop: 10 }}>In the real build, adding an option maps to dropping a registered class into <span style={{ fontFamily: T.mono }}>core/{stages[manageIdx].id}/</span> with a <span style={{ fontFamily: T.mono }}>@register(...)</span> decorator.</div>
          </div>
        </div>
      )}
    </div>
  );
}
