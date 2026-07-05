import { useEffect, useRef, useState, isValidElement, Fragment } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { getRegistry, getKbs, deleteKb, buildKb, runQuery, rewritePrompt, suggestOption, diagnoseDeep, kbImageUrl, getAvailableProviders } from "./api.js";

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

// ---------- Capability tags: which real registry ids work on text vs text+images ----------
const CAPABILITY = {
  parsing: { assorted: "both", pdf: "both", docx: "both", pptx: "both", md: "both", txt: "both" },
  chunking: { recursive: "text", fixed: "text", semantic: "text", sentence: "text", markdown_structure: "text" },
  embedding: { "minilm-l6": "text", "bge-small": "text", "gemini-text-embedding": "text",
               "openai-text-embedding-3-small": "text", "openai-text-embedding-3-large": "text" },
  retrieval: { semantic: "both", keyword: "text", "hybrid-rrf": "both" },
  reranking: { none: "both", "cross-encoder": "text" },
  generation: { "claude-sonnet": "both", "gemini-2.5-flash": "both", "gemini-2.5-pro": "both", "gpt-4o-mini": "both",
                "llama-3.1-8b-instant": "text", "llama-4-scout": "both" },
};
const CapTag = ({ cap }) => (
  <span style={{
    fontFamily: T.mono, fontSize: 9.5, padding: "1px 6px", borderRadius: 99, marginLeft: 6,
    background: cap === "both" ? T.greenSoft : "#EEF1F6", color: cap === "both" ? T.green : T.sub,
  }}>{cap === "both" ? "text+image" : "text only"}</span>
);

const STAGE_ORDER = ["parsing", "chunking", "embedding", "retrieval", "reranking", "react", "generation"];
// Build-time stages live on the Knowledge Base tab; query-time stages on the Query Lab tab.
const BUILD_STAGES = ["parsing", "chunking", "embedding"];
const QUERY_STAGES = ["retrieval", "reranking", "react", "generation"];
const STAGE_LABELS = {
  parsing: "Parsing", chunking: "Chunking", embedding: "Embedding", retrieval: "Retrieval",
  reranking: "Re-ranking", react: "ReAct loop", generation: "Generation",
};

const DEFAULT_CONFIG = {
  parsing_method: "assorted", extract_images: true, ocr: false,
  chunking_method: "recursive", chunk_size: 512, chunk_overlap: 64,
  embedding_method: "minilm-l6", image_embedding_method: "caption-text-embed", caption_provider: "claude",
  retrieval_method: "hybrid-rrf", top_k: 10, result_mode: "text-only", web_enabled: false,
  reranking_method: "cross-encoder", rerank_keep_top: 4, vision_rerank_enabled: false,
  react_enabled: false, react_max_iterations: 3, react_judge_method: "gemini-2.5-flash",
  generation_method: "claude-sonnet", vision_grounded: false,
  temperature: 0.7, top_p: 1.0, gen_top_k: 40, max_tokens: 1024,
};

// Which provider backs each generation option -- openai/groq's chat-completions
// API has no top_k knob, so that field is disabled (not silently dropped) for them.
const GEN_PROVIDER = {
  "claude-sonnet": "claude", "gemini-2.5-flash": "gemini", "gemini-2.5-pro": "gemini",
  "gpt-4o-mini": "openai", "llama-3.1-8b-instant": "groq", "llama-4-scout": "groq",
};

const RESULT_MODES = ["text-only", "joint", "separate-merge"];
const RESULT_MODE_LABELS = { "text-only": "Text only", joint: "Text + Images (joint rank)", "separate-merge": "Text + Images (separate, merge top-k)" };
const IMAGE_EMBED_OPTIONS = ["caption-text-embed", "clip-local"];
const PROVIDERS = ["claude", "gemini", "openai", "groq"];
const REWRITE_GUIDELINES = {
  claude: "Best for precise, structure-preserving rewrites — use when the prompt already has a shape you want kept.",
  gemini: "Fastest and cheapest — use for quick iteration when you'll throw away most drafts.",
  openai: "Good second opinion with a different house style from Claude/Gemini.",
};

const LEARN = {
  parsing: {
    concept: "Every format becomes one shape: text blocks + images + tables, each tagged with page/section. 'Assorted' just looks at the file extension and routes to the right parser — same output schema either way.",
    table: [["assorted", "Mixed corpus (pdf+docx+pptx+md together) — the default for real use", "both"], ["format-specific", "Single-format corpus, slightly simpler traces", "both"], ["OCR toggle", "Turn on when pages are scans/photos of text, not real text layers", "both"]],
    demo: "Toy: a 1-page scanned PDF with no text layer → OCR off gives 0 words; OCR on recovers the sentence via Tesseract, tagged source_ocr:true so you always know which text came from where.",
  },
  chunking: {
    concept: "Chunk size trades off two errors: too large dilutes the embedding (low precision); too small splits a fact across chunks (low recall). Markdown structure groups by heading instead of a token window.",
    table: [["recursive", "Safe general default; overlap protects boundary splits", "text"], ["fixed", "Short, uniform, fact-dense text (FAQs)", "text"], ["semantic", "Long, structurally loose prose (essays, transcripts)", "text"], ["markdown_structure", "Docs where headings carry real structure (specs, wikis)", "text"]],
    demo: "chunk_size/overlap are manual inputs (default 512/64 tokens) — toy: one paragraph split at 256 vs 512 shows a mid-sentence cut appear and disappear.",
  },
  embedding: {
    concept: "Embeddings map text to a vector so cosine similarity ≈ semantic similarity. Local models are free; API embeddings cost per call. Once a KB is built, the embedding model is LOCKED — query time always reuses whatever built the KB.",
    table: [["minilm-l6", "Default for ≤20 files — free, fast, private", "text"], ["gemini / openai-*", "Demonstrating API-embedding cost/quality trade-off — needs that provider's key", "text"], ["caption-text-embed", "Images: caption once via vision LLM, embed in the SAME space as text", "both"], ["clip-local", "Pure visual similarity ('find charts like this') — its own space, needs its own query encoder", "both"]],
    demo: "Toy: 'car' vs 'automobile' vs 'banana' → cosine ≈0.82, 0.82, 0.11. The vector space encodes meaning, not just words.",
  },
  retrieval: {
    concept: "Semantic ranks by embedding similarity; Keyword (BM25) ranks by weighted term overlap. Hybrid-RRF fuses both by RANK, not raw score. When the web toggle is on, KB results are blended with live Tavily search the same way — RRF across KB and web rankings.",
    table: [["semantic", "Conceptual/paraphrased queries", "both (with caption/CLIP images)"], ["keyword", "Exact terms, codes, acronyms embeddings blur", "text"], ["hybrid-rrf", "Real queries mix both — usually the best default", "both"], ["Result mode", "text-only / joint (needs shared embedding space) / separate-merge (always safe)", "both"]],
    demo: "Toy query 'BLEU score formula' — Semantic alone ranks a paraphrase above the exact formula chunk; Hybrid-RRF fixes it.",
  },
  reranking: {
    concept: "Retrieval optimizes for recall (cast a wide net cheaply); reranking optimizes precision on that smaller set. Cross-encoders score (query, chunk) jointly instead of comparing two frozen vectors. Images compete for the SAME 'Keep top' budget as text — an image's caption is scored through the identical cross-encoder as a text surrogate, so a query with no visual intent doesn't automatically drag images into the answer.",
    table: [["none", "Retrieval is already precise, or teaching the baseline", "both"], ["cross-encoder", "Best precision-per-rupee, zero API cost — images scored via caption surrogate", "both"], ["Vision-LLM relevance (toggle)", "Extra pass judging images by what they actually show, not just the caption — real per-query cost", "images"]],
    demo: "Toy: 8 retrieved chunks + 3 images, cross-encoder flips rank 5 to rank 1 because it reads query+chunk together, and drops 2 of the 3 images that don't clear the shared keep-top bar.",
  },
  react: {
    concept: "After reranking, a judge LLM checks whether the context can actually answer the query. If not, it rewrites/narrows the query and re-runs retrieval → reranking, up to a capped number of iterations. The web toggle applies here too — if on, web is re-queried on EVERY iteration, not just the first pass, so an aggressive loop's real cost shows up in the trace.",
    table: [["Off (default)", "Pipeline behaves exactly like a plain RAG run — zero added cost or latency", "n/a"], ["On, judge = Gemini Flash", "Cheap enough to run every iteration without dominating run cost", "n/a"], ["Max iterations", "Hard cap (1-5, default 3) so a stubborn query can't loop forever", "n/a"]],
    demo: "Toy: reranked context scores low → judge narrows 'how transformers work' to 'self-attention formula transformer' → second retrieval pass finds the exact formula chunk.",
  },
  generation: {
    concept: "Grounding forces the model to answer from retrieved context, not memory. Vision-grounded generation sends actual image bytes instead of just the caption — costs more, lets the model read the figure itself.",
    table: [["claude-sonnet", "Strongest grounded, citation-style answers", "both"], ["gemini-2.5-flash", "~8× cheaper, good for high-volume/demo use", "both"], ["Vision-grounded off (default)", "Caption text is enough — cheaper, faster", "both"], ["Vision-grounded on", "Answer genuinely depends on reading the image", "both"]],
    demo: "Toy: caption-only gets a chart's trend right but invents an exact number; vision-grounded reads it correctly.",
  },
};

const REF_FREE_LABELS = { faithfulness: "Faithfulness", answer_relevancy: "Answer relevancy", context_precision_without_reference: "Context precision" };

const DEFAULT_PROMPT = "You are a helpful assistant. Answer using ONLY the provided context. Cite text chunk ids like [#47], and image ids like [img#12] whenever a figure actually informed your answer -- omit an image tag if you didn't rely on it. If the context is insufficient, say so.";

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
const NumberField = ({ label, value, onChange, min = 1, max = 4096, step = 1, disabled = false }) => (
  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: disabled ? "#B7C1CC" : T.sub }}>
    {label}
    <input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
      onChange={e => onChange(+e.target.value)}
      style={{ width: 68, padding: "4px 6px", borderRadius: 6, border: `1px solid ${T.line}`, fontFamily: T.mono, fontSize: 12,
               background: disabled ? "#F3F5F8" : "#fff", color: disabled ? "#B7C1CC" : T.ink }} />
  </label>
);

// ---------- Eval report block (shown after every stage) ----------
const EXAMPLE_LABELS = { top: "Top", bottom: "Bottom", largest: "Largest", smallest: "Smallest" };

function EvalReport({ evalFree, needsAck, acknowledged, onAck }) {
  if (!evalFree) return null;
  const scores = evalFree.scores || {};
  const examples = evalFree.examples || {};
  return (
    <div style={{ marginTop: 10, background: "#F6F8FB", border: `1px solid ${T.line}`, borderRadius: 10, padding: 12 }}>
      <div style={{ fontFamily: T.disp, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: T.sub, marginBottom: 6 }}>Reference-free report</div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontSize: 12 }}>
        {Object.entries(scores).map(([k, v]) => (
          <span key={k} style={{ fontFamily: T.mono }}>
            {REF_FREE_LABELS[k] || k}: <b style={{ color: typeof v === "number" && v <= 1 ? scoreColor(v) : T.ink }}>{typeof v === "number" ? v : String(v)}</b>
          </span>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 12.5, color: "#155E6E", lineHeight: 1.5 }}>
        <b style={{ fontFamily: T.disp }}>Recommendation:</b> {evalFree.recommendation}
      </div>
      {Object.keys(examples).length > 0 && (
        <div style={{ marginTop: 8, display: "flex", gap: 10, flexWrap: "wrap" }}>
          {Object.entries(examples).map(([k, ex]) => (
            <div key={k} style={{ flex: "1 1 220px", fontSize: 11.5, background: "#fff", border: `1px solid ${T.line}`, borderRadius: 8, padding: "6px 8px" }}>
              <span style={{ fontFamily: T.disp, color: T.sub }}>{EXAMPLE_LABELS[k] || k}{ex.score != null ? ` (${ex.score})` : ex.n_tokens != null ? ` (${ex.n_tokens} tok)` : ""}:</span>{" "}
              <span style={{ color: T.ink }}>{ex.text}</span>
            </div>
          ))}
        </div>
      )}
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

// ---------- Structured error card -- StageError {stage, method, what_failed, hint} ----------
function ErrorCard({ error }) {
  if (!error) return null;
  return (
    <div style={{ marginTop: 10, background: T.redSoft, border: `1px solid #F3B9AF`, borderRadius: 10, padding: 12 }}>
      <div style={{ fontFamily: T.disp, fontSize: 12.5, fontWeight: 700, color: T.red }}>✗ {error.method} failed</div>
      <div style={{ marginTop: 4, fontSize: 12.5, color: T.red }}>{error.what_failed}</div>
      <div style={{ marginTop: 6, fontSize: 12.5, color: "#7A2E22" }}>→ {error.hint}</div>
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
          <span style={{ fontFamily: T.mono, fontWeight: 700, minWidth: 140, color: T.ink }}>{opt}{cap && cap !== "n/a" && <CapTag cap={cap === "text" ? "text" : "both"} />}</span>
          <span style={{ color: T.sub }}>{when}</span>
        </div>
      ))}
      <div style={{ fontFamily: T.disp, fontSize: 11.5, letterSpacing: 1, textTransform: "uppercase", color: T.accent, margin: "12px 0 6px" }}>Toy demo</div>
      <div style={{ fontSize: 12.5, lineHeight: 1.6, color: T.sub, fontStyle: "italic" }}>{content.demo}</div>
    </div>
  );
}

// ---------- Per-stage data panel: parsed docs / chunks / retrieved & reranked items ----------
function ItemsRow({ label, meta, text, rationale }) {
  const [open, setOpen] = useState(false);
  const preview = text.length > 150 ? text.slice(0, 150) + "…" : text;
  return (
    <div onClick={() => setOpen(o => !o)} style={{ padding: "8px 10px", borderBottom: `1px solid ${T.line}`, cursor: "pointer" }}>
      <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
        <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 700, color: T.ink }}>{label}</span>
        {meta.map((m, i) => isValidElement(m) ? <Fragment key={i}>{m}</Fragment> : <Chip key={i}>{m}</Chip>)}
        <span style={{ marginLeft: "auto", fontFamily: T.mono, fontSize: 10, color: T.sub }}>{open ? "▾" : "▸"}</span>
      </div>
      {rationale && <div style={{ marginTop: 3, fontSize: 11, color: T.sub, fontStyle: "italic" }}>{rationale}</div>}
      <div style={{ marginTop: 4, fontSize: 12, color: T.sub, whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
        {open ? text : preview}
      </div>
    </div>
  );
}

function ItemsPanel({ id, items }) {
  return (
    <div style={{ margin: "6px 0 0", background: "#FAFCFE", border: `1px solid ${T.line}`, borderRadius: 8, maxHeight: 360, overflowY: "auto" }}>
      {items.map((it, i) => {
        if (id === "parsing") {
          return <ItemsRow key={it.doc_id} label={it.doc_id} meta={[it.format, `${it.n_pages} pages`, `${it.n_words} words`]} text={it.text} />;
        }
        if (id === "chunking") {
          return <ItemsRow key={it.chunk_id} label={it.chunk_id} meta={[it.doc_id, it.page != null ? `page ${it.page}` : null, `${it.n_tokens} tok`].filter(Boolean)} text={it.text} />;
        }
        // retrieval / reranking
        const meta = [`score ${it.score}`, it.kind];
        if (it.rank_change != null) {
          const up = it.rank_change > 0;
          meta.push(
            <Chip key="rc" color={up ? "#1B7A4A" : it.rank_change < 0 ? "#B4483A" : T.accent}
                  bg={up ? "#E4F5EC" : it.rank_change < 0 ? "#FBEAE7" : T.accentSoft}>
              {up ? `↑${it.rank_change}` : it.rank_change < 0 ? `↓${-it.rank_change}` : "–"}
            </Chip>
          );
        }
        const rationale = it.matched_terms?.length ? `matched: ${it.matched_terms.join(", ")}` : null;
        return <ItemsRow key={`${it.id}-${i}`} label={`#${it.rank} ${it.id}`} meta={meta} text={it.text} rationale={rationale} />;
      })}
    </div>
  );
}

// ---------- Image gallery: thumbnails for images actually cited in the answer ----------
function citedImageIds(answerText) {
  const ids = new Set();
  const re = /\[img#([^\]]+)\]/g;
  let m;
  while ((m = re.exec(answerText || ""))) ids.add(m[1]);
  return ids;
}

function ImageGallery({ kbName, items, answerText }) {
  const images = items.filter(it => it.kind === "image");
  if (images.length === 0) return null;
  const cited = citedImageIds(answerText);
  // Strictly citation-gated: an image renders only when the answer cites its
  // [img#id] -- i.e. the model says the figure substantiated the answer. No
  // fallback: if nothing is cited, nothing is shown.
  const shown = images.filter(img => cited.has(img.id));
  if (shown.length === 0) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontFamily: T.disp, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: T.sub, marginBottom: 6 }}>
        Supporting figures
      </div>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {shown.map(img => (
          <div key={img.id} style={{ width: 160 }}>
            <img src={kbImageUrl(kbName, img.id)} alt={img.text || img.id}
              style={{ width: "100%", height: 110, objectFit: "contain", borderRadius: 8, border: `1px solid ${T.line}`, background: "#EEF1F6" }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- Stage card ----------
function StageCard({ id, idx, method, methodOptions, status, error, onSelectMethod, onSuggest, suggestion, onManage,
                     trace, expanded, onToggle, learnOpen, onLearnToggle, config, onConfigChange, runMode,
                     acknowledged, onAck, output }) {
  const live = status === "running", done = status === "done", skipped = status === "skipped";
  const needsAck = runMode === "step" && done && !error;
  const label = STAGE_LABELS[id];
  const [dataOpen, setDataOpen] = useState(false);
  const hasItems = ["parsing", "chunking", "retrieval", "reranking"].includes(id) && output?.items?.length > 0;

  return (
    <div style={{ display: "flex", gap: 14, opacity: skipped ? 0.5 : 1 }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 26 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 99, display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: T.mono, fontSize: 10, fontWeight: 700, flexShrink: 0, transition: "all .3s",
          background: error ? T.red : done ? T.accent : live ? T.ink : "#fff",
          color: done || live || error ? "#fff" : T.sub,
          border: `2px solid ${error ? T.red : done || live ? T.accent : T.line}`,
          boxShadow: live ? `0 0 0 5px ${T.accentSoft}` : "none",
        }}>{error ? "✗" : done ? "✓" : idx + 1}</div>
        <div style={{ flex: 1, width: 2, background: done ? T.accent : T.line, transition: "background .4s", minHeight: 18 }} />
      </div>

      <div style={{ flex: 1, background: T.card, border: `1px solid ${live ? T.accent : T.line}`, borderRadius: 12, padding: 14, marginBottom: 14, transition: "border .3s" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15, color: T.ink }}>{label}</div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            {done && output && <Chip>{output.lat} ms</Chip>}
            {done && output && output.cost > 0 && <Chip color={T.amber} bg={T.amberSoft}>${output.cost.toFixed(4)}</Chip>}
            <button onClick={() => onLearnToggle(id)} title="Learn: theory, guidelines, demo"
              style={{ width: 24, height: 24, borderRadius: 99, border: `1px solid ${learnOpen ? T.accent : T.line}`, background: learnOpen ? T.accent : "#fff", color: learnOpen ? "#fff" : T.sub, cursor: "pointer", fontSize: 12, fontFamily: T.disp, lineHeight: 1 }}>?</button>
          </div>
        </div>

        {id === "react" ? (
          <div style={{ marginTop: 10 }}>
            <Toggle label="Enable ReAct refinement loop" checked={config.react_enabled} onChange={() => onConfigChange("react_enabled", !config.react_enabled)} />
          </div>
        ) : (
          <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
            <select value={method} onChange={e => onSelectMethod(e.target.value)}
              style={{ flex: 1, minWidth: 160, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.body, fontSize: 13, color: T.ink, background: "#FAFCFE" }}>
              {methodOptions.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
            <button onClick={onManage} title="Add / remove options"
              style={{ padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, background: "#fff", cursor: "pointer", fontSize: 13, color: T.sub }}>⚙︎</button>
            <button onClick={onSuggest}
              style={{ padding: "8px 12px", borderRadius: 8, border: "none", background: T.ink, color: "#fff", cursor: "pointer", fontSize: 12, fontFamily: T.disp }}>✦ Suggest</button>
          </div>
        )}
        {CAPABILITY[id] && <div style={{ marginTop: 4 }}><CapTag cap={CAPABILITY[id]?.[method] || "text"} /></div>}

        {/* Stage-specific manual controls */}
        {id === "parsing" && (
          <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap", alignItems: "center" }}>
            <Toggle label="Extract images" checked={config.extract_images} onChange={() => onConfigChange("extract_images", !config.extract_images)} />
            <Toggle label="OCR scanned pages (Tesseract)" checked={config.ocr} onChange={() => onConfigChange("ocr", !config.ocr)} />
          </div>
        )}
        {id === "parsing" && config.extract_images && (
          <div style={{ marginTop: 8, background: T.greenSoft, borderRadius: 8, padding: "8px 10px", fontSize: 12, color: "#065F46" }}>
            Image embedding method: <select value={config.image_embedding_method} onChange={e => onConfigChange("image_embedding_method", e.target.value)}
              style={{ marginLeft: 6, fontSize: 12, padding: "3px 6px", borderRadius: 6, border: `1px solid ${T.line}` }}>
              {IMAGE_EMBED_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
            {config.image_embedding_method === "caption-text-embed" && (
              <span style={{ marginLeft: 14 }}>
                Caption provider: <select value={config.caption_provider} onChange={e => onConfigChange("caption_provider", e.target.value)}
                  style={{ marginLeft: 6, fontSize: 12, padding: "3px 6px", borderRadius: 6, border: `1px solid ${T.line}` }}>
                  {PROVIDERS.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
              </span>
            )}
          </div>
        )}
        {id === "chunking" && (
          <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
            <NumberField label="Chunk size (tokens)" value={config.chunk_size} onChange={v => onConfigChange("chunk_size", v)} min={16} max={4096} />
            <NumberField label="Overlap (tokens)" value={config.chunk_overlap} onChange={v => onConfigChange("chunk_overlap", v)} min={0} max={1024} />
          </div>
        )}
        {id === "embedding" && config.kbLocked && (
          <div style={{ marginTop: 8, fontSize: 11.5, color: T.sub, fontStyle: "italic" }}>
            Locked to this KB's embedding model — query time always reuses whatever built the KB, never a dropdown.
          </div>
        )}
        {id === "retrieval" && (
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
            <NumberField label="Top K" value={config.top_k} onChange={v => onConfigChange("top_k", v)} min={1} max={50} />
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: T.sub, flexWrap: "wrap" }}>
              Result mode:
              <select value={config.result_mode} onChange={e => onConfigChange("result_mode", e.target.value)}
                style={{ fontSize: 12.5, padding: "5px 8px", borderRadius: 6, border: `1px solid ${T.line}`, flex: 1, minWidth: 200 }}>
                {RESULT_MODES.map(o => <option key={o} value={o}>{RESULT_MODE_LABELS[o]}</option>)}
              </select>
            </div>
          </div>
        )}
        {id === "reranking" && (
          <div style={{ marginTop: 8 }}>
            <NumberField label="Keep top" value={config.rerank_keep_top} onChange={v => onConfigChange("rerank_keep_top", v)} min={1} max={20} />
            <div style={{ marginTop: 8 }}>
              <Toggle label="Vision-LLM relevance scoring for images (extra cost/latency)" checked={config.vision_rerank_enabled} onChange={() => onConfigChange("vision_rerank_enabled", !config.vision_rerank_enabled)} />
            </div>
          </div>
        )}
        {id === "react" && config.react_enabled && (
          <div style={{ marginTop: 10 }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
              <select value={config.react_judge_method} onChange={e => onConfigChange("react_judge_method", e.target.value)}
                style={{ flex: 1, minWidth: 160, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.body, fontSize: 13, color: T.ink, background: "#FAFCFE" }}>
                {methodOptions.map(o => <option key={o} value={o}>{o} {o === "gemini-2.5-flash" ? "(default)" : ""}</option>)}
              </select>
            </div>
            <NumberField label="Max iterations" value={config.react_max_iterations} onChange={v => onConfigChange("react_max_iterations", v)} min={1} max={5} />
          </div>
        )}
        {id === "generation" && (
          <div style={{ marginTop: 10 }}>
            <Toggle label="Vision-grounded generation (send image, not just caption)" checked={config.vision_grounded} onChange={() => onConfigChange("vision_grounded", !config.vision_grounded)} />
            <div style={{ display: "flex", gap: 16, marginTop: 10, flexWrap: "wrap" }}>
              <NumberField label="Temperature" value={config.temperature} onChange={v => onConfigChange("temperature", v)} min={0} max={2} step={0.1} />
              <NumberField label="Top-p" value={config.top_p} onChange={v => onConfigChange("top_p", v)} min={0} max={1} step={0.05} />
              <NumberField label="Top-k" value={config.gen_top_k} onChange={v => onConfigChange("gen_top_k", v)} min={1} max={500}
                disabled={["openai", "groq"].includes(GEN_PROVIDER[method])} />
              <NumberField label="Max tokens" value={config.max_tokens} onChange={v => onConfigChange("max_tokens", v)} min={64} max={8192} step={64} />
            </div>
            {["openai", "groq"].includes(GEN_PROVIDER[method]) && (
              <div style={{ marginTop: 6, fontSize: 11, color: T.sub, fontStyle: "italic" }}>
                Top-k is disabled — OpenAI/Groq's chat API has no top-k sampling parameter.
              </div>
            )}
          </div>
        )}

        {suggestion && (
          <div style={{ marginTop: 10, background: T.accentSoft, borderLeft: `3px solid ${T.accent}`, borderRadius: "0 8px 8px 0", padding: "8px 12px", fontSize: 12.5, color: "#155E6E", lineHeight: 1.5 }}>
            <b style={{ fontFamily: T.disp }}>Advisor:</b> {suggestion}
          </div>
        )}

        {learnOpen && <LearnDrawer content={LEARN[id]} />}

        {(live || done || error) && (
          <div style={{ marginTop: 10 }}>
            <button onClick={onToggle} style={{ background: "none", border: "none", cursor: "pointer", fontFamily: T.mono, fontSize: 11, color: T.accent, padding: 0 }}>
              {live ? "● streaming…" : expanded ? "▾ trace" : "▸ trace"}
            </button>
            {done && hasItems && (
              <button onClick={() => setDataOpen(o => !o)} style={{ background: "none", border: "none", cursor: "pointer", fontFamily: T.mono, fontSize: 11, color: T.accent, padding: 0, marginLeft: 14 }}>
                {dataOpen ? "▾ data" : "▸ data"}
              </button>
            )}
            {(done || error) && expanded && (
              <pre style={{ margin: "6px 0 0", background: "#0F1B2D", color: "#C7E5EE", borderRadius: 8, padding: 12, fontFamily: T.mono, fontSize: 11.5, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{trace}</pre>
            )}
            {done && hasItems && dataOpen && <ItemsPanel id={id} items={output.items} />}
            {error && <ErrorCard error={error} />}
            {done && output && <EvalReport evalFree={output.evalFree} needsAck={needsAck} acknowledged={acknowledged} onAck={onAck} />}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------- KB picker ----------
function KbPicker({ kbs, selectedKb, onSelect, onDelete, creating, onStartCreate, onCancelCreate,
                    newKbName, onNewKbName, newFiles, onNewFiles, uploadError, onBuild, building }) {
  return (
    <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14, marginBottom: 16 }}>
      <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15, marginBottom: 10 }}>Knowledge Base</div>
      {!creating ? (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <select value={selectedKb} onChange={e => onSelect(e.target.value)}
            style={{ flex: 1, minWidth: 160, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13, background: "#FAFCFE" }}>
            <option value="">— choose a KB —</option>
            {kbs.map(kb => <option key={kb.name} value={kb.name}>{kb.name} ({kb.n_chunks} chunks, {kb.embedding_method})</option>)}
          </select>
          {selectedKb && (
            <button onClick={() => onDelete(selectedKb)}
              style={{ padding: "8px 12px", borderRadius: 8, border: `1px solid ${T.line}`, background: "#fff", color: T.red, cursor: "pointer", fontSize: 12.5 }}>
              Delete
            </button>
          )}
          <button onClick={onStartCreate}
            style={{ padding: "8px 12px", borderRadius: 8, border: "none", background: T.accent, color: "#fff", cursor: "pointer", fontSize: 12.5, fontFamily: T.disp }}>
            + New KB
          </button>
        </div>
      ) : (
        <div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <input value={newKbName} onChange={e => onNewKbName(e.target.value)} placeholder="KB name (letters, numbers, - _)"
              style={{ flex: 1, minWidth: 160, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13 }} />
            <button onClick={onCancelCreate}
              style={{ padding: "8px 12px", borderRadius: 8, border: `1px solid ${T.line}`, background: "#fff", cursor: "pointer", fontSize: 12.5 }}>Cancel</button>
          </div>
          <input type="file" multiple accept=".pdf,.docx,.pptx,.md,.markdown,.txt"
            onChange={e => onNewFiles(Array.from(e.target.files))}
            style={{ marginTop: 10, fontSize: 12.5 }} />
          <div style={{ fontSize: 11.5, color: T.sub, marginTop: 6 }}>≤20 files, ≤25MB/file, ≤150MB total. PDF/DOCX/PPTX/MD/TXT.</div>
          {newFiles.length > 0 && (
            <div style={{ fontSize: 12, color: T.sub, marginTop: 6 }}>{newFiles.length} file(s) selected, {(newFiles.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(1)}MB total</div>
          )}
          {uploadError && <div style={{ marginTop: 8, color: T.red, fontSize: 12.5 }}>{uploadError}</div>}
          <button onClick={onBuild} disabled={building || !newKbName || newFiles.length === 0}
            style={{ marginTop: 10, padding: "8px 16px", borderRadius: 8, border: "none",
                     background: building ? T.sub : T.accent, color: "#fff", cursor: building ? "default" : "pointer",
                     fontSize: 12.5, fontFamily: T.disp, fontWeight: 600 }}>
            {building ? "Building…" : "Build KB"}
          </button>
        </div>
      )}
    </div>
  );
}

const MAX_FILES = 20, MAX_FILE_BYTES = 25 * 1e6, MAX_TOTAL_BYTES = 150 * 1e6;
function validateUpload(files) {
  if (files.length === 0) return "Attach at least one document.";
  if (files.length > MAX_FILES) return `Too many files (${files.length}) — max ${MAX_FILES} per KB.`;
  const oversized = files.find(f => f.size > MAX_FILE_BYTES);
  if (oversized) return `'${oversized.name}' is ${(oversized.size / 1e6).toFixed(1)}MB — max 25MB per file.`;
  const total = files.reduce((a, f) => a + f.size, 0);
  if (total > MAX_TOTAL_BYTES) return `Corpus totals ${(total / 1e6).toFixed(1)}MB — max 150MB per KB.`;
  return "";
}

// ---------- App ----------
export default function RAGLab() {
  const [registry, setRegistry] = useState(null);
  const [kbs, setKbs] = useState([]);
  const [selectedKb, setSelectedKb] = useState("");
  const [creatingKb, setCreatingKb] = useState(false);
  const [newKbName, setNewKbName] = useState("");
  const [newFiles, setNewFiles] = useState([]);
  const [uploadError, setUploadError] = useState("");
  const [building, setBuilding] = useState(false);

  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [runtime, setRuntime] = useState({});     // { [stageId]: { status, output, error } }
  const [expanded, setExpanded] = useState({});
  const [learnOpen, setLearnOpen] = useState({});
  const [acks, setAcks] = useState({});
  const [suggestions, setSuggestions] = useState({});
  const [manageStage, setManageStage] = useState(null);
  const [registryOverrides, setRegistryOverrides] = useState({}); // local-only add/remove of dropdown options
  const [newOpt, setNewOpt] = useState("");

  const [keysOpen, setKeysOpen] = useState(false);
  const [keys, setKeys] = useState({ claude: "", gemini: "", openai: "", groq: "", tavily: "" });
  const [envProviders, setEnvProviders] = useState({}); // { [provider]: true } for providers with a key in the server's .env
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [rewriteProvider, setRewriteProvider] = useState("claude");
  const [rewriting, setRewriting] = useState(false);
  const [query, setQuery] = useState("How does self-attention work in the Transformer paper?");
  const [messages, setMessages] = useState([]);       // [{role: "user"|"assistant", content}] -- prior turns
  const [historyTurns, setHistoryTurns] = useState(3); // how many prior Q/A pairs to send with each query
  const [running, setRunning] = useState(false);
  const [runMode, setRunMode] = useState("all"); // "all" | "step"
  const [mainTab, setMainTab] = useState("kb"); // "kb" | "lab" | "analysis"
  const [historyOpen, setHistoryOpen] = useState(true);
  const [tab, setTab] = useState("trace");
  const [hasDataset, setHasDataset] = useState(false);
  const [reference, setReference] = useState("");
  const [finalSummary, setFinalSummary] = useState(null); // { answer, total_cost_usd, total_latency_ms, total_tokens, diagnosis, stage_reports }
  const [deepDive, setDeepDive] = useState(null); // { loading, text, error }

  const pacingQueue = useRef([]);
  const readyToReveal = useRef(true);

  useEffect(() => {
    getRegistry().then(setRegistry).catch(() => setRegistry({}));
    getKbs().then(setKbs).catch(() => setKbs([]));
    getAvailableProviders().then(setEnvProviders).catch(() => setEnvProviders({}));
  }, []);

  const methodOptionsFor = (stageId) => registryOverrides[stageId] || registry?.[stageId] || [];
  // A provider is usable either because the user pasted a key in this session
  // or because the server already has one configured via .env -- only when
  // neither is true do we need to interrupt and ask for a key.
  const hasKey = (provider) => !!keys[provider] || !!envProviders[provider];

  const onConfigChange = (key, value) => setConfig(p => ({ ...p, [key]: value }));

  const resetRuntime = () => {
    setRuntime({});
    setExpanded({});
    setAcks({});
    setFinalSummary(null);
    setDeepDive(null);
    pacingQueue.current = [];
    readyToReveal.current = true;
  };

  // ---- KB selection / creation ----
  const selectKb = (name) => {
    setSelectedKb(name);
    if (!name) return;
    const kb = kbs.find(k => k.name === name);
    if (!kb) return;
    setConfig(p => ({ ...p, embedding_method: kb.embedding_method, kbLocked: true }));
    setRuntime(p => ({
      ...p,
      parsing: { status: "done", output: { out: `${kb.n_docs} docs loaded from saved KB`, lat: 0, cost: 0, evalFree: null } },
      chunking: { status: "done", output: { out: `${kb.n_chunks} chunks (already built)`, lat: 0, cost: 0, evalFree: null } },
      embedding: { status: "done", output: { out: `${kb.n_chunks} vectors, dim ${kb.embedding_dim}`, lat: 0, cost: 0, evalFree: null } },
    }));
  };

  const startCreateKb = () => { setCreatingKb(true); setNewKbName(""); setNewFiles([]); setUploadError(""); };
  const cancelCreateKb = () => setCreatingKb(false);

  const handleNewFiles = (files) => {
    setNewFiles(files);
    setUploadError(validateUpload(files));
  };

  const applyEvent = (item, targetSetter) => {
    if (item.type === "report") {
      targetSetter(item);
    } else if (item.type === "error") {
      targetSetter(item);
    }
    // {"type": "done", ...} is consumed by the caller directly, not through this path
  };

  const doBuildKb = async () => {
    const err = validateUpload(newFiles);
    if (err) { setUploadError(err); return; }
    setBuilding(true);
    resetRuntime();
    try {
      await buildKb({
        name: newKbName,
        files: newFiles,
        apiKeys: keys,
        config: {
          parsing_method: config.parsing_method, extract_images: config.extract_images, ocr: config.ocr,
          chunking_method: config.chunking_method, chunk_size: config.chunk_size, chunk_overlap: config.chunk_overlap,
          embedding_method: config.embedding_method, image_embedding_method: config.image_embedding_method,
          caption_provider: config.caption_provider,
        },
      }, (item) => {
        if (item.type === "done") return;
        setRuntime(p => ({
          ...p,
          [item.stage]: item.type === "error"
            ? { status: "done", error: item }
            : { status: "done", output: { out: item.output_preview, lat: item.trace.latency_ms, cost: item.trace.cost_usd, tok: item.trace.tokens, evalFree: item.eval_reference_free, trace: item.trace, items: item.trace.extra?.items } },
        }));
        setExpanded(p => ({ ...p, [item.stage]: true }));
      });
      const freshKbs = await getKbs();
      setKbs(freshKbs);
      setCreatingKb(false);
      selectKb(newKbName);
    } catch (e) {
      setUploadError(String(e.message || e));
    } finally {
      setBuilding(false);
    }
  };

  const handleDeleteKb = async (name) => {
    await deleteKb(name);
    setSelectedKb("");
    setKbs(await getKbs());
    resetRuntime();
  };

  // ---- Query run, with client-side step-mode pacing ----
  const revealItem = (item) => {
    if (item.type === "done") {
      setFinalSummary(item);
      if (item.answer) {
        setMessages(p => [...p, { role: "user", content: query }, { role: "assistant", content: item.answer }]);
      }
      return;
    }
    setRuntime(p => ({
      ...p,
      [item.stage]: item.type === "error"
        ? { status: "done", error: item }
        : { status: "done", output: { out: item.output_preview, lat: item.trace.latency_ms, cost: item.trace.cost_usd, tok: item.trace.tokens, evalFree: item.eval_reference_free, trace: item.trace, items: item.trace.extra?.items } },
    }));
    setExpanded(p => ({ ...p, [item.stage]: true }));
  };

  const onEvent = (item) => {
    if (runMode === "all") { revealItem(item); return; }
    if (readyToReveal.current) {
      revealItem(item);
      readyToReveal.current = false;
    } else {
      pacingQueue.current.push(item);
    }
  };

  const acknowledge = (stageId) => {
    setAcks(p => ({ ...p, [stageId]: true }));
    if (pacingQueue.current.length) {
      const next = pacingQueue.current.shift();
      revealItem(next);
    } else {
      readyToReveal.current = true;
    }
  };

  const run = async () => {
    if (running || !selectedKb) return;
    setRunning(true);
    setTab("trace");
    setRuntime(p => ({
      ...p,
      retrieval: undefined, reranking: undefined, generation: undefined,
      react: config.react_enabled ? undefined : { status: "skipped" },
    }));
    setAcks({});
    setFinalSummary(null);
    setDeepDive(null);
    pacingQueue.current = [];
    readyToReveal.current = true;
    setRuntime(p => ({ ...p, retrieval: { status: "running" } }));

    const history = messages.slice(-historyTurns * 2).map(({ role, content }) => ({ role, content }));

    try {
      await runQuery({
        kb_name: selectedKb, query, reference: reference || null, api_keys: keys,
        retrieval_method: config.retrieval_method, top_k: config.top_k, result_mode: config.result_mode,
        web_enabled: config.web_enabled, reranking_method: config.reranking_method, rerank_keep_top: config.rerank_keep_top,
        vision_rerank_enabled: config.vision_rerank_enabled,
        react_enabled: config.react_enabled, react_max_iterations: config.react_max_iterations,
        react_judge_method: config.react_judge_method, generation_method: config.generation_method,
        system_prompt: prompt, vision_grounded: config.vision_grounded,
        temperature: config.temperature, top_p: config.top_p, gen_top_k: config.gen_top_k, max_tokens: config.max_tokens,
        history,
      }, onEvent);
    } catch (e) {
      setFinalSummary({ type: "done", answer: null, error: String(e.message || e) });
    } finally {
      setRunning(false);
    }
  };

  const newChat = () => {
    setMessages([]);
    resetRuntime();
  };

  const rewrite = async () => {
    if (!hasKey(rewriteProvider)) { setKeysOpen(true); alert(`Add a ${rewriteProvider} API key first -- none found in the server's .env either.`); return; }
    setRewriting(true);
    try {
      const resp = await rewritePrompt({ prompt, provider: rewriteProvider, apiKey: keys[rewriteProvider] });
      setPrompt(resp.text);
    } catch (e) {
      alert(String(e.message || e));
    } finally {
      setRewriting(false);
    }
  };

  const suggest = async (stageId) => {
    const provider = ["claude", "gemini", "openai"].find(hasKey) || null;
    if (!provider) { setKeysOpen(true); alert("Add at least one API key first -- none found in the server's .env either."); return; }
    try {
      const resp = await suggestOption({
        stage: stageId, options: methodOptionsFor(stageId), context: query, provider, apiKey: keys[provider],
      });
      setSuggestions(p => ({ ...p, [stageId]: resp.text }));
    } catch (e) {
      alert(String(e.message || e));
    }
  };

  const runDeepDive = async () => {
    const judgeProvider = GEN_PROVIDER[config.generation_method];
    if (!hasKey(judgeProvider)) {
      setKeysOpen(true);
      alert(`Add a ${judgeProvider} API key first -- the deep-dive auto-matches generation's provider, and none found in the server's .env either.`);
      return;
    }
    setDeepDive({ loading: true });
    try {
      const resp = await diagnoseDeep({ query, stageReports: finalSummary.stage_reports, provider: judgeProvider, apiKey: keys[judgeProvider] });
      setDeepDive({ loading: false, text: resp.text });
    } catch (e) {
      setDeepDive({ loading: false, error: String(e.message || e) });
    }
  };

  const doneAll = STAGE_ORDER.every(id => runtime[id]?.status === "done" || runtime[id]?.status === "skipped");
  const totalLat = finalSummary?.total_latency_ms ?? 0;
  const totalCost = finalSummary?.total_cost_usd ?? 0;
  const genOutput = runtime.generation?.output;
  const REF_FREE = genOutput?.evalFree ? Object.entries(genOutput.evalFree.scores || {}).map(([k, v]) => ({ m: REF_FREE_LABELS[k] || k, v })) : [];

  const kbInfo = kbs.find(k => k.name === selectedKb) || null;

  const renderStage = (id, i) => {
    const r = runtime[id] || {};
    const methodKey = id === "react" ? "react_judge_method" : `${id}_method`;
    return (
      <StageCard key={id} id={id} idx={i}
        method={config[methodKey]} methodOptions={id === "react" ? methodOptionsFor("judge") : methodOptionsFor(id)}
        status={r.status || "idle"} error={r.error}
        onSelectMethod={v => onConfigChange(methodKey, v)}
        onSuggest={() => suggest(id)} suggestion={suggestions[id]}
        onManage={() => setManageStage(id)}
        trace={r.error ? undefined : (r.output ? `in : ${STAGE_LABELS[id]} input\ncfg: ${config[methodKey]}\nout: ${r.output.out}` : "")}
        expanded={!!expanded[id]} onToggle={() => setExpanded(p => ({ ...p, [id]: !p[id] }))}
        learnOpen={!!learnOpen[id]} onLearnToggle={() => setLearnOpen(p => ({ ...p, [id]: !p[id] }))}
        config={{ ...config, kbLocked: id === "embedding" && !!config.kbLocked }} onConfigChange={onConfigChange}
        runMode={runMode} acknowledged={!!acks[id]} onAck={() => acknowledge(id)} output={r.output} />
    );
  };

  if (!registry) {
    return <div style={{ padding: 40, fontFamily: T.body, color: T.sub }}>Loading registry…</div>;
  }

  return (
    <div style={{ minHeight: "100vh", background: T.paper, fontFamily: T.body, color: T.ink }}>
      <style>{`
        .gb-markdown p { margin: 0 0 8px; }
        .gb-markdown p:last-child { margin-bottom: 0; }
        .gb-markdown ul, .gb-markdown ol { margin: 0 0 8px; padding-left: 20px; }
        .gb-markdown li { margin-bottom: 3px; }
        .gb-markdown code { background: rgba(15,27,45,.07); border-radius: 4px; padding: 1px 5px; font-family: ${T.mono}; font-size: .92em; }
        .gb-markdown pre { background: #0F1B2D; color: #C7E5EE; border-radius: 8px; padding: 10px 12px; overflow-x: auto; margin: 0 0 8px; }
        .gb-markdown pre code { background: none; padding: 0; }
        .gb-markdown table { border-collapse: collapse; margin: 0 0 8px; font-size: .95em; }
        .gb-markdown th, .gb-markdown td { border: 1px solid ${T.line}; padding: 4px 8px; }
        .gb-markdown blockquote { margin: 0 0 8px; padding-left: 10px; border-left: 3px solid ${T.line}; color: ${T.sub}; }
        .gb-markdown h1, .gb-markdown h2, .gb-markdown h3 { font-family: ${T.disp}; margin: 10px 0 6px; }
      `}</style>
      {/* Top bar */}
      <div style={{ background: T.ink, color: "#fff", padding: "12px 20px", display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <div style={{ fontFamily: T.disp, fontWeight: 700, fontSize: 17, letterSpacing: .3 }}>
          Glass<span style={{ color: "#67D6EC" }}>Box</span> <span style={{ fontWeight: 400, opacity: .7 }}>· AI Training Lab</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <span style={{ background: "#1D3247", padding: "5px 14px", borderRadius: 99, fontSize: 12.5, fontFamily: T.disp, border: "1px solid #2E4B66" }}>RAG Studio</span>
          <span style={{ padding: "5px 14px", borderRadius: 99, fontSize: 12.5, opacity: .7, background: "#1D3247", border: "1px solid #2E4B66" }}>ReAct Loop · live</span>
          <span style={{ padding: "5px 14px", borderRadius: 99, fontSize: 12.5, opacity: .45 }}>Agent Eval · soon</span>
        </div>
        <button onClick={() => setKeysOpen(o => !o)} style={{ marginLeft: "auto", background: "none", border: "1px solid #2E4B66", color: "#9FD9E8", padding: "6px 14px", borderRadius: 8, cursor: "pointer", fontSize: 12.5 }}>
          {["claude", "gemini", "openai", "groq", "tavily"].some(hasKey) ? "🔑 Keys set" : "🔑 API keys"}
        </button>
      </div>

      {keysOpen && (
        <div style={{ background: "#fff", borderBottom: `1px solid ${T.line}`, padding: "14px 20px", display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[["claude", "Anthropic key (Claude Sonnet)"], ["gemini", "Google key (Gemini)"], ["openai", "OpenAI key (GPT-4o / text-embedding-3-*)"], ["groq", "Groq key (Llama 3.1/4 via Groq)"], ["tavily", "Tavily key (web search)"]].map(([k, label]) => (
            <label key={k} style={{ fontSize: 12, color: T.sub, display: "flex", flexDirection: "column", gap: 4, flex: 1, minWidth: 220 }}>
              {label}
              <input type="password" placeholder={envProviders[k] ? "already set via server .env" : "stored in session only"} value={keys[k]}
                onChange={e => setKeys(p => ({ ...p, [k]: e.target.value }))}
                style={{ padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontFamily: T.mono, fontSize: 12 }} />
              {envProviders[k] && !keys[k] && <span style={{ fontSize: 11, color: T.accent }}>✓ using server .env key</span>}
            </label>
          ))}
        </div>
      )}

      {/* Main tabs: build-time | query-time | inspection */}
      <div style={{ background: "#fff", borderBottom: `1px solid ${T.line}`, padding: "0 20px", display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
        {[["kb", "1 · Knowledge Base"], ["lab", "2 · Query Lab"], ["analysis", "3 · Analysis"]].map(([m, label]) => (
          <button key={m} onClick={() => setMainTab(m)}
            style={{ padding: "12px 18px", border: "none", background: "none", cursor: "pointer", fontFamily: T.disp, fontSize: 13.5, fontWeight: 600,
                     color: mainTab === m ? T.accent : T.sub, borderBottom: mainTab === m ? `2px solid ${T.accent}` : "2px solid transparent", position: "relative" }}>
            {label}
            {m === "analysis" && finalSummary && mainTab !== "analysis" && (
              <span style={{ position: "absolute", top: 10, right: 6, width: 7, height: 7, borderRadius: 99, background: T.amber }} />
            )}
          </button>
        ))}
        {kbInfo && (
          <span style={{ marginLeft: "auto", fontFamily: T.mono, fontSize: 11.5, color: T.sub, padding: "4px 0" }}>
            kb: <b style={{ color: T.ink }}>{kbInfo.name}</b> · {kbInfo.n_chunks} chunks · {kbInfo.embedding_method} 🔒
          </span>
        )}
      </div>

      <div style={{ padding: 20, maxWidth: 1280, margin: "0 auto" }}>
        {/* ---------- TAB 1: Knowledge Base (build-time) ---------- */}
        {mainTab === "kb" && (
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Build once · query many times</div>

          <KbPicker
            kbs={kbs} selectedKb={selectedKb} onSelect={selectKb} onDelete={handleDeleteKb}
            creating={creatingKb} onStartCreate={startCreateKb} onCancelCreate={cancelCreateKb}
            newKbName={newKbName} onNewKbName={setNewKbName} newFiles={newFiles} onNewFiles={handleNewFiles}
            uploadError={uploadError} onBuild={doBuildKb} building={building}
          />

          {selectedKb && !creatingKb && (
            <div style={{ margin: "-6px 0 16px", display: "flex", justifyContent: "flex-end" }}>
              <button onClick={() => setMainTab("lab")}
                style={{ padding: "7px 14px", borderRadius: 8, border: "none", background: T.ink, color: "#fff", fontFamily: T.disp, fontSize: 12.5, fontWeight: 600, cursor: "pointer" }}>
                Continue to Query Lab →
              </button>
            </div>
          )}

          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", margin: "4px 0 10px" }}>Build pipeline</div>
          {BUILD_STAGES.map((id, i) => renderStage(id, i))}
        </div>
        )}

        {/* ---------- TAB 2: Query Lab (query-time) ---------- */}
        {mainTab === "lab" && !selectedKb && (
          <div style={{ maxWidth: 560, margin: "40px auto", textAlign: "center", background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 28 }}>
            <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15 }}>No knowledge base selected</div>
            <div style={{ fontSize: 13, color: T.sub, marginTop: 6, lineHeight: 1.6 }}>Build a new KB or pick a saved one first — the query pipeline needs something to retrieve from.</div>
            <button onClick={() => setMainTab("kb")}
              style={{ marginTop: 14, padding: "8px 16px", borderRadius: 8, border: "none", background: T.accent, color: "#fff", fontFamily: T.disp, fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
              ← Go to Knowledge Base
            </button>
          </div>
        )}
        {mainTab === "lab" && selectedKb && (
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
          {/* LEFT: prompt → query → pipeline stages */}
          <div style={{ flex: "1 1 460px", minWidth: 340 }}>

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

          {/* Query block */}
          <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14, marginBottom: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
              <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15 }}>Query</div>
              <div style={{ display: "flex", gap: 6 }}>
                {[["all", "Run all"], ["step", "Step-by-step"]].map(([m, label]) => (
                  <button key={m} onClick={() => setRunMode(m)}
                    style={{ padding: "6px 12px", borderRadius: 8, border: `1px solid ${runMode === m ? T.accent : T.line}`,
                             background: runMode === m ? T.accentSoft : "#fff", color: runMode === m ? T.accent : T.sub,
                             fontFamily: T.disp, fontSize: 12, cursor: "pointer" }}>{label}</button>
                ))}
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <textarea value={query} onChange={e => setQuery(e.target.value)} rows={2}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!running && selectedKb) run(); } }}
                placeholder="Ask a question… (Enter to run, Shift+Enter for a new line)"
                style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13, background: "#FAFCFE",
                         fontFamily: T.body, resize: "vertical", boxSizing: "border-box" }} />
              <button onClick={run} disabled={running || !selectedKb}
                style={{ padding: "10px 18px", borderRadius: 8, border: "none", background: running || !selectedKb ? T.sub : T.accent, color: "#fff", fontFamily: T.disp, fontWeight: 600, fontSize: 13, cursor: running || !selectedKb ? "default" : "pointer", alignSelf: "flex-start" }}>
                {running ? "Running…" : "Run ▸"}
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 10, flexWrap: "wrap" }}>
              <Toggle label="🌐 Blend live web search (Tavily) into retrieval + ReAct" checked={config.web_enabled} onChange={() => onConfigChange("web_enabled", !config.web_enabled)} />
              <NumberField label="History turns to keep" value={historyTurns} onChange={setHistoryTurns} min={0} max={20} />
            </div>
            {runMode === "step" && <div style={{ fontSize: 11.5, color: T.sub, marginTop: 6 }}>Each stage pauses at "Acknowledge & continue" below it — advance stage by stage.</div>}
          </div>

          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", margin: "4px 0 10px" }}>Pipeline · signal chain</div>
          {QUERY_STAGES.map((id, i) => renderStage(id, i))}
        </div>

        {/* RIGHT: conversation + response */}
        <div style={{ flex: "1 1 420px", minWidth: 340, position: "sticky", top: 20 }}>
          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Conversation</div>

          {messages.length > 0 && (
            <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14, marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <button onClick={() => setHistoryOpen(o => !o)}
                  style={{ background: "none", border: "none", cursor: "pointer", fontFamily: T.disp, fontWeight: 600, fontSize: 14, color: T.ink, padding: 0 }}>
                  {historyOpen ? "▾" : "▸"} Chat history · {messages.length / 2} turn(s)
                </button>
                <button onClick={newChat}
                  style={{ marginLeft: "auto", padding: "5px 12px", borderRadius: 8, border: `1px solid ${T.line}`, background: "#fff",
                           color: T.sub, cursor: "pointer", fontSize: 12 }}>
                  ↺ New chat
                </button>
              </div>
              {historyOpen && (
                <div style={{ marginTop: 10, maxHeight: 260, overflowY: "auto", background: "#F6F8FB", borderRadius: 8, padding: 10 }}>
                  {messages.map((m, i) => (
                    <div key={i} style={{ marginBottom: 8, fontSize: 12.5 }}>
                      <b style={{ fontFamily: T.disp, color: m.role === "user" ? T.ink : T.accent }}>
                        {m.role === "user" ? "You" : "Assistant"}:
                      </b>{" "}
                      {m.role === "assistant"
                        ? <span className="gb-markdown"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{m.content}</ReactMarkdown></span>
                        : m.content}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Response block */}
          <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14 }}>
            <div style={{ fontFamily: T.disp, fontWeight: 600, fontSize: 15, marginBottom: 8 }}>Response</div>
            {running && !finalSummary && <div style={{ color: T.sub, fontSize: 13 }}>● Running — watch the stages stream on the left…</div>}
            {!running && !finalSummary && <div style={{ color: T.sub, fontSize: 13 }}>Ask a question above — the answer lands here, with any figures the model actually cited.</div>}
            {finalSummary?.answer && (
              <div style={{ background: T.greenSoft, borderRadius: 8, padding: 12, fontSize: 13, lineHeight: 1.6 }}>
                <div className="gb-markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                    {finalSummary.answer}
                  </ReactMarkdown>
                </div>
                <ImageGallery kbName={selectedKb} items={runtime.reranking?.output?.items || runtime.retrieval?.output?.items || []} answerText={finalSummary.answer} />
              </div>
            )}
            {finalSummary && !finalSummary.answer && finalSummary.error && (
              <div style={{ background: T.redSoft, borderRadius: 8, padding: 12, fontSize: 13, color: T.red }}>{finalSummary.error}</div>
            )}
            {finalSummary && (
              <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
                <Chip>Σ latency {(totalLat / 1000).toFixed(2)} s</Chip>
                <Chip color={T.amber} bg={T.amberSoft}>Σ cost ${totalCost.toFixed(4)}</Chip>
                <Chip color={T.green} bg={T.greenSoft}>{finalSummary.total_tokens ?? 0} tokens</Chip>
                <button onClick={() => setMainTab("analysis")}
                  style={{ marginLeft: "auto", padding: "5px 12px", borderRadius: 8, border: "none", background: T.ink, color: "#fff",
                           fontFamily: T.disp, fontSize: 12, cursor: "pointer" }}>
                  Full analysis →
                </button>
              </div>
            )}
          </div>
        </div>
        </div>
        )}

        {/* ---------- TAB 3: Analysis ---------- */}
        {mainTab === "analysis" && (
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <div style={{ fontFamily: T.disp, fontSize: 13, color: T.sub, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>Run & inspect</div>
          <div style={{ background: T.card, border: `1px solid ${T.line}`, borderRadius: 12, padding: 14 }}>
            <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${T.line}` }}>
              {["trace", "evaluation", "compare"].map(t => (
                <button key={t} onClick={() => setTab(t)}
                  style={{ padding: "8px 14px", border: "none", background: "none", cursor: "pointer", fontFamily: T.disp, fontSize: 13, color: tab === t ? T.accent : T.sub, borderBottom: tab === t ? `2px solid ${T.accent}` : "2px solid transparent" }}>
                  {t[0].toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>

            {tab === "trace" && (
              <div style={{ paddingTop: 12 }}>
                {!running && !doneAll && !finalSummary && <div style={{ color: T.sub, fontSize: 13, padding: "24px 0", textAlign: "center" }}>Run a query in the <b>Query Lab</b> — per-stage timing, tokens and cost land here.</div>}
                <div style={{ fontSize: 13 }}>
                  {STAGE_ORDER.map(id => {
                    const r = runtime[id] || {};
                    const st = r.status || "idle";
                    return (
                      <div key={id} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: `1px dashed ${T.line}`, opacity: st === "idle" ? .35 : 1 }}>
                        <span style={{ fontFamily: T.mono, fontSize: 12 }}>{r.error ? "✗" : st === "done" ? "✓" : st === "running" ? "●" : st === "skipped" ? "–" : "○"} {STAGE_LABELS[id]}</span>
                        <span style={{ fontFamily: T.mono, fontSize: 12, color: T.sub }}>{r.output ? `${r.output.lat} ms${r.output.tok ? ` · ${r.output.tok} tok` : ""}` : ""}</span>
                      </div>
                    );
                  })}
                  {finalSummary && (
                    <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
                      <Chip>Σ latency {(totalLat / 1000).toFixed(2)} s</Chip>
                      <Chip color={T.amber} bg={T.amberSoft}>Σ cost ${totalCost.toFixed(4)}</Chip>
                      <Chip color={T.green} bg={T.greenSoft}>{finalSummary.total_tokens ?? 0} tokens</Chip>
                    </div>
                  )}
                </div>
              </div>
            )}

            {tab === "evaluation" && (
              <div style={{ paddingTop: 12 }}>
                <div style={{ fontSize: 12, color: T.sub, marginBottom: 10, lineHeight: 1.5 }}>
                  Every stage gets its own report (see the left panel) — parsing/chunking/embedding/retrieval/react use fast, free, rule-based heuristics. This tab is the final generation-stage rollup, computed once using the real answer and built on the actual <span style={{ fontFamily: T.mono }}>ragas</span> package.
                </div>
                <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13, color: T.sub, marginBottom: 12, cursor: "pointer" }}>
                  <input type="checkbox" checked={hasDataset} onChange={e => setHasDataset(e.target.checked)} />
                  Gold answer attached (enables reference-based metrics)
                  {hasDataset && (
                    <input value={reference} onChange={e => setReference(e.target.value)} placeholder="gold answer text"
                      style={{ marginLeft: 8, flex: 1, padding: "4px 8px", borderRadius: 6, border: `1px solid ${T.line}`, fontSize: 12 }} />
                  )}
                </label>
                <div style={{ fontFamily: T.disp, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: T.sub, margin: "6px 0" }}>Reference-free</div>
                {REF_FREE.length === 0 && <div style={{ color: T.sub, fontSize: 12.5 }}>Run the pipeline to see scores here.</div>}
                {REF_FREE.map(r => (
                  <div key={r.m} style={{ padding: "8px 0" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                      <span>{r.m}</span><b style={{ fontFamily: T.mono, color: typeof r.v === "number" ? scoreColor(r.v) : T.ink }}>{typeof r.v === "number" ? r.v.toFixed(2) : String(r.v)}</b>
                    </div>
                    {typeof r.v === "number" && <Bar v={r.v} color={scoreColor(r.v)} />}
                  </div>
                ))}
                {genOutput?.evalFree?.recommendation && (
                  <div style={{ marginTop: 8, background: T.accentSoft, borderLeft: `3px solid ${T.accent}`, borderRadius: "0 8px 8px 0", padding: "10px 12px", fontSize: 12.5, color: "#155E6E" }}>
                    <b style={{ fontFamily: T.disp }}>Recommendation:</b> {genOutput.evalFree.recommendation}
                  </div>
                )}
                {finalSummary?.diagnosis && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontFamily: T.disp, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: T.sub, margin: "6px 0" }}>Diagnosis — scores below threshold</div>
                    <div style={{ background: T.redSoft, borderLeft: `3px solid ${T.red}`, borderRadius: "0 8px 8px 0", padding: "10px 12px", fontSize: 12.5, color: T.ink, lineHeight: 1.6 }}>
                      {finalSummary.diagnosis.summary}
                    </div>
                    <button onClick={runDeepDive} disabled={deepDive?.loading}
                      style={{ marginTop: 8, padding: "6px 12px", borderRadius: 8, border: "none", background: T.ink, color: "#fff", cursor: deepDive?.loading ? "default" : "pointer", fontSize: 12, fontFamily: T.disp }}>
                      {deepDive?.loading ? "Diagnosing…" : "✦ Deep-dive (extra LLM call)"}
                    </button>
                    {deepDive?.text && (
                      <div style={{ marginTop: 8, background: "#F6F8FB", borderRadius: 8, padding: 12, fontSize: 12.5, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{deepDive.text}</div>
                    )}
                    {deepDive?.error && <div style={{ marginTop: 8, color: T.red, fontSize: 12 }}>{deepDive.error}</div>}
                  </div>
                )}
                <div style={{ fontFamily: T.disp, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: T.sub, margin: "14px 0 6px" }}>
                  With reference {hasDataset ? "" : "· attach a gold answer to unlock"}
                </div>
                <div style={{ opacity: hasDataset ? 1 : .35 }}>
                  {genOutput?.evalWithReference
                    ? Object.entries(genOutput.evalWithReference.scores || {}).map(([k, v]) => (
                        <div key={k} style={{ padding: "8px 0" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}><span>{k}</span><b style={{ fontFamily: T.mono }}>{typeof v === "number" ? v.toFixed(2) : String(v)}</b></div>
                        </div>
                      ))
                    : <div style={{ color: T.sub, fontSize: 12.5 }}>Attach a gold answer and run to see reference-based metrics.</div>}
                </div>
              </div>
            )}

            {tab === "compare" && (
              <div style={{ paddingTop: 12, fontSize: 13, lineHeight: 1.7 }}>
                <div style={{ background: "#F6F8FB", borderRadius: 8, padding: 12 }}>
                  <b style={{ fontFamily: T.disp }}>A/B pipeline compare</b> — run two configs on the same query and diff traces, scores, cost and latency side-by-side. <i>(Not built yet — sequenced after core works, per the build brief.)</i>
                </div>
              </div>
            )}
          </div>
        </div>
        )}

        <div style={{ marginTop: 16, fontSize: 11.5, color: T.sub, lineHeight: 1.6, textAlign: "center" }}>
          Wired to the real backend (<span style={{ fontFamily: T.mono }}>core/</span> + <span style={{ fontFamily: T.mono }}>api/</span>) — see SETUP.md to run both servers locally.
        </div>
      </div>

      {/* Manage options modal -- local-only; adding a real option means dropping a registered
          class into core/{stage}/ with a @register(...) decorator, per the platform's design. */}
      {manageStage && (
        <div onClick={() => setManageStage(null)} style={{ position: "fixed", inset: 0, background: "rgba(15,27,45,.45)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }}>
          <div onClick={e => e.stopPropagation()} style={{ background: "#fff", borderRadius: 14, padding: 20, width: 340, maxWidth: "90vw" }}>
            <div style={{ fontFamily: T.disp, fontWeight: 700, fontSize: 16 }}>Manage {STAGE_LABELS[manageStage]} options</div>
            <div style={{ marginTop: 12 }}>
              {methodOptionsFor(manageStage).map(o => (
                <div key={o} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 0", borderBottom: `1px dashed ${T.line}`, fontSize: 13 }}>
                  {o}
                  <button onClick={() => setRegistryOverrides(p => ({ ...p, [manageStage]: methodOptionsFor(manageStage).filter(x => x !== o) }))}
                    disabled={methodOptionsFor(manageStage).length <= 1}
                    style={{ border: "none", background: "none", color: T.red, cursor: "pointer", fontSize: 12 }}>remove</button>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <input value={newOpt} onChange={e => setNewOpt(e.target.value)} placeholder="New option name…"
                style={{ flex: 1, padding: "8px 10px", borderRadius: 8, border: `1px solid ${T.line}`, fontSize: 13 }} />
              <button onClick={() => { if (newOpt.trim()) { setRegistryOverrides(p => ({ ...p, [manageStage]: [...methodOptionsFor(manageStage), newOpt.trim()] })); setNewOpt(""); } }}
                style={{ padding: "8px 14px", borderRadius: 8, border: "none", background: T.accent, color: "#fff", cursor: "pointer", fontSize: 13 }}>Add</button>
            </div>
            <div style={{ fontSize: 11.5, color: T.sub, marginTop: 10 }}>In the real build, adding an option maps to dropping a registered class into <span style={{ fontFamily: T.mono }}>core/{manageStage}/</span> with a <span style={{ fontFamily: T.mono }}>@register(...)</span> decorator.</div>
          </div>
        </div>
      )}
    </div>
  );
}
