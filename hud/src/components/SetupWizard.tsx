import { useState } from "react";
import { submitSetup, uploadSource, type SetupStatus } from "../api";

interface Props {
  onComplete: () => void;
  initial?: SetupStatus | null;
}

type Mode = "local" | "cloud";

/**
 * First-run setup (Phase B). Shown when /api/setup reports configured=false.
 * Local vs cloud is purely which OpenAI-compatible base_url + key + model names
 * we write — the engine's LLM client is provider-agnostic. Soul-files stay
 * optional (blank-bootstrap); only a system prompt personalizes, and even that
 * has a default if left empty.
 */
export default function SetupWizard({ onComplete, initial }: Props) {
  // On reconfigure (already configured) pre-fill from current config; on first
  // run use built-in defaults. The API key is never echoed back, so never seeded.
  const seed = initial && initial.configured ? initial : null;
  const [mode, setMode] = useState<Mode>("local");

  // Local (LM Studio / Ollama)
  const [baseUrl, setBaseUrl] = useState(seed?.llm_base_url ?? "http://localhost:1234/v1");
  const [apiKey, setApiKey] = useState("lm-studio");
  const [modelLight, setModelLight] = useState(seed?.model_light ?? "");
  const [modelHeavy, setModelHeavy] = useState(seed?.model_heavy ?? "");
  const [embedModel, setEmbedModel] = useState(seed?.embedding_model ?? "text-embedding-bge-large-en-v1.5");
  const [embedDim, setEmbedDim] = useState(seed?.embedding_dim ?? 1024);

  // Cloud (OpenAI-compatible)
  const [cloudBase, setCloudBase] = useState("https://api.openai.com/v1");
  const [cloudKey, setCloudKey] = useState("");
  const [cloudModel, setCloudModel] = useState("gpt-4o-mini");
  const [cloudEmbed, setCloudEmbed] = useState("text-embedding-3-small");
  const [cloudDim, setCloudDim] = useState(1536);

  // Identity (all optional except — softly — a name)
  const [operatorName, setOperatorName] = useState(seed?.operator_name ?? "");
  const [nodeName, setNodeName] = useState(seed?.node_name ?? "Lumos");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [idFile, setIdFile] = useState<File | null>(null);
  const [knFile, setKnFile] = useState<File | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setError(null);
    if (mode === "local" && !modelLight.trim()) {
      setError("Enter your model name exactly as it appears in LM Studio (e.g. qwen/qwen3.5-9b).");
      return;
    }
    if (mode === "cloud" && !cloudKey.trim()) {
      setError("Paste your API key to use a cloud provider.");
      return;
    }
    setBusy(true);
    try {
      const payload =
        mode === "local"
          ? {
              llm_base_url: baseUrl.trim(),
              llm_api_key: apiKey.trim() || "lm-studio",
              model_light: modelLight.trim(),
              model_heavy: (modelHeavy.trim() || modelLight.trim()),
              embedding_model: embedModel.trim(),
              embedding_dim: embedDim,
              model_swap_enabled: true,
              operator_name: operatorName.trim(),
              node_name: nodeName.trim() || "Lumos",
              system_prompt_text: systemPrompt,
            }
          : {
              llm_base_url: cloudBase.trim(),
              llm_api_key: cloudKey.trim(),
              model_light: cloudModel.trim(),
              model_heavy: cloudModel.trim(),
              embedding_model: cloudEmbed.trim(),
              embedding_dim: cloudDim,
              model_swap_enabled: false,
              operator_name: operatorName.trim(),
              node_name: nodeName.trim() || "Lumos",
              system_prompt_text: systemPrompt,
            };
      const loc =
        lat.trim() && lon.trim() && !Number.isNaN(Number(lat)) && !Number.isNaN(Number(lon))
          ? { operator_lat: Number(lat), operator_lon: Number(lon) }
          : {};
      // Upload any provided files; their saved server paths become the sources,
      // and /setup kicks off a background ingest to embed them.
      const srcs: Record<string, string> = {};
      if (idFile) srcs.identity_source = (await uploadSource("identity", idFile)).path;
      if (knFile) srcs.knowledge_source = (await uploadSource("knowledge", knFile)).path;
      await submitSetup({ ...payload, ...loc, ...srcs });
      onComplete();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const detectLocation = () => {
    setError(null);
    if (!navigator.geolocation) {
      setError("This browser can't auto-detect location — enter it manually.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLat(pos.coords.latitude.toFixed(6));
        setLon(pos.coords.longitude.toFixed(6));
      },
      () => setError("Couldn't detect location — enter it manually."),
    );
  };

  return (
    <div className="flex h-full w-full items-center justify-center overflow-y-auto p-6">
      <div className="panel hr-inset w-full max-w-xl rounded-md border border-line p-7 shadow-2xl">
        <div className="mb-1 text-sm font-medium tracking-tight text-fg">
          {nodeName || "Lumos"} OS · setup
        </div>
        <div className="mb-6 font-mono text-2xs text-muted">
          One-time setup. Your files (chat history, knowledge) are optional — memory
          grows as you talk. All you really need is a brain to think with.
        </div>

        {/* backend choice */}
        <div className="mb-2 font-mono text-2xs uppercase tracking-widest text-muted">
          the brain
        </div>
        <div className="mb-5 flex items-center gap-1 font-mono text-2xs">
          <button
            type="button"
            onClick={() => setMode("local")}
            className={
              "rounded-sm border px-3 py-1.5 transition-colors " +
              (mode === "local"
                ? "border-accent/60 bg-accent/10 text-accent"
                : "border-line text-muted hover:text-fg")
            }
          >
            local · free + private
          </button>
          <button
            type="button"
            onClick={() => setMode("cloud")}
            className={
              "rounded-sm border px-3 py-1.5 transition-colors " +
              (mode === "cloud"
                ? "border-accent/60 bg-accent/10 text-accent"
                : "border-line text-muted hover:text-fg")
            }
          >
            cloud · API key
          </button>
        </div>

        {mode === "local" ? (
          <div className="space-y-4">
            <div className="font-mono text-2xs text-muted">
              Runs on your machine via LM Studio (or Ollama). Free, fully private,
              nothing leaves your computer. You need the app running with your models
              loaded.
            </div>
            <Text label="server url" value={baseUrl} onChange={setBaseUrl} placeholder="http://localhost:1234/v1" hint="LM Studio default shown · Ollama is http://localhost:11434/v1" />
            <Text label="chat model" value={modelLight} onChange={setModelLight} placeholder="qwen/qwen3.5-9b" hint="EXACT name as loaded in LM Studio" />
            <Text label="heavy model (optional)" value={modelHeavy} onChange={setModelHeavy} placeholder="openai/gpt-oss-20b — leave blank to reuse the chat model" />
            <Text label="embedding model" value={embedModel} onChange={setEmbedModel} placeholder="text-embedding-bge-large-en-v1.5" hint="must be loaded in LM Studio too" />
            <Num label="embedding dimension" value={embedDim} onChange={setEmbedDim} hint="bge-large = 1024 · must match the model" />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="font-mono text-2xs text-muted">
              Uses a cloud provider (OpenAI shown). Easiest to start — no downloads —
              but costs a little per use and your messages go to that provider.
            </div>
            <Text label="provider url" value={cloudBase} onChange={setCloudBase} placeholder="https://api.openai.com/v1" hint="any OpenAI-compatible endpoint" />
            <Text label="api key" value={cloudKey} onChange={setCloudKey} placeholder="sk-…" type="password" hint="stored locally, in your config only" />
            <Text label="chat model" value={cloudModel} onChange={setCloudModel} placeholder="gpt-4o-mini" />
            <Text label="embedding model" value={cloudEmbed} onChange={setCloudEmbed} placeholder="text-embedding-3-small" />
            <Num label="embedding dimension" value={cloudDim} onChange={setCloudDim} hint="OpenAI text-embedding-3-small = 1536" />
          </div>
        )}

        {/* identity */}
        <div className="mb-2 mt-6 border-t border-line pt-5 font-mono text-2xs uppercase tracking-widest text-muted">
          identity
        </div>
        <div className="space-y-4">
          <Text label="your name" value={operatorName} onChange={setOperatorName} placeholder="what should the AI call you?" />
          <Text label="ai name" value={nodeName} onChange={setNodeName} placeholder="Lumos" />
          <div>
            <div className="mb-2 font-mono text-2xs uppercase tracking-widest text-muted">
              system prompt · who is your AI? (optional)
            </div>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={5}
              placeholder="Describe who your AI is — its name, voice, values, and what it should know about you. Paste one from another AI, or leave blank for a friendly default you can change later."
              className="w-full resize-y rounded-sm border border-line bg-bg px-3 py-2 font-mono text-xs leading-relaxed text-fg outline-none focus:border-accent/50"
            />
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between gap-3">
              <span className="font-mono text-2xs uppercase tracking-widest text-muted">
                location · for local sky / weather / timing (optional)
              </span>
              <button
                type="button"
                onClick={detectLocation}
                className="font-mono text-2xs uppercase tracking-widest text-accent transition-colors hover:text-fg"
              >
                detect
              </button>
            </div>
            <div className="flex gap-3">
              <input
                type="text"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
                placeholder="latitude"
                inputMode="decimal"
                spellCheck={false}
                className="w-full rounded-sm border border-line bg-bg px-3 py-2 font-mono text-xs text-fg outline-none focus:border-accent/50"
              />
              <input
                type="text"
                value={lon}
                onChange={(e) => setLon(e.target.value)}
                placeholder="longitude"
                inputMode="decimal"
                spellCheck={false}
                className="w-full rounded-sm border border-line bg-bg px-3 py-2 font-mono text-xs text-fg outline-none focus:border-accent/50"
              />
            </div>
            <div className="mt-1 font-mono text-2xs text-muted">
              leave blank to skip · enables aircraft, space-weather and planetary timing near you
            </div>
          </div>
          <div>
            <div className="mb-2 font-mono text-2xs uppercase tracking-widest text-muted">
              your files (optional) · memory + knowledge
            </div>
            <div className="space-y-2">
              <label className="flex items-center justify-between gap-3">
                <span className="font-mono text-2xs text-muted">chat history</span>
                <input
                  type="file"
                  accept=".json,.jsonl,.txt,.md"
                  onChange={(e) => setIdFile(e.target.files?.[0] ?? null)}
                  className="max-w-[14rem] font-mono text-2xs text-fg file:mr-2 file:cursor-pointer file:rounded-sm file:border file:border-line file:bg-bg file:px-2 file:py-1 file:font-mono file:text-2xs file:text-accent"
                />
              </label>
              <label className="flex items-center justify-between gap-3">
                <span className="font-mono text-2xs text-muted">knowledge corpus</span>
                <input
                  type="file"
                  accept=".jsonl,.json,.txt,.md"
                  onChange={(e) => setKnFile(e.target.files?.[0] ?? null)}
                  className="max-w-[14rem] font-mono text-2xs text-fg file:mr-2 file:cursor-pointer file:rounded-sm file:border file:border-line file:bg-bg file:px-2 file:py-1 file:font-mono file:text-2xs file:text-accent"
                />
              </label>
            </div>
            {seed && (seed.identity_file || seed.knowledge_file) ? (
              <div className="mt-1 font-mono text-2xs text-accent">
                ✓ currently loaded:{" "}
                {[seed.identity_file, seed.knowledge_file].filter(Boolean).join(" · ")}
                {" "}— leave blank to keep, or choose a file to replace.
              </div>
            ) : (
              <div className="mt-1 font-mono text-2xs text-muted">
                drop in an AI chat export or any transcript — Lumos indexes it as
                memory (any text works). leave blank to start fresh.
              </div>
            )}
          </div>
        </div>

        {error && <div className="mt-5 font-mono text-2xs text-err">{error}</div>}

        <div className="mt-7 flex items-center justify-between">
          <div className="font-mono text-2xs text-muted">
            you can change all of this later in settings
          </div>
          <button
            type="button"
            onClick={submit}
            disabled={busy}
            className="rounded-sm border border-accent/60 bg-accent/10 px-4 py-2 font-mono text-2xs uppercase tracking-widest text-accent transition-colors hover:text-fg disabled:opacity-40"
          >
            {busy ? "saving…" : "begin"}
          </button>
        </div>
      </div>
    </div>
  );
}

interface TextProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: string;
  type?: string;
}

function Text({ label, value, onChange, placeholder, hint, type = "text" }: TextProps) {
  return (
    <div>
      <div className="mb-1.5 font-mono text-2xs uppercase tracking-widest text-muted">
        {label}
      </div>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete="off"
        className="w-full rounded-sm border border-line bg-bg px-3 py-2 font-mono text-xs text-fg outline-none focus:border-accent/50"
      />
      {hint && <div className="mt-1 font-mono text-2xs text-muted">{hint}</div>}
    </div>
  );
}

interface NumProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
}

function Num({ label, value, onChange, hint }: NumProps) {
  return (
    <div>
      <div className="mb-1.5 font-mono text-2xs uppercase tracking-widest text-muted">
        {label}
      </div>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value) || 0)}
        className="w-40 rounded-sm border border-line bg-bg px-3 py-2 font-mono text-xs text-fg outline-none focus:border-accent/50"
      />
      {hint && <div className="mt-1 font-mono text-2xs text-muted">{hint}</div>}
    </div>
  );
}
