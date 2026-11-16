// free-model-aggregator-02: one server, two client wires, and a real free
// model behind every answer.
//
// The brief at $MAKEFUN_PROMPT_FILE carries two documents: the free model
// list (each free OpenRouter model's gateway id, its published rating in
// milli-points, what it can do, and its free flag) and the six prompts (each
// one's id, its api, whether it streams, its weight, and the requirements a
// model must meet to serve it). This server:
//
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//      there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI),
//      POST /v1/chat/completions and POST /v1/messages, each with a JSON
//      body or with server-sent events;
//   4. answers each prompt by dialling a real free model through the
//      gateway on Chat Completions. OPENAI_API_KEY in the sandbox is a dummy
//      the gateway swaps for the OpenRouter key the entrant pushed; the real
//      key never enters the sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so never wait for anything else.
//
// ROUTING (quality is 60% of the score)
//   The candidates for a prompt are the brief's free models that meet its
//   requirements, ranked by rating_milli, highest first, ties broken by id
//   so the order is the same on every run. The server dials the best one
//   first. A permanent refusal (400, 401, 403, 404) blacklists its model
//   for the trial and the server moves to the next best. A transient
//   refusal (429, 5xx, a timeout, a broken body) retries the SAME model
//   once, then fails over: the model stays eligible for later prompts. A
//   prompt gets at most four attempts at six seconds each, 24 s worst
//   case, inside the env's 30 s per-answer cap; the clean path still takes
//   one call per prompt. The refusals a prompt recovered from are reported
//   on its answer in the header x-makefun-upstream-failures, the only way
//   the failover column can see them: the env sees the answers and never
//   the gateway's traffic. A streamed answer is buffered whole before it
//   is relayed, so a provider that sends a finish chunk twice, or usage
//   of zero output tokens, cannot break the stream's order or its usage:
//   the relay holds the first finish chunk to the end, drops the rest,
//   and appends a corrected usage chunk when the provider's own says the
//   answer produced no output token although it produced text.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests, and the gateway's
//   answers: no wall clock and no unseeded randomness, so a replay leaves the
//   same state export byte for byte. The model listings carry fixed
//   timestamps (see lib/wire.ts).

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  anthropicError,
  anthropicEventsText,
  anthropicModelList,
  anthropicToOpenAI,
  isRecord,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamText,
  openAIToAnthropic,
  type Json,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway call may take this long before it counts as refused. Six seconds is a whole short completion from a free model; the env waits 30 s per answer, and four tries at six seconds is 24 s worst case, inside that cap. The old 20 s timeout times four tries was 80 s: past the cap, so a slow model lost the prompt to the env's own abort instead of to a failover. */
const GATEWAY_TIMEOUT_MS = 6_000;
/** The most attempts one prompt gets: a first dial, a same-model retry on a transient refusal, and failovers. Four attempts at six seconds is 24 s worst case, inside the env's 30 s per-answer cap, and six prompts at four attempts is 24 calls, inside the trial's 60-call cap; the clean path still takes one call per prompt. */
const MAX_ATTEMPTS_PER_PROMPT = 4;
/** The most refusal rows one answer's trail header may carry. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
/** The provider's public endpoint: the catalog's base_url, never the gateway's trial address. */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: 'openai';
    models: CatalogModel[];
  }>;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys`
 * as an array: the document and the key it was found under. A brief that
 * spells its lists in the sibling's names still reads.
 */
function arrayDocument(
  docs: readonly Json[],
  keys: readonly string[],
): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider
 * that names it: a row that gives the bare OpenRouter id (`vendor/model:free`)
 * and names `openrouter` is routed as the gateway spells it
 * (`openrouter/vendor/model:free`). The brief lists free models, so a row is
 * free unless it says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id =
    named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/')
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, model: ModelRow): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (model.context_length === null || model.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it
 * serves. The base_url is the provider's public endpoint, never the gateway's
 * trial-local address: the gateway's origin differs between a live pass and
 * its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte. This lists every free model in the brief —
 * exactly the fourteen dialable ids the catalog column scores — so keep it
 * that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as
 * listing more.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** The models that refused a call for good in this trial: never dialled again in it. Only a permanent refusal lands here (a caller gate, a dead endpoint, a model the gateway will not route). A transient refusal (a rate limit, a 5xx, a timeout) does not: the model stays eligible for the next prompt, and for one same-model retry on the prompt it refused. The old code blacklisted every refusal, so one 429 on the best model demoted every later prompt to a worse one. */
const permanentlyRefusedModels = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does not
 * already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/**
 * The candidates for a request, best first: every free model that meets the
 * requirements, ranked by the brief's rating_milli, ties broken by id so
 * the order is the same on every run. Models permanently refused in this
 * trial are left out: re-dialling one only burns the trial's call budget.
 * A model that only refused transiently stays in: it may serve the next
 * prompt, and it gets one same-model retry on the prompt it refused.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, model) && !permanentlyRefusedModels.has(model.id))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** Whether an HTTP refusal is transient: worth one same-model retry, and never a reason to blacklist the model. 429 is a rate limit and 5xx is the provider's trouble; 408 is a timeout the gateway reports. Every other 4xx is permanent: the request itself is the problem, and the same model would refuse it again. */
function isTransientStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's
  // requirements; this fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One refusal row for the trail header: the status, the code the gateway sent, and the model. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a gateway refusal's body, or '' when it names none. */
function errorCodeOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = value.error.code;
    if (typeof code === 'string' && code !== '') return code.slice(0, 64);
    const type = value.error.type;
    return typeof type === 'string' ? type.slice(0, 64) : '';
  } catch {
    return '';
  }
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extraHeaders: Record<string, string> = {},
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** The trail header for an answer served after refusals: the rows, at most eight, in refusal order. */
function trailHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return {
    'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS)),
  };
}

/** One gateway call's outcome. `answered` carries the whole body text: the entry buffers the upstream answer before it answers the client, so a stream that stalls or breaks fails this attempt instead of hanging the client's connection after headers were sent. `transport` is a fetch that never answered at all (a timeout, a dropped connection). */
type DialResult =
  | { outcome: 'answered'; status: number; text: string }
  | { outcome: 'transport'; message: string };

/**
 * POST one request to the gateway's Chat Completions route, naming the model
 * this entry dialled, and read the whole answer. The fixed timeout covers
 * headers and body together: a provider that answers slowly but steadily
 * finishes a short completion inside it, and a hung one is cut off instead
 * of eating the env's 30 s per-answer cap. An answer over the cap is a
 * failed attempt, not a body this server forwards.
 */
async function dial(model: string, body: Json): Promise<DialResult> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch (err) {
    return { outcome: 'transport', message: (err as Error).message };
  }
  try {
    const text = await upstream.text();
    if (text.length > 2 * 1024 * 1024) return { outcome: 'transport', message: 'the gateway answered more than 2 MiB' };
    return { outcome: 'answered', status: upstream.status, text };
  } catch (err) {
    return { outcome: 'transport', message: (err as Error).message };
  }
}

/** One prompt's answer, served: a parsed JSON completion, a buffered SSE stream, or no model at all. The stream is buffered text, not a live body: the entry reads the whole upstream answer before it answers the client, so a broken stream fails the attempt instead of the client's connection. */
type ServedAnswer =
  | { kind: 'json'; answer: Json; model: string; failures: FailureRow[] }
  | { kind: 'stream'; text: string; model: string; failures: FailureRow[] }
  | { kind: 'refused'; failures: FailureRow[] };

/**
 * Dial the gateway for one prompt, best model first. A permanent refusal
 * (400, 401, 403, 404: the request itself is the problem) blacklists its
 * model for the trial and moves to the next candidate at once. A transient
 * refusal (429, 5xx, a timeout, a broken body) retries the SAME model once,
 * then fails over: the model stays eligible for later prompts, and the
 * prompt is only lost when the attempt budget is spent or no eligible
 * candidate is left. Every HTTP refusal joins the trail the answer may
 * carry; a timeout or a broken body has no HTTP answer to report, so it
 * leaves no row. The attempt count and the per-call timeout are both fixed,
 * so the worst case is four attempts at six seconds: inside the env's 30 s
 * per-answer cap, and no choice here reads the wall clock.
 */
async function serveWithFailover(
  body: Json,
  requires: Requirement,
  stream: boolean,
): Promise<ServedAnswer> {
  const failures: FailureRow[] = [];
  const candidates = candidatesFor(requires);
  let index = 0;
  /** Whether the model at `index` has already had its one same-model retry. */
  let retriedSame = false;
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    while (index < candidates.length && permanentlyRefusedModels.has(candidates[index].id)) index += 1;
    if (index >= candidates.length) break;
    const model = candidates[index];
    const result = await dial(model.id, { ...body, model: model.id });
    if (result.outcome === 'transport') {
      // No HTTP answer at all (a timeout, a dropped connection): transient,
      // and there is no refusal row to report.
      log(`dial ${model.id} failed without an answer: ${result.message}`);
      if (!retriedSame) {
        retriedSame = true;
        continue;
      }
      index += 1;
      retriedSame = false;
      continue;
    }
    if (result.status < 200 || result.status >= 300) {
      const code = errorCodeOf(result.text) === '' ? `http_${result.status}` : errorCodeOf(result.text);
      log(`dial ${model.id} refused HTTP ${result.status}: ${result.text.slice(0, 200)}`);
      failures.push({ status: result.status, code, model: model.id });
      if (isTransientStatus(result.status)) {
        if (!retriedSame) {
          retriedSame = true;
          continue;
        }
        index += 1;
        retriedSame = false;
        continue;
      }
      permanentlyRefusedModels.add(model.id);
      index += 1;
      retriedSame = false;
      continue;
    }
    // A 2xx answer only counts when its body is usable. A body that is not
    // is a failed attempt of this model, transient, with no trail row:
    // nothing was refused, the answer was just broken.
    if (stream) {
      if (result.text.trim() !== '') return { kind: 'stream', text: result.text, model: model.id, failures };
      log(`dial ${model.id} answered an empty stream`);
    } else {
      try {
        const answer: unknown = JSON.parse(result.text);
        if (isRecord(answer) && Array.isArray(answer.choices)) {
          return { kind: 'json', answer, model: model.id, failures };
        }
        log(`dial ${model.id} answered a JSON body with no choices`);
      } catch {
        log(`dial ${model.id} answered a body that is not JSON`);
      }
    }
    if (!retriedSame) {
      retriedSame = true;
      continue;
    }
    index += 1;
    retriedSame = false;
  }
  return { kind: 'refused', failures };
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. The trail rides along, so the failover column sees the refusals behind the loss. */
function refused(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  failures: FailureRow[],
): void {
  const last = failures[failures.length - 1];
  const detail =
    last === undefined ? '' : ` (the gateway answered HTTP ${last.status} for ${last.model ?? 'a model'})`;
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `every free model refused this prompt${detail}`, trailHeader(failures));
  } else {
    sendError(res, wire, 502, 'upstream_error', `every free model refused this prompt${detail}`, trailHeader(failures));
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const served = await serveWithFailover(ensureMaxTokens(body), requires, body.stream === true);
  if (served.kind === 'refused') {
    return refused(res, 'openai', served.failures);
  }
  const { model, failures } = served;
  if (body.stream === true) {
    if (served.kind !== 'stream') return refused(res, 'openai', failures);
    // The upstream stream was buffered whole, then translated: the answer
    // still leaves here as server-sent events, in order, ending [DONE].
    const text = openAIStreamText(served.text, model);
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': Buffer.byteLength(text),
      ...trailHeader(failures),
    });
    res.end(text);
    return;
  }
  if (served.kind !== 'json') return refused(res, 'openai', failures);
  sendJson(res, 200, openAIJsonWithModel(served.answer, model), trailHeader(failures));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const served = await serveWithFailover(ensureMaxTokens(anthropicToOpenAI(body, 'auto')), requires, body.stream === true);
  if (served.kind === 'refused') {
    return refused(res, 'anthropic', served.failures);
  }
  const { model, failures } = served;
  if (body.stream === true) {
    if (served.kind !== 'stream') return refused(res, 'anthropic', failures);
    const text = anthropicEventsText(served.text, model);
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      'content-length': Buffer.byteLength(text),
      ...trailHeader(failures),
    });
    res.end(text);
    return;
  }
  if (served.kind !== 'json') return refused(res, 'anthropic', failures);
  sendJson(res, 200, openAIToAnthropic(served.answer, model), trailHeader(failures));
}

/**
 * The gateway refuses a request with no output cap: a body that names none
 * gets one. The env always sends max_tokens; this is a guard for a client
 * that does not.
 */
function ensureMaxTokens(body: Json): Json {
  if (typeof body.max_tokens === 'number' || typeof body.max_completion_tokens === 'number') {
    return body;
  }
  return { ...body, max_tokens: 1024 };
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) {
        throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      }
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') {
        log(`registered: the trial holds ${reply.requests} prompts`);
      }
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw lastError;
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. A process whose
    // stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
