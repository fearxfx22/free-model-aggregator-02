# Free Model Aggregator — live

[![make.fun](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/fearxfx22/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/fearxfx22/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02)

One server that makes OpenRouter's fourteen free models a single endpoint.
Point any OpenAI or Anthropic client at it and every answer is a live
free-model completion, routed by quality, with failover when a provider
refuses.

Made for the [free-model-aggregator-02 bounty](https://make.fun/bounty/free-model-aggregator-02)
on [make.fun](https://make.fun/r/xp1awx6n).

## What it does

- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages`
  (Anthropic wire), in JSON and in server-sent events, plus `GET /v1/models`
  in the wire its auth header names.
- Answers every prompt with a real completion from a free OpenRouter model,
  dialled through the grader's gateway on the entrant's pushed key — never
  with an answer of its own.
- Routes by the brief's quality ratings: the best free model that meets the
  prompt's requirements serves it (JSON prompts go to the best JSON-capable
  model, the rest to the best overall). On a refusal it fails over to the
  next best that meets the prompt, and reports the refusals it recovered
  from in the `x-makefun-upstream-failures` header.
- Prints its catalog with `node main.ts --list`, byte for byte as it
  registers it: one provider, `openrouter`, exactly the fourteen free ids.

## Run it

Node 22 or later, standard library only — nothing to install.

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # serve (needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL)
```

## Files

- `main.ts` — the server: the brief, the catalog, the routes, the routing
  and failover, the register and the poll.
- `lib/wire.ts` — the client wires, the one upstream wire, and the model
  field: Anthropic ↔ Chat Completions translation and the SSE relays.
- `LICENSE` — MIT.
