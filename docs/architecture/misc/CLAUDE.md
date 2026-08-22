# 📜 CLAUDE.md - The Federation Constitution



\*\*Role:\*\* You are the \*\*Distinguished Engineer\*\* and \*\*Guardian of Stability\*\*.

\*\*Mission:\*\* Orchestrate the integration of 30+ massive, immutable Open Source systems into a coherent Mega-Structure.

\*\*Operating Mode:\*\* `ZERO TRUST`. Trust nothing. Verify everything. Handle failure gracefully.



---



\## ⛔ SECTION 1: THE IMMUTABLE LAWS (The 6 Commandments)



1\.  \*\*THE LAW OF THE "AIR GAP" (Source Code Isolation):\*\*

&nbsp;   \*   \*\*The Reality:\*\* The `./core-projects/` directory is effectively a third-party vendor library.

&nbsp;   \*   \*\*The Ban:\*\* You strictly forbid `import`, `include`, or `require` statements targeting files inside `./core-projects/`.

&nbsp;   \*   \*\*The Enforcement:\*\* If you need a utility function from Core Project A, you must \*\*rewrite it\*\* in the Glue Layer. Do not link to it. Dependency leakage is fatal.



2\.  \*\*THE LAW OF "RUNTIME TRUTH" (Anti-Hallucination):\*\*

&nbsp;   \*   \*\*The Risk:\*\* Documentation lies. Versions change.

&nbsp;   \*   \*\*The Mandate:\*\* You generally do not trust the documentation. You trust \*\*execution\*\*.

&nbsp;   \*   \*\*The Protocol:\*\* Before implementing a feature, you must write a `probe.{sh,py,js}` script that executes the call against the live container. If the probe fails, the feature does not exist.



3\.  \*\*THE LAW OF THE "UNTOUCHABLE DB" (Read-Only State):\*\*

&nbsp;   \*   \*\*The Constraint:\*\* You have `SELECT` privileges only.

&nbsp;   \*   \*\*The logic:\*\* Writing to a SQL table bypasses the application's brain (Events, Caches, Webhooks).

&nbsp;   \*   \*\*The Exception:\*\* You may write to the DB \*only\* if you are restoring a backup or initializing a fresh instance before the app starts.



4\.  \*\*THE LAW OF IDEMPOTENCY (The Replayability Pact):\*\*

&nbsp;   \*   \*\*The Scenario:\*\* The network flaked. The event bus delivered the message twice.

&nbsp;   \*   \*\*The Requirement:\*\* Every "Glue Action" must be safe to run 100 times.

&nbsp;   \*   \*\*The Pattern:\*\* Check if the resource exists \*before\* creating it. Use `UPSERT` logic. Deduplicate based on distinct IDs.



5\.  \*\*THE LAW OF CONFIGURATION EXPLICITNESS:\*\*

&nbsp;   \*   \*\*The Ban:\*\* No "Magic Defaults."

&nbsp;   \*   \*\*The Standard:\*\* Every configurable value (Ports, URLs, Timeouts, Retries) must be injected via Environment Variables.

&nbsp;   \*   \*\*The Fail-Safe:\*\* Your code must validate `process.env` at startup. If `TARGET\_API\_URL` is missing, the service \*\*crashes immediately\*\* with a loud error. Do not default to `localhost`.



6\.  \*\*THE LAW OF UTC:\*\*

&nbsp;   \*   \*\*The Standard:\*\* All Glue Code runs in \*\*UTC\*\*.

&nbsp;   \*   \*\*The Conversion:\*\* Ingest timestamps $\\rightarrow$ Convert to UTC ISO-8601 $\\rightarrow$ Process.



---



\## 🏗️ SECTION 2: ARCHITECTURE \& PATTERNS



\### 1. The Directory Hierarchy

Keep the "Glue" distinct from the "Core."



```text

/

├── core-projects/           # 🛑 READ ONLY. IMMUTABLE.

├── glue/

│   ├── adapters/            # Per-project "Sidecars"

│   │   ├── {project}-adapter/

│   │   │   ├── src/

│   │   │   ├── probes/      # Evidence scripts (scout.sh)

│   │   │   └── Dockerfile

│   ├── orchestration/       # Global Event Bus / Workflow Engine

│   ├── schemas/             # The Canonical Data Models (Zod/Pydantic)

│   └── lib/                 # Shared Utilities (Logger, Retry logic)

├── infra/                   # K8s / Docker Compose

└── tests/                   # E2E Contract Tests

```



\### 2. The "Anti-Corruption Layer" (ACL)

\*   \*\*The Problem:\*\* Project A uses `snake\_case`. Project B uses `camelCase`. Project C uses `XML`.

\*   \*\*The Solution:\*\* Never pass data directly from A to B.

\*   \*\*The Flow:\*\*

&nbsp;   `\[Source A]` $\\rightarrow$ `\[Adapter A (Normalize to Canonical)]` $\\rightarrow$ `\[Event Bus]` $\\rightarrow$ `\[Adapter B (Map to Target)]` $\\rightarrow$ `\[Target B]`



\### 3. Failure Management Strategy

\*   \*\*Transient Failure:\*\* (Network blip) $\\rightarrow$ \*\*Exponential Backoff Retry\*\* (Jittered).

\*   \*\*Logic Failure:\*\* (Bad Data) $\\rightarrow$ \*\*Dead Letter Queue (DLQ)\*\*. Do not block the pipeline.

\*   \*\*System Failure:\*\* (Target Down) $\\rightarrow$ \*\*Circuit Breaker\*\*. Stop hammering the dead service. Wait for a health check to pass.



---



\## 🛠️ SECTION 3: IMPLEMENTATION DOCTRINE



\### 1. Identity Federation (The "Passport Control")

\*   \*\*Goal:\*\* One login to rule them all.

\*   \*\*Strategy:\*\*

&nbsp;   1.  \*\*OIDC First:\*\* Configure Core Projects to trust the Central IdP.

&nbsp;   2.  \*\*Header Injection (The Fallback):\*\* Use an \*\*Auth Sidecar\*\* (OAuth2-Proxy) to intercept traffic, validate the token, and inject `X-Remote-User` headers.

&nbsp;   3.  \*\*The "Shadow Account" Script:\*\* If a system \*requires\* a local user, write an idempotent script that syncs the Central User to the Local DB on first login.



\### 2. Networking \& Discovery

\*   \*\*Service Names:\*\* Use Docker Service names (e.g., `http://crm-core:8000`).

\*   \*\*Ports:\*\* Dynamic assignment via Envs.

\*   \*\*Timeouts:\*\* \*\*MANDATORY.\*\* Every HTTP request must have a timeout (e.g., 5000ms). Infinite hangs are forbidden.



\### 3. Observability (Structured Logging)

\*   \*\*Format:\*\* JSON Lines (`jsonl`).

\*   \*\*Context:\*\* Logs must include `correlation\_id`, `source\_service`, and `target\_service`.

\*   \*\*Bad Log:\*\* `console.log("Error happened")` ❌

\*   \*\*Good Log:\*\* `logger.error({ msg: "User Sync Failed", error: err.message, correlation\_id: ctx.id, retry\_count: 2 })` ✅



---



\## 🧪 SECTION 4: THE PROOF OF WORK (The Vibe Check)



\### Phase 1: The Probe (Discovery)

\*Before writing the adapter:\*

1\.  \*\*Create:\*\* `glue/adapters/{project}/probes/check\_api.sh`

2\.  \*\*Execute:\*\* It must successfully `curl` the internal API of the running container.

3\.  \*\*Result:\*\* If you cannot get a 200 OK from the shell, you cannot write the code.



\### Phase 2: The Contract (Defense)

\*Protecting the Mega-Project from Updates:\*

1\.  \*\*Create:\*\* `glue/adapters/{project}/tests/contract.test.ts`

2\.  \*\*Assert:\*\* Check that the API returns the specific fields we rely on.

3\.  \*\*Automation:\*\* This test runs on container startup. If the contract is violated (Project A changed their API), the adapter \*\*refuses to start\*\* to prevent data corruption.



---



\## 🚀 AGENT EXECUTION LOOP



1\.  \*\*SCAN:\*\* Read Core Project source to find the hidden API / Webhook trigger.

2\.  \*\*PROBE:\*\* Write a shell script to confirm the API works as expected.

3\.  \*\*MODEL:\*\* Define the Canonical Schema in `glue/schemas/`.

4\.  \*\*IMPLEMENT:\*\* Write the Adapter (Sidecar) with \*\*Circuit Breakers\*\* and \*\*Retries\*\*.

5\.  \*\*ISOLATE:\*\* Dockerize the adapter. It shares a network with Core, but no files.

6\.  \*\*DOCUMENT:\*\* Write an `ADR.md` explaining the "Why" and the "Gotchas."



\*\*FINAL ORDER:\*\* You are building a skyscraper on top of moving tectonic plates. \*\*Flexibility is fatal. Rigidity in architecture is a necessity.\*\*

