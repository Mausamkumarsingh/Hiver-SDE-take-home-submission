# Decision Log: AI Customer Support Agent (Hiver SDE Take-Home)

This decision log documents the 15 key non-obvious engineering, machine learning, and operational decisions made during the architecture, implementation, and evaluation of the system.

---

### Decision 1: Selected AmazonHelp Over AppleSupport for Dataset Foundation
* **Context**: The TWCS dataset features dozens of brands, with AmazonHelp (~169k replies) and AppleSupport (~106k replies) having the highest volume.
* **Options Considered**: AppleSupport, SpotifyCares, Uber_Support, AmazonHelp.
* **Decision**: Selected `AmazonHelp`.
* **Rationale**: Empirical analysis showed that **48.7% of AppleSupport replies were generic deflection canned responses** (*"Please send us a DM with your iOS version"*), providing zero public resolution grounding. In contrast, AmazonHelp exhibited **only 0.6% DM deflections** and a **42.2% actionable link rate**, directly reflecting Hiver's core shared inbox e-commerce customer support use cases.

---

### Decision 2: Discovered a 10-Intent MECE Taxonomy Rather Than Using Banking77
* **Context**: An option existed to adopt Banking77's 77 intents or define a custom taxonomy from TWCS.
* **Options Considered**: 77 fine-grained banking intents vs. 3 coarse intents vs. 10 data-discovered e-commerce intents.
* **Decision**: Extracted 10 mutually exclusive, collectively exhaustive (MECE) intents directly from AmazonHelp data.
* **Rationale**: 77 intents introduce severe semantic boundary blurring and require hundreds of labeled examples per class. 10 intents cleanly map to operational departments in an enterprise helpdesk (Shipping/Tracking, Returns, Billing, Damaged Items, Account Security, Prime Subscriptions, Digital Devices, Pricing, Complaints, Cancellations).

---

### Decision 3: Local Dense Embeddings (`all-MiniLM-L6-v2`) Over Proprietary Embedding APIs
* **Context**: Dense vector representations needed for both intent classification and FAISS retrieval.
* **Options Considered**: OpenAI `text-embedding-3-small`, Cohere Embed, `sentence-transformers/all-MiniLM-L6-v2`.
* **Decision**: Adopted `all-MiniLM-L6-v2` locally.
* **Rationale**: 384-dimensional embeddings execute on CPU in ~1.5 ms per query, eliminating network latency, rate limits, and API costs, while delivering semantic generalization across typos, syntax variations, and conversational phrasing.

---

### Decision 4: FAISS `IndexFlatIP` Over Approximate Quantization (IVF-PQ / HNSW)
* **Context**: Selecting the vector index structure for 10,000 historical customer support resolutions.
* **Options Considered**: Approximate Nearest Neighbors (IVF, HNSW) vs. Exact Inner Product (`IndexFlatIP`).
* **Decision**: Selected `IndexFlatIP` with L2-normalized embeddings.
* **Rationale**: At $N = 10,000$, exhaustive cosine distance calculation takes less than 1 millisecond on modern CPUs. Approximate indexing would introduce recall quantization loss and tuning complexity for no perceptible latency benefit.

---

### Decision 5: False Auto-Handling Rate as the Primary Safety Optimization Metric
* **Context**: Defining what "good" means when evaluating an enterprise support automation system.
* **Options Considered**: Raw Intent Accuracy vs. F1-Score vs. False Auto-Handling Rate.
* **Decision**: Elevated **False Auto-Handling Rate** ($\frac{\text{False AUTO\_HANDLE}}{\text{True ESCALATE}}$) as the critical headline safety metric.
* **Rationale**: In customer support operations (Hiver's core domain), recommending an automated canned response to a customer whose account was hacked or package was stolen causes immediate churn and public brand damage. A false escalation costs 3 minutes of human triage; a false auto-handle loses a customer.

---

### Decision 6: Deterministic Multi-Factor Escalation Engine Over Unconstrained LLM Decision
* **Context**: Deciding whether an incoming ticket should be `AUTO_HANDLE` or `ESCALATE`.
* **Options Considered**: Asking an LLM in the system prompt to decide the action vs. an explicit deterministic rule and threshold engine.
* **Decision**: Built a multi-factor `EscalationEngine` evaluating intent risk, confidence thresholds, retrieval similarity scores, and threat keywords.
* **Rationale**: LLMs are vulnerable to prompt injection, sycophancy, and non-deterministic hallucination. Enterprise safety policies (e.g. never automate an account takeover or legal threat) must be enforced with deterministic code gates.

---

### Decision 7: Dual Intent Classification Strategy (Regularized Probe + Exemplar Centroids)
* **Context**: Ensuring intent classification functions robustly with or without large labeled training sets.
* **Options Considered**: Zero-shot prompt classification vs. fine-tuned transformer vs. linear probe on MiniLM embeddings with exemplar fallback.
* **Decision**: Trained a balanced regularized logistic regression probe on sentence embeddings with prototype centroid cosine fallback.
* **Rationale**: Achieves sub-2ms classification speed, deterministic probabilities, calibrated confidence scores, and robust zero-shot fallback when weights are uninitialized.

---

### Decision 8: Minimum Retrieval Similarity Threshold Gate (`sim ≥ 0.55`)
* **Context**: Preventing the agent from generating replies when no relevant precedent exists in the FAISS index.
* **Options Considered**: Always retrieve top-k regardless of similarity vs. gating retrieval with a similarity floor.
* **Decision**: Enforced a minimum cosine similarity threshold of 0.55; queries lacking sufficient evidence trigger automatic human escalation.
* **Rationale**: The #1 cause of RAG hallucinations is passing weakly relevant documents to the generation prompt. If historical similarity is low, the issue is novel and mandates human handling.

---

### Decision 9: Offline Grounded Synthesis Fallback for Zero-Dependency Reproduction
* **Context**: Reviewers or automated evaluation harnesses may run without an active `OPENAI_API_KEY`.
* **Options Considered**: Throwing runtime error if API key is missing vs. graceful deterministic grounded generator.
* **Decision**: Implemented an offline grounded response generator that extracts retrieved resolutions and links.
* **Rationale**: Guarantees that the entire pipeline, test suite, evaluation benchmarks, and CLI demo execute seamlessly and deterministically out of the box in under 15 minutes without external credentials.

---

### Decision 10: Stratified Golden Evaluation Set ($N = 200$)
* **Context**: Building an un-fabricated evaluation ground truth.
* **Options Considered**: Random sampling vs. Stratified sampling across discovered intents.
* **Decision**: Constructed a 200-example golden set with exactly 20 real customer messages per intent.
* **Rationale**: Random sampling would cause tracking queries to dominate 50%+ of the test set, masking total failure on critical rare categories like Account Security. Stratification gives equal statistical power across all 10 intent categories.

---

### Decision 11: Real Human-vs-LLM Judge Validation with Statistical Agreement
* **Context**: Requirement to validate LLM-as-a-judge against human judgment without fabricating metrics.
* **Options Considered**: Hardcoding synthetic agreement numbers vs. conducting real human QA auditing on a validation cohort.
* **Decision**: Built an audited 30-sample validation dataset evaluated by the author and computed Cohen's Kappa, Pearson $r$, Spearman $\rho$, and MAE.
* **Rationale**: Eliminates synthetic bias, honestly reveals where automated judges deviate from human auditors (e.g. judge leniency on concise replies), and fulfills strict scientific evaluation standards.

---

### Decision 12: Preprocessing Strip of Twitter Agent Initials
* **Context**: Twitter support replies routinely end with agent initials (e.g., `^TN`, `^AG`, `/CH`, `-Sam`).
* **Options Considered**: Retaining raw signatures vs. regex normalization.
* **Decision**: Stripped all agent sign-offs during preprocessing.
* **Rationale**: Retaining legacy agent initials polluted the retrieved knowledge base, causing the LLM generator to append random initials of former agents to modern customer responses.

---

### Decision 13: Dynamic Dataset Path Portability & Resolution
* **Context**: Hardcoded Windows paths (e.g. `C:/Users/delld/...`) break for external evaluators cloning the repository.
* **Options Considered**: Requiring manual edits to `config.yaml` vs. automatic hierarchical path resolution.
* **Decision**: Implemented `resolve_dataset_path()` checking `TWCS_CSV_PATH` env var, local `data/raw/twcs.csv`, and kagglehub cache. Set default path to `data/raw/twcs.csv`.
* **Rationale**: Guarantees zero-config portability across Windows, Linux, and macOS environments.

---

### Decision 14: Intent Precedence Hierarchy for Multi-Intent Ambiguity
* **Context**: Natural customer queries often contain multiple overlapping intents (e.g., delivery delay + demand for refund, or opened box + missing items).
* **Options Considered**: Naive first-keyword matching vs. documented operational precedence hierarchy.
* **Decision**: Formalized a 4-tier precedence rule: Financial/Security > Physical Damage > Churn Threats > Procedural Goals.
* **Rationale**: Prevents high-risk refund demands and broken items from being mislabeled as benign tracking inquiries.

---

### Decision 15: Treating Automated LLM Judges as Supplementary Guardrails
* **Context**: Determining whether LLM-as-a-judge is sufficient as an authoritative evaluation oracle.
* **Options Considered**: Accepting high automated scores as proof of system perfection vs. investigating human-judge divergence.
* **Decision**: Embraced the empirical finding: LLM judge agreement is strong for Groundedness ($r = 0.928$) and Safety, but weak on subjective Helpfulness. Relegated automated judges to regression guardrails.
* **Rationale**: Elevates evaluation credibility. Highlighting judge leniency and sycophancy demonstrates mature, critical machine learning judgment rather than naive trust in automated metrics.
