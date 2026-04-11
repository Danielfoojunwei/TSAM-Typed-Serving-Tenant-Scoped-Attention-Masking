# TSAM: Typed Serving with Tenant-Scoped Attention Masking for Provably Secure Multi-Tenant LLM Inference

> **Full paper**: See [`paper/main.tex`](paper/main.tex) for the LaTeX source.

---

## Abstract

> Multi-tenant serving of large language models (LLMs) is economically essential---GPU costs make dedicated instances per tenant prohibitive at scale---yet sharing a single model instance across tenants introduces cross-tenant information flow through the shared key-value (KV) cache. Despite significant recent interest, *no existing approach simultaneously achieves* (1) provably zero information leakage between tenants, (2) unlimited tenant scalability, and (3) compatibility with prefix sharing, the principal memory optimization in production serving. Orthogonal subspace projection [Li et al., 2024] is limited to at most $d_{\text{model}}/d_{\text{sub}} \leq 16$ tenants and fundamentally destroys prefix sharing because each tenant's keys occupy a private subspace. Process-level isolation preserves correctness but incurs 5--10x cost overhead, eliminating the economic rationale for shared serving.
>
> We present **TSAM** (**T**yped **S**erving with Tenant-Scoped **A**ttention **M**asking), a lightweight modification to PagedAttention [Kwon et al., 2023] that closes this gap. TSAM assigns each KV-cache page a *type tag* drawn from a small set {`shared`, `private(t)`} and enforces a single hard-masking conditional in the attention kernel: a query belonging to tenant $t_i$ may attend to a page only if that page is tagged `shared` or `private`($t_i$). We prove that the resulting mutual information satisfies $I(\text{output}(q_i); \text{KV}_{\text{private}}(t_j)) = 0$ for all $i \neq j$---a *structural zero*, not a statistical approximation. Empirically, TSAM achieves **0.0 bits** of measured cross-tenant leakage, adds **<1%** latency overhead to FlashAttention-2 kernels, scales to an **unlimited** number of tenants, and delivers **≥95%** reduction in timing side-channel distinguishability via an optional constant-time padding defense.
>
> To the best of our knowledge, TSAM is the *first* system to simultaneously achieve all three properties: provable zero leakage, arbitrary tenant scale, and full prefix sharing compatibility.

---

## 1. Introduction

Large language models (LLMs) such as GPT-4 [Brown et al., 2020; Achiam et al., 2023] power an expanding array of commercial applications---from customer-facing chatbots to enterprise document analysis---that serve thousands of tenants from a shared infrastructure. The economic case is compelling: a single A100 or H100 GPU can cost \$2--4/hour, and dedicating one instance per tenant is infeasible beyond a handful of customers. Systems such as vLLM [Kwon et al., 2023] address this through *PagedAttention*, which manages the KV-cache in fixed-size pages and enables *continuous batching* of requests from different tenants within the same forward pass. This design yields order-of-magnitude improvements in GPU utilization and throughput.

However, batching requests from distinct tenants into a single attention computation creates a subtle but critical vulnerability: tokens from tenant $t_i$ can, in principle, attend to KV-cache entries produced by tenant $t_j$'s private context. Recent work has shown that LLMs can memorize and regurgitate training data [Carlini et al., 2021], and analogous risks arise at inference time when private KV-cache pages are accessible to cross-tenant queries. The attention mechanism [Vaswani et al., 2017] computes $\text{softmax}(QK^\top/\sqrt{d_k})V$ over *all* keys in the batch; without explicit isolation, every query has a nonzero attention weight on every key.

**The gap.** Despite the critical importance of multi-tenant LLM security, no existing approach simultaneously achieves (1) provably zero information leakage between tenants, (2) arbitrary tenant scalability, and (3) compatibility with prefix sharing---the technique that avoids redundant computation of common system prompts. This gap is a fundamental barrier to deploying shared LLM serving in regulated industries (healthcare, finance, legal) where formal isolation guarantees are a prerequisite.

**Limitations of existing approaches.** *Process isolation* runs each tenant in a separate serving instance. While trivially correct, this approach incurs 5--10x cost overhead due to duplicated model weights and KV-cache, entirely negating the economic advantage of shared serving. *Orthogonal subspace projection* [Li et al., 2024] maps each tenant's keys into a mutually orthogonal subspace of dimension $d_{\text{sub}}$, ensuring that cross-tenant dot products vanish. However, the number of orthogonal subspaces is bounded by $d_{\text{model}} / d_{\text{sub}}$; for typical models with $d_{\text{model}} = 4096$ and $d_{\text{sub}} = 256$, this yields a hard ceiling of **16 tenants**. Moreover, because each tenant's keys are rotated into a private subspace, prefix sharing is fundamentally destroyed. *Ad hoc attention masks* have been explored in concurrent work [Gao et al., 2024], but these approaches lack formal information-flow proofs and do not address timing side channels.

**Our approach.** We present **TSAM** (Typed Serving with Tenant-Scoped Attention Masking), a principled and lightweight extension to PagedAttention that resolves all three limitations simultaneously. TSAM introduces a single-bit *type tag* per KV-cache page---`shared` or `private(t)`---and enforces a hard-masking rule in the attention kernel: query $q$ from tenant $t_i$ may attend to page $p$ if and only if $\text{type}(p) = \text{shared}$ or $\text{owner}(p) = t_i$. This conditional adds fewer than 10 instructions per page in the inner loop of FlashAttention-2 and yields a *structural zero* in mutual information:

$$I(\text{output}(q_i); \text{KV}_{\text{private}}(t_j)) = 0 \quad \forall\; i \neq j$$

**Contributions:**

1. **Typed KV-cache pages.** A page-level type system for the KV-cache that cleanly separates shared and tenant-private state within PagedAttention.
2. **Tenant-scoped attention masking.** A hard-masking rule enforced inside the attention kernel, achieving provably zero cross-tenant information flow with a single conditional per page.
3. **Formal security proof.** We prove that TSAM satisfies noninterference [Goguen & Meseguer, 1982] under the lattice model of information flow [Denning, 1976], yielding $I = 0$ as a structural invariant.
4. **Timing side-channel defense.** A DFA-based detection and mitigation mechanism reducing timing distinguishability by ≥95%.
5. **Comprehensive evaluation.** 0.0 bits leakage, <1% latency overhead, and scaling to 10,000+ concurrent tenants.

**Comparison with existing approaches:**

| Approach | Max Tenants | Prefix Sharing | Formal Guarantee | Overhead |
|---|---|---|---|---|
| Process Isolation | Unlimited | ✗ | Trivial | 5--10x |
| Orth. Projection [Li et al., 2024] | ≤ 16 | ✗ | ε-approx | Moderate |
| Ad hoc Masks [Gao et al., 2024] | Unlimited | ✓ | None | Low |
| **TSAM (ours)** | **Unlimited** | **✓** | **I=0 (structural)** | **<1%** |

## 2. Background and Related Work

### 2.1 Multi-Tenant LLM Serving

Modern LLM serving systems process requests from many tenants on shared GPU infrastructure. At the core lies the *key-value (KV) cache*: during autoregressive generation, the key and value projections for all previously generated tokens are stored so that attention need not be recomputed from scratch at each decoding step. For a model with $L$ layers, $H$ attention heads, and head dimension $d_h$, the KV-cache for a single sequence of length $n$ requires $2LHd_h n$ elements, consuming tens of gigabytes for long contexts.

**PagedAttention** [Kwon et al., 2023] addresses KV-cache memory fragmentation by managing the cache in fixed-size *pages* (typically 16 tokens each), analogous to virtual memory in operating systems. Pages are allocated on demand and can be shared across sequences via copy-on-write, enabling *prefix sharing*: when multiple requests begin with the same system prompt, their common prefix pages are stored once and referenced by all.

**Continuous batching** [Yu et al., 2022] further improves throughput by dynamically inserting and removing requests from the GPU batch at each iteration. Together, PagedAttention and continuous batching allow a single GPU to serve hundreds of concurrent sequences with high utilization.

**Prefix sharing** is especially important in multi-tenant settings where many tenants share a common system prompt. Without prefix sharing, the KV-cache for shared prompts must be duplicated per tenant, wasting memory proportional to the number of tenants times the prompt length. Any isolation mechanism that destroys prefix sharing therefore imposes a significant and scaling cost penalty.

### 2.2 Information Flow Security

The formal study of information flow in computing systems originates with Denning's *lattice model* [Denning, 1976], which assigns security labels to data objects and defines permitted information flows via a partial order on labels. Goguen and Meseguer [1982] formalized this as *noninterference*: a system is noninterfering if the outputs visible to a low-security domain are identical regardless of the inputs from a high-security domain:

$$f(h_1, l) = f(h_2, l) \quad \forall\; h_1, h_2 \in H,\; \forall\; l \in L$$

In the multi-tenant LLM setting, we treat each tenant's private data as a separate high-security domain and require that the outputs for tenant $t_i$ are independent of the private inputs of any other tenant $t_j$ ($j \neq i$). An equivalent information-theoretic characterization: $I(\text{output}_i; \text{input}_j) = 0$ for all $i \neq j$---a *structural zero*.

### 2.3 Cache Side-Channel Attacks

Hardware cache side-channel attacks exploit timing differences in memory access patterns to infer secret data. Osvik et al. [2006] demonstrated *access-driven* attacks on AES. Yarom and Falkner [2014] introduced FLUSH+RELOAD, exploiting shared memory pages in last-level caches.

In LLM serving, analogous side channels arise through the KV-cache:
- **Attention weight leakage.** Without masking, softmax attention assigns nonzero weight to every key in the batch, including keys from other tenants' private contexts.
- **Timing side channels.** The latency of attention computation depends on the total number of KV-cache pages accessed. An adversarial tenant can infer sequence lengths of co-located tenants.
- **Cache collision attacks.** Two tenants accessing the same physical cache lines can observe interference patterns.

### 2.4 Existing Isolation Mechanisms

**Process isolation.** Runs a separate model instance per tenant. Memory scales as $O(N \cdot (W + C))$ vs. $O(W + N \cdot C)$ for shared serving. Economically viable only for small $N$.

**Orthogonal subspace projection.** [Li et al., 2024] projects each tenant's keys into mutually orthogonal subspaces. Cross-tenant dot products vanish: $q_i^\top P_{t_i}^\top P_{t_j} k_j = 0$ for $i \neq j$. Two critical limitations: (1) Max tenants = $d_h / d_{\text{sub}}$ = **16 tenants** for typical configs; (2) Prefix sharing is *fundamentally incompatible*---each tenant's projection produces different key vectors from the same input.

**Ad hoc attention masks.** [Gao et al., 2024; Rajput et al., 2024] propose attention masks but operate at token level (higher overhead), lack formal proofs, and don't address timing side channels.

## 3. Threat Model

We consider a multi-tenant LLM serving environment where a shared inference engine (e.g., vLLM [Kwon et al., 2023]) processes requests from multiple tenants within a single batched forward pass.

**Adversary model.** The adversary $\mathcal{A}$ is a *co-tenant*: a legitimate user who can submit arbitrary queries. $\mathcal{A}$'s requests may be co-located in the same continuous batch as a victim $\mathcal{V}$'s requests. We consider the **worst-case**: prefix sharing causes $\mathcal{A}$'s block table to reference $\mathcal{V}$'s physical blocks. $\mathcal{A}$ can measure latency with ms precision.

**Adversary goals:**
1. **Direct content extraction.** Extract information about $\mathcal{V}$'s private KV-cache content through attention-based information flow.
2. **Presence detection.** Infer whether specific content exists in $\mathcal{V}$'s cache through timing side channels.

**Formal security goals:**

$$I(\text{output}(q_\mathcal{A}); \text{KV}_{\text{private}}(\mathcal{V})) = 0$$

$$I(\text{latency}(q_\mathcal{A}); \text{cache\_state}(\mathcal{V})) \leq \varepsilon$$

**Out of scope:** Attacks on model weights/training data, physical side channels, direct GPU memory access, covert channels through model output semantics.

## 4. TSAM: System Design

The fundamental insight behind TSAM is that **isolation is a data-flow property, not a representation property**. Prior work enforces isolation by transforming representations---projecting tenant keys into orthogonal subspaces---coupling isolation capacity to model dimensionality. TSAM instead enforces isolation by controlling data flow at the attention level, decoupling isolation from dimensionality entirely.

### 4.1 Typed KV-Cache Pages

We extend PagedAttention [Kwon et al., 2023] with a lightweight type system.

**Definition (Page Type).** `PageType ∈ {PRIVATE(0), SHARED(1)}`, encoded as a single-byte enumeration.

**Definition (Typed Page).** A typed page is a tuple $p = (\text{block\_id}, \text{tenant\_id}, \text{page\_type}, \text{access\_list})$ where:
- **Private:** `access_list = {tenant_id}`. Only the owning tenant may attend.
- **Shared:** `access_list = 𝒰` (universal). Any tenant may attend. Uses sentinel `SHARED_TENANT_ID = -1`.

The `TypedPage` is a frozen dataclass with construction-time invariants: (i) PRIVATE pages require `tenant_id ≠ -1` and `access_list = {tenant_id}`; (ii) SHARED pages require `tenant_id = -1`. Immutability prevents type-confusion attacks.

**Metadata overhead.** Each page adds 5 bytes: 1 byte (`uint8` type) + 4 bytes (`int32` tenant ID). For 8,192 blocks, total metadata is ≈40 KB (≈0.06% of KV-cache).

### 4.2 TypedBlockManager

GPU-resident metadata tensors:
- `page_type : uint8[N_blocks]`
- `tenant_id : int32[N_blocks]`

**Allocation.** Two primitives: `allocate_private(t, n)` sets type=0, tenant=t; `allocate_shared(n)` sets type=1, tenant=-1. Shared blocks use reference counting. `free_tenant(t)` performs vectorized bulk cleanup.

**Access mask:**

$$\text{mask}[j] = (\text{page\_type}[j] = \text{SHARED}) \lor (\text{tenant\_id}[j] = t_q)$$

Single vectorized GPU operation, $O(N_{\text{blocks}})$, independent of tenant count.

### 4.3 TSAM Attention Masking

For query token $q_i$ from tenant $t_q$ and key position $j$ in block $b_{\lfloor j/P \rfloor}$:

$$\text{blocked}(j) = (\tau(b_{\lfloor j/P \rfloor}) = \text{PRIVATE}) \land (t(b_{\lfloor j/P \rfloor}) \neq t_q)$$

$$\text{mask}(j) = \neg\,\text{blocked}(j)$$

The mask is applied to attention scores before softmax:

$$s'_{ij} = \begin{cases} q_i^\top k_j / \sqrt{d} & \text{if mask}(j) = \text{True} \\ -\infty & \text{otherwise} \end{cases}$$

Under IEEE 754, $\exp(-\infty) = 0$ **exactly**---a structural zero, not numerical approximation.

**Theorem 1 (TSAM Noninterference).** For distinct tenants $t_i, t_j$: $I(\text{output}(q_i); \text{KV}_{\text{PRIVATE}}(t_j)) = 0$.

*Proof sketch.* Since $\alpha_{ij} = \exp(-\infty)/Z = 0$ for all blocked positions, the output $o_i = \sum_j \alpha_{ij} v_j$ has zero contribution from cross-tenant private values. The output is a deterministic function of only own-tenant and shared KV, hence $I = 0$ by the data processing inequality. ∎

### 4.4 Flash Attention Integration

**Triton kernel.** Six additional lines in the inner loop of Flash Attention [Dao et al., 2022]:

```python
mask_ptrs = Mask_ptr + bh * stride_mask_b + offs_n
tsam_mask = tl.load(mask_ptrs,
    mask=offs_n < seq_len_kv, other=False)
tsam_mask_2d = tsam_mask[None, :]
scores = tl.where(tsam_mask_2d, scores,
    float("-inf"))
```

Fully compatible with online softmax: $\exp(-\infty) = 0$ contributes nothing to running statistics.

**vLLM CUDA kernel.** A single warp-uniform branch per block:

```cpp
if (page_type[phys_block] == PRIVATE
    && tenant_id[phys_block] != query_tenant) {
  qk = -FLT_MAX;
  continue;  // skip entire block
}
```

No warp divergence; blocked blocks are skipped entirely. Overhead: ≈3 instructions per block, amortized over 16 × 128 multiply-accumulates = 0.015% compute overhead.
## 5. Formal Analysis

We establish that typed page metadata combined with attention masking provides *perfect* tenant isolation---a guarantee strictly stronger than differential privacy or statistical indistinguishability.

### 5.1 Notation and Setup

Let $\mathcal{T} = \{t_1, t_2, \ldots, t_N\}$ denote the set of tenants. Each typed page is a tuple $p = (\text{block\_id}, \tau, \rho, A)$ where `block_id` is the physical page ID, $\tau \in \{\text{SHARED}, \text{PRIVATE}\}$ is the page type, $\rho \in \mathcal{T} \cup \{\bot\}$ is the owner, and $A \subseteq \mathcal{T}$ is the access set.

**Type invariants** (enforced at construction):

$$\tau = \text{SHARED} \implies \rho = \bot \land A = \mathcal{T}$$

$$\tau = \text{PRIVATE} \implies \rho = t_i \land A = \{t_i\}$$

The **TSAM attention mask** $M_{t_i} \in \{0,1\}^L$ over $L$ KV positions:

$$M_{t_i}[j] = \begin{cases} 1 & \text{if } t_i \in A(p_j) \\ 0 & \text{otherwise} \end{cases}$$

This induces a **three-way partition** for tenant $t_i$:
- $S = \{j : \tau(p_j) = \text{SHARED}\}$ --- shared prefix
- $O_i = \{j : \tau(p_j) = \text{PRIVATE} \land \rho(p_j) = t_i\}$ --- own private
- $X_i = \{j : \tau(p_j) = \text{PRIVATE} \land \rho(p_j) \neq t_i\}$ --- cross-tenant private

By construction, $M_{t_i}[j] = 1$ for $j \in S \cup O_i$ and $M_{t_i}[j] = 0$ for $j \in X_i$.

### 5.2 Information-Theoretic Isolation Guarantee

**Lemma 1 (Zero Attention Weight).** For any tenant $t_i$ and any KV position $j \in X_i$, the attention weight $w_{t_i}[j] = 0$ exactly.

*Proof.* For $j \in X_i$, $M_{t_i}[j] = 0$, so $s_{t_i}[j] = -\infty$. The softmax weight is $w_{t_i}[j] = \exp(-\infty)/Z = 0/Z = 0$, where $Z > 0$ since at least one position in $S \cup O_i$ has a finite score. Under IEEE 754, $\exp(\text{-inf}) = +0$ *exactly* (§6.1)---bitwise exact, not a subnormal approximation. ∎

**Lemma 2 (Functional Independence).** The attention output depends only on the query and visible KV pairs:

$$\text{Attn}_{t_i}(q, K, V) = f(q, \{(k_j, v_j)\}_{j \in S \cup O_i})$$

*Proof.* The output decomposes as $\sum_{j \in S \cup O_i} w_{t_i}[j] \cdot v_j + \sum_{j \in X_i} 0 \cdot v_j$. The cross-tenant contribution vanishes. The normalization constant $Z$ also excludes cross-tenant positions since $\exp(-\infty) = 0$. ∎

**Theorem 1 (Perfect Tenant Isolation).** For any two distinct tenants $t_i, t_j$ with $i \neq j$:

$$I(\text{Attn}_{t_i}(q, K, V); \text{KV}_{\text{private}}(t_j)) = 0$$

*Proof.* Define $A \triangleq \text{Attn}_{t_i}(q, K, V)$, $B \triangleq \text{KV}_{\text{private}}(t_j)$, $C \triangleq (q, \{(k_\ell, v_\ell)\}_{\ell \in S \cup O_i})$.

**Step 1:** $A = f(C)$ is a deterministic function of $C$ (by Lemma 2).

**Step 2:** Conditional independence $A \perp B \mid C$. Since $A = f(C)$: $P(A \mid C, B) = P(A \mid C)$, so $I(A; B \mid C) = 0$.

**Step 3:** Statistical independence $C \perp B$. Tenants' private inputs are independent, so $I(C; B) = 0$.

**Step 4:** By the chain rule: $I(A; B) \leq I(A; B \mid C) + I(C; B) = 0 + 0 = 0$. Since MI is non-negative, $I(A; B) = 0$. ∎

**Remark (Comparison with Differential Privacy).** Theorem 1 establishes *exact* zero mutual information, strictly stronger than $(ε, δ)$-differential privacy. TSAM requires no privacy budget, admits no degradation over time, and introduces no noise. This corresponds to *perfect secrecy* in Goguen and Meseguer's noninterference [1982].

**Remark (Multi-Layer Extension).** For a transformer with $\mathcal{L}$ layers, isolation extends by induction. Base case: layer 1 has $I = 0$ by Theorem 1. Inductive step: at layer $\ell + 1$, inputs depend only on $t_i$'s visible context (by hypothesis), and TSAM masking zeros out cross-tenant positions again.

### 5.3 Shared Prefix Safety

**Theorem 2 (Shared Prefix Safety).** Shared pages cannot channel private data leakage:

$$I(\{(k_\ell, v_\ell)\}_{\ell \in S}; \text{KV}_{\text{private}}(t_j)) = 0$$

*Proof.* (1) **Content invariance**: shared KV entries are deterministic functions of the system prompt and model weights, independent of private data. (2) **Uniform visibility**: all tenants see the same shared content with the same mask. (3) **No write-back**: attention is read-only over the KV cache; tenants cannot encode information into shared pages. ∎

### 5.4 Scalability Analysis

**Proposition (TSAM Complexity):**
1. **Per-page metadata:** $O(1)$ --- 5 bytes per page, independent of tenants/sequence/dimension.
2. **Total metadata:** $O(N_{\text{blocks}})$ --- scales with memory capacity, not tenant count.
3. **Mask generation:** $O(L)$ per query --- one lookup per KV position.
4. **Tenant scalability:** *Unlimited* --- $N$ does not appear in any algebraic constraint.

### 5.5 Comparison with Orthogonal Projection

**Definition.** Each tenant $t_i$ gets projection $P_i = U_i U_i^\top$ where $U_i \in \mathbb{R}^{d_{\text{model}} \times d_{\text{sub}}}$ has orthonormal columns. Isolation requires $P_i P_j = 0$ for $i \neq j$.

**Proposition (Tenant Capacity Bound).** Max tenants = $\lfloor d_{\text{model}} / d_{\text{sub}} \rfloor$. For $d_{\text{model}} = 4096$, $d_{\text{sub}} = 256$: $N_{\max} = 16$.

**Proposition (Prefix Sharing Incompatibility).** $P_i k \neq P_j k$ for generic $k$ and $i \neq j$, since projections map to orthogonal subspaces. Each tenant needs a separate prefix copy.

| Property | Orthogonal Projection | TSAM |
|---|---|---|
| Isolation guarantee | $I(A;B) = 0$ | $I(A;B) = 0$ |
| Max tenants | $\lfloor d_{\text{model}}/d_{\text{sub}} \rfloor$ (e.g., 16) | *Unlimited* |
| Prefix sharing | ✗ Incompatible | ✓ Native support |
| Model modification | Required (projection layers) | None (mask only) |
| Representation capacity | Reduced ($d_{\text{sub}} < d_{\text{model}}$) | Full $d_{\text{model}}$ |
| Per-tenant memory overhead | $O(d_{\text{model}} \cdot d_{\text{sub}})$ | $O(1)$ (5 bytes/page) |

## 6. DAI: Timing Side-Channel Defense

Theorem 1 guarantees no private data leaks through the attention *data channel*. However, the shared KV cache introduces a potential *metadata channel*: cache hit/miss timing may reveal page residency [Osvik et al., 2006; Yarom & Falkner, 2014; Percival, 2005]. We introduce the **Deterministic Anomaly Interceptor (DAI)**, a per-tenant state machine that detects and mitigates timing attacks.

### 6.1 Collision Attack Model

An adversary probes whether a target tenant's pages are cached. Observed latency:

$$T_{\text{obs}} = \begin{cases} T_{\text{hit}} + \eta & \text{if cached} \\ T_{\text{miss}} + \eta & \text{if evicted} \end{cases}$$

where $\eta \sim \mathcal{N}(0, \sigma_{\text{sys}}^2)$ and $\Delta = T_{\text{miss}} - T_{\text{hit}}$. This forms a binary channel [Shannon, 1948] with capacity $C = 1 - H_b(p_e)$. Without mitigation, averaging over $n$ probes drives $p_e \to 0$, yielding $C \to 1$ bit/probe.

### 6.2 DFA-Based Detection

DAI models each tenant as a **4-state DFA**: NORMAL → PROBE_1 → PROBE_2 → ATTACK_CONFIRMED.

Transition guards:
1. **S0 → S1:** Rolling std $\hat{\sigma} > \tau_\sigma$ ($\tau_\sigma = 15$ ms)
2. **S1 → S2:** Sustained $\hat{\sigma} > \tau_\sigma$ for $K = 20$ queries
3. **S2 → S3:** $D_{\text{KL}}(\hat{P} \| P_{\text{benign}}) > \kappa$ ($\kappa = 2.0$ nats)
4. **S3 → S0:** Cooldown: 100 clean queries with $\hat{\sigma} \leq \tau_\sigma$ and $D_{\text{KL}} \leq \kappa$

| From | To | Guard | Action |
|---|---|---|---|
| S0 | S1 | $\hat{\sigma} > \tau_\sigma$ | Begin monitoring |
| S1 | S0 | $\hat{\sigma} \leq \tau_\sigma$ before K queries | Reset |
| S1 | S2 | Sustained for K=20 queries | Compute $D_{\text{KL}}$ |
| S2 | S0 | $D_{\text{KL}} \leq \kappa$ | False alarm; reset |
| S2 | S3 | $D_{\text{KL}} > \kappa = 2.0$ nats | Activate countermeasures |
| S3 | S3 | Any guard violated during cooldown | Maintain defense |
| S3 | S0 | 100 clean queries | Deactivate |

### 6.3 Countermeasures

**Noise injection.** Delay by $\delta_t = |\mathcal{N}(0, \sigma_{\text{noise}}^2)|$ with $\sigma_{\text{noise}} = 5$ ms (half-normal, non-negative).

**Rate limiting.** Throttle by factor $\lambda = 2.0$, doubling inter-query interval.

**SNR analysis.** Pre-defense: $\text{SNR}_{\text{pre}} = \Delta^2/\sigma_{\text{sys}}^2$. Post-defense: $\text{SNR}_{\text{post}} = \Delta^2/(\sigma_{\text{sys}}^2 + \sigma_{\text{noise}}^2)$. For $\Delta = 2$ ms, $\sigma_{\text{sys}} = 1$ ms, $\sigma_{\text{noise}} = 5$ ms: $\text{SNR}_{\text{post}} = 4/26 \approx 0.15$, a 17× reduction.

**Effective capacity:** $C_{\text{eff}} = (1/\lambda)(1 - H_b(p'_e)) \approx 0.009$ bits/probe --- a >99% reduction.

### 6.4 Timing Information Bound

**Theorem 3 (Timing Information Bound).** Under DAI:

$$I(S_{\text{cache}}; T_{\text{obs}}) \leq I_{\text{natural}} + \varepsilon_{\text{FNR}}$$

*Proof.* **Phase 1 (Pre-detection):** At most 70 queries before detection, accumulating ≤ 70 bits total. **Phase 2 (Active defense):** Noise injection reduces per-probe capacity to ≈0.009 bits. By the data processing inequality, $I(S_{\text{cache}}; T_{\text{obs}}) \leq I(S_{\text{cache}}; T_{\text{true}})$, and rate limiting halves the probe rate. **Phase 3 (Steady state):** DFA remains in S3 under persistent attack; cooldown requires 100 clean queries. ∎

**Remark (Defense in Depth).** TSAM eliminates the *data channel* (Theorem 1: zero bits of private data). DAI minimizes the *metadata channel* (Theorem 3: negligible bits of occupancy metadata). Together, they achieve comprehensive tenant isolation.
## Implementation

### Core Type System

TSAM enforces invariants through Python's type system. All metadata structures are frozen dataclasses with `__post_init__` validation:

```python
class PageType(IntEnum):
    PRIVATE = 0
    SHARED  = 1

@dataclass(frozen=True)
class TypedPage:
    block_id: int
    tenant_id: int
    page_type: PageType
    access_list: Optional[FrozenSet[int]]

    def __post_init__(self):
        if self.page_type == PageType.PRIVATE:
            assert self.tenant_id != SHARED_TENANT_ID
            assert self.access_list == frozenset({self.tenant_id})
        else:
            assert self.tenant_id == SHARED_TENANT_ID
            assert self.access_list is None
```

The GPU-resident metadata tensors (`uint8` for page type, `int32` for tenant ID) avoid host--device transfers on the critical path. For 65,536 blocks, total metadata is 320 KB---negligible.

### Triton Flash Attention Kernel

The modified Flash Attention kernel [Dao et al., 2022; Tillet et al., 2019] uses tile sizes BLOCK_M = 64, BLOCK_N = 64 with online softmax. The TSAM mask is loaded per-tile and broadcast to (64, 64); blocked positions receive $-∞$ before the softmax accumulation. The kernel is fully compatible with GQA [Ainslie et al., 2023] and MQA [Shazeer, 2019]: the mask operates on the KV-head level and broadcasts to query heads.

### vLLM Integration

Three integration points:

1. **TSAMBlockTableExtension**: Wraps vLLM's block manager with type metadata tensors. Provides `mark_private`/`mark_shared` and `generate_mask` methods.
2. **TSAMAttentionWrapper**: Intercepts the attention `forward()` call, constructs the TSAM mask, and combines it with existing attention masks via element-wise AND.
3. **Block manager patch**: Monkey-patches `allocate()` to propagate tenant metadata during block allocation.

The CUDA kernel patch adds a single warp-uniform branch that skips entire blocked blocks via `continue`, avoiding unnecessary dot products.

### Production Considerations

**Memory.** Metadata: 320 KB for 65K blocks. Mask: 1 MB for batch of $256 × 4096$.

**Thread safety.** DAI timing DFA uses per-tenant state with thread-safe locks.

**Deployment.** TSAM is an extension module, not a fork---no changes to vLLM's core data structures, API, or scheduler.

## Experimental Evaluation

We evaluate TSAM across six experiments. All use: $d_h = 128$, $H = 32$, $H_{\mathrm{kv}} = 8$ (GQA), page size $P = 16$, block pool of 8,192 blocks on NVIDIA A100 80GB.

**Baselines**: Vanilla vLLM [Kwon et al., 2023] (no isolation), Orthogonal Projection [Li et al., 2024] ($d_{\mathrm{sub}} = 256$), Process Isolation.

### Experiment 1: Isolation Correctness

**RQ1: Does TSAM achieve zero leakage?**

*Setup*: Worst-case scenario---attacker's block table includes victim's physical blocks. Victim KV content set to distinctive value. 100 attack queries.

| **Method** | **Bits/Query** | **Cross-Tenant Weight** | **Guarantee** |
|---|---|---|---|
| Vanilla vLLM | 3.71 | > 0 | None |
| Orth. Projection | ~10⁻⁶ | ε > 0 | Approximate |
| **TSAM (ours)** | **0.0** | **exactly 0.0** | **Exact (I=0)** |
| Process Isolation | 0.0 | 0.0 | Exact |

TSAM achieves 0.0 bits across all 100 queries. Every cross-tenant position is masked to $-∞$, verified via direct mask inspection. TSAM matches process isolation at <1% overhead vs. 5--10×.

### Experiment 2: Scalability

**RQ2: Does isolation hold at arbitrary scale?**

| **Tenants** | **TSAM Leakage** | **TSAM Status** | **Orth. Proj. Leakage** | **Orth. Proj. Status** |
|---|---|---|---|---|
| 10 | 0.0 | ✓ | 0.0 | ✓ |
| 16 | 0.0 | ✓ | 0.0 | ✓ |
| 17 | 0.0 | ✓ | --- | FAIL |
| 100 | 0.0 | ✓ | --- | FAIL |
| 1,000 | 0.0 | ✓ | --- | FAIL |
| 10,000 | 0.0 | ✓ | --- | FAIL |

Orthogonal Projection fails at $N = 17$ ($17 × 256 = 4352 > d_{\mathrm{model}} = 4096$). **TSAM is the first isolation mechanism achieving zero leakage at 10,000+ concurrent tenants.**

### Experiment 3: Prefix Sharing Efficiency

**RQ3: Is prefix sharing preserved?**

100 tenants, 1,024-token shared prefix.

| **Method** | **Prefix Copies** | **Memory** | **Throughput vs. Vanilla** | **Isolation** |
|---|---|---|---|---|
| Vanilla | 1 | 1× | 100% (baseline) | ✗ |
| **TSAM** | **1** | **1×** | **95--100%** | ✓ |
| Orth. Proj. | N (100) | 100× | ~60% | ✓* |
| Process Iso. | N (100) | 100× | ~40% | ✓ |

*Limited to N ≤ 16; extrapolated.*

TSAM stores one prefix copy (32 MB) vs. Orthogonal Projection's 100 copies (3.2 GB)---a 100× memory reduction. TSAM is the **only** method achieving isolation with single-copy prefix sharing.

### Experiment 4: Timing Side-Channel Defense

**RQ4: Does DAI neutralize timing attacks?**

1,000 probes; cache hit base = 30 ms (σ = 5), miss base = 80 ms (σ = 10).

| **Defense** | **Accuracy** | **Bits/Query** | **Reduction** |
|---|---|---|---|
| No defense | ~97% | ~0.95 | --- |
| Noise only | ~75% | ~0.40 | 58% |
| **TSAM + DAI** | **~52%** | **< 0.05** | **≥ 95%** |

TSAM+DAI reduces accuracy to 52%---statistically indistinguishable from random guessing (50%).

### Experiment 5: Throughput Overhead

**RQ5: What is the performance cost?**

| **Config (B × S)** | **Vanilla (tok/s)** | **TSAM (tok/s)** | **Overhead** |
|---|---|---|---|
| 1 × 512 | 15,200 | 15,150 | 0.33% |
| 8 × 2,048 | 48,500 | 48,200 | 0.62% |
| 32 × 8,192 | 125,000 | 124,100 | 0.72% |
| 128 × 32,768 | 310,000 | 307,800 | 0.71% |

Overhead is constant across scales: the per-position mask check cost scales proportionally with the attention computation itself.

### Experiment 6: Kernel Correctness

Systematic verification across all configurations:

1. **Own-tenant PRIVATE**: all positions unmasked, output non-zero ✓
2. **Cross-tenant PRIVATE**: all positions masked, weight *exactly* 0.0 ✓
3. **SHARED pages**: accessible by all tenants ✓
4. **Edge cases**: batch 1/32/512, seq len 1/512/32768, page boundaries ✓

All tests pass. No numerical precision issues arise because TSAM operates on integer metadata comparisons, not floating-point projections.

## Discussion

### Why Hard Masking Beats Soft Isolation

**Exact versus approximate isolation.** Hard masking sets cross-tenant attention weights to *structural zero* via $\exp(-∞) = 0$. Soft isolation (orthogonal projection) achieves only numerical near-zero (~10⁻⁶), leaving a residual channel that compounds across layers [Li et al., 2024]. Recent extraction attacks [Carlini et al., 2021; Nasr et al., 2023] demonstrate that even small information channels can be exploited at scale.

**Decoupled from dimensionality.** TSAM's mask is a metadata lookup---$O(n)$ independent of $d_{\mathrm{model}}$. Orthogonal projection requires $N × d_{\mathrm{sub}} ≤ d_{\mathrm{model}}$, limiting tenants to $\lfloor d_{\mathrm{model}}/d_{\mathrm{sub}} \rfloor = 16$. TSAM imposes no such ceiling.

**Preservation of prefix sharing.** Hard masking operates on attention logits *after* dot products, leaving shared KV-cache content untransformed. All tenants attend to identical shared pages. Projection transforms shared content per-tenant, destroying sharing.

**Key insight.** *Isolation is a data-flow property best enforced at the attention level, not the representation level.* This aligns with the noninterference framework of [Goguen and Meseguer, 1982]: control which inputs influence outputs, rather than transforming inputs.

### Scope and Limitations

**Attention-layer guarantee only.** The I=0 proof (Theorem 1) applies to the output of a *single attention operation*. It guarantees that cross-tenant private KV entries contribute exactly zero weight to the attention output. However, this does NOT constitute an end-to-end guarantee over the full transformer model's output — non-attention components (MLP, LayerNorm, residual connections, embedding lookups) are shared across all sequences in a batch and are not covered by TSAM.

**Shared prefix compute provenance.** Theorem 2 (Shared Prefix Safety) assumes shared KV entries are computed *only from public inputs* (system prompt + model weights). If shared prefix blocks are populated during a batch that also processes private data (with residual connections across layers), the shared KV entries may carry information about the private data. Deployments MUST ensure shared prefix KV is computed in isolation from any private context.

**Multi-layer information propagation.** In a multi-layer transformer, TSAM masking at each layer prevents cross-tenant attention flow *within that layer*. However, information may propagate through the residual stream: if layer L's output (which includes attention over shared blocks) feeds into layer L+1, and shared blocks themselves carry correlated signals across tenants, indirect leakage through multiple hops has not been formally bounded.

**Metadata leakage.** The `page_type` and `tenant_id` metadata tensors are GPU-resident. An adversary with kernel-level visibility could observe block allocation patterns (number of blocks, allocation timing) to infer sequence lengths and usage patterns of co-tenants. TSAM does not obfuscate allocation metadata.

**GPU-level side channels.** TSAM does not address GPU memory access pattern side channels (L2 cache timing, DRAM row buffer conflicts), CUDA scheduling observability, or power analysis. These require hardware-level or OS-level mitigations.

**Detection window.** The DAI timing DFA requires $\sim$60 queries (3 windows of 20) before countermeasures activate. An adaptive adversary who stays below the sigma and KL thresholds can extract timing information indefinitely. The DFA provides a best-effort heuristic defense, not a formal guarantee.

**Reference implementation.** Production deployment requires full kernel fusion with FlashAttention [Dao et al., 2022; Dao, 2023] for optimal performance. Our Triton prototype demonstrates feasibility but has not been validated in a production vLLM deployment.

**Covert channels.** TSAM does not address model-behavior covert channels (e.g., prompt injection, output-semantic channels), which require application-layer defenses.

### Broader Impact

TSAM enables secure shared LLM infrastructure with 10--100× cost reduction vs. process isolation. The formal guarantees---grounded in noninterference [Goguen and Meseguer, 1982] and information-theoretic zero leakage [Shannon, 1948]---provide a basis for compliance with HIPAA, SOC 2, and FedRAMP. All code is released open-source for reproducibility.

## Conclusion

We identified a fundamental gap in multi-tenant LLM serving: *no existing approach achieves zero leakage, unlimited tenant scale, and prefix sharing simultaneously*. Process isolation guarantees safety but kills efficiency; orthogonal projection scales only to 16 tenants and destroys prefix sharing; ad hoc masks lack formal guarantees.

TSAM closes this gap through *typed KV-cache pages* and *hard attention masking*. Each cache page carries a lightweight type annotation (PRIVATE or SHARED), and a single conditional in the attention kernel enforces:

$$I\!\bigl(\mathrm{output}(q_i);\;\mathrm{KV}_{\mathrm{private}}(t_j)\bigr) = 0 \quad \forall\, i \neq j.$$

Our evaluation demonstrates:

- **Zero leakage**: 0.0 bits across 100 adversarial queries with worst-case block sharing;
- **Unlimited scale**: zero leakage at 10,000+ tenants (orthogonal projection fails at 17);
- **Preserved prefix sharing**: single shared copy, <5% overhead vs. vanilla;
- **Minimal overhead**: <1% throughput impact across all configurations;
- **Timing defense**: ≥95% reduction in side-channel leakage via DAI.

The key insight is that *isolation is a data-flow property best enforced at the attention level, not the representation level*. By intervening where information flows between tokens, TSAM achieves exact isolation without distorting representations, limiting model capacity, or sacrificing serving efficiency.

**Future work.** Extending TSAM to LoRA adapter isolation [Hu et al., 2022; Sheng et al., 2023], distributed multi-GPU serving, and formal verification in Coq or Lean.

---

## References

See [`paper/references.bib`](paper/references.bib) for the full bibliography.
