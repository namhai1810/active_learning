# Confidence-Safe Failure-Driven Synthetic Acquisition — Protocol v7

> **Trạng thái tài liệu:** đây là đặc tả độc lập và là source-of-truth cho toàn bộ model flow, protocol thí nghiệm và cách diễn giải kết quả v7.

## Đặc tả chuẩn v7

### 1. Mục tiêu và phạm vi

Protocol v7 là một closed loop active-learning pipeline cho NLI, gồm hai nhánh huấn luyện dùng **cùng đúng 128 ví dụ được chọn**:

- **Continuation (online):** sau mỗi vòng, tiếp tục huấn luyện LoRA hiện tại bằng 32 ví dụ mới và giữ nguyên AdamW state. Đây là model trực tiếp điều khiển các vòng thu thập tiếp theo và là kết quả online chính của v7.
- **Consolidation (controlled replay):** sau khi kết thúc thu thập, khởi tạo một LoRA mới từ base model rồi huấn luyện đúng một epoch trên toàn bộ 128 ví dụ. Nhánh này cô lập chất lượng của tập dữ liệu được chọn và tạo đối chứng one-pass công bằng.

Flow gồm target-only verification, refresh giả thuyết theo checkpoint hiện tại, chấm lại toàn bộ candidate pool ở mỗi vòng, huấn luyện continuation có optimizer continuity và consolidation cố định sau acquisition.

### 2. Flow end-to-end

```text
Inputs
  ├─ base student: Qwen/Qwen3-1.7B
  ├─ teacher:      Qwen/Qwen3-14B
  ├─ initial synthetic pool + hypothesis archive
  ├─ discovery / verification / stopping subsets
  └─ budget = 128 = 4 rounds × 32 examples
        │
        ▼
Round 1 initialization
  ├─ load the configured initial pool and archive
  ├─ mark all 5 initial hypotheses provisional-active
  └─ do not refresh before round 1
        │
        ▼
For round r = 1..4
  ├─ if r > 1: discovery refresh with current continuation checkpoint
  │    ├─ find current errors with uniform_gold_blind scoring
  │    ├─ propose 1–2 hypotheses only when active count < 5
  │    ├─ merge against the full historical archive
  │    └─ freshly verify all candidates; near-duplicates may reactivate
  ├─ retain certified + unresolved hypotheses
  ├─ generate synthetic pairs, run QC, merge cross-round duplicates
  ├─ re-score every eligible pair with the current checkpoint
  ├─ select 16 pairs = 32 examples; exclude previously selected IDs
  ├─ continuation update on only the new 32 examples
  │    └─ carry LoRA weights + full AdamW state into the next round
  └─ evaluate the fixed stopping subset for diagnostics
        │
        ▼
After round 4
  ├─ final discovery/verification audit (no new acquisition/training)
  ├─ consolidation: fresh base LoRA, one epoch on all 128 examples
  └─ evaluate both branches on gain report, sealed HANS and MNLI m/mm
```

Ba bộ HANS có vai trò khác nhau và không được trộn:

- **Discovery subset:** tìm failure để refresh giả thuyết.
- **Verification probes:** quyết định trạng thái certified / unresolved / refuted.
- **Stopping subset:** chỉ theo dõi tiến triển sau mỗi vòng; v7 **không** dùng small-gain early stopping.

### 3. Cấu hình cố định và entry point

| Thành phần | Giá trị v7 |
|---|---|
| Student | `Qwen/Qwen3-1.7B`, `float32` |
| Teacher | `Qwen/Qwen3-14B`, `bfloat16` |
| Tổng budget | 128 examples |
| Số vòng acquisition | 4 |
| Mỗi vòng | 16 synthetic pairs = 32 examples |
| Số giả thuyết active tối đa | 5 |
| Student LoRA | `r=8`, `alpha=16`; q/k/v/o + gate/up/down projections |
| Training mỗi vòng | 1 epoch, batch 4, accumulation 8, LR `1e-5` |
| Config | `configs/stage2_ours_verified_v7_dual_closed_loop.yaml` |
| Runner | `run_ours_verified_v7_dual_closed_loop.sh` |

Chạy từ repository root, trong môi trường `active_lr`, với hai GPU vật lý khác nhau:

```bash
TEACHER_GPU=1 STUDENT_GPU=0 bash run_ours_verified_v7_dual_closed_loop.sh
```

Runner ánh xạ teacher vào logical `cuda:0` và student vào logical `cuda:1`. Nó từ chối ghi đè một run đã hoàn tất và chỉ resume run còn dang dở.

#### 3.1 Student input, prediction và training objective

Student nhận một prompt gồm `Premise`, `Hypothesis` và cue `Label:`. Model không sinh câu trả lời tự do. Với từng nhãn trong `entailment`, `neutral`, `contradiction`, nó tính trung bình conditional log-likelihood của các token trong chuỗi nhãn; nhãn có score lớn nhất là prediction. Phân phối ba lớp là softmax trên ba score đó:

```text
q_y = mean_t log p(token_t của label y | prompt, các token label trước đó)
p_y = softmax(q)_y
NLL_gold = -log p_gold
H_norm = -sum_y p_y log(p_y) / log(3)
```

Khi đánh giá HANS, `entailment` giữ nguyên; cả `neutral` và `contradiction` được ánh xạ thành `non-entailment`.

Embedding dùng cho novelty là mean pooling có attention mask trên final hidden layer của cùng formatted prompt, sau đó L2-normalize. Pair embedding được tạo bằng trung bình hai member embeddings rồi L2-normalize lần nữa.

Objective huấn luyện thực tế của cả hai nhánh v7 là **per-example causal-LM cross-entropy** trên token của gold label (kèm EOS); token prompt bị mask khỏi loss. V7 giữ pair structure ở khâu generation/QC/selection, nhưng call-site `train_lora_stateful` không truyền pairwise-loss config, vì vậy **không có contrastive hoặc consistency auxiliary loss** trong run này. Các trường pairwise objective xuất hiện trong config tổng không phải objective được thực thi bởi runner v7.

#### 3.2 Frozen data roles và information firewall

30,000 HANS examples được chia cố định một lần bằng hash với `split_seed=20261011`:

| Role | Số examples | Quyền sử dụng trong flow |
|---|---:|---|
| `hypothesis_discovery` | 3,750 | phát hiện lỗi và refresh giả thuyết |
| `verification` | 3,750 | context độc lập để tạo/kiểm tra probes |
| `intervention_pool` | 3,750 | seed/context cho synthetic intervention |
| `gain_eval` | 3,750 | chia tiếp thành stopping và gain report |
| `final_test` | 15,000 | sealed HANS final evaluation |

`gain_eval` được chia prediction-blind thành stopping subset 900 examples (30 cho mỗi HANS subcase) và gain-report subset 2,850 examples. Stopping subset có thể được xem trong loop; gain report chỉ dùng cho báo cáo cuối.

Trong loop, sealed final-test text không được đưa vào discovery, verification, generation, selection hoặc stopping. Sau khi model đã train xong, code mới materialize 15,000 final rows và chạy leakage audit. Audit sau cùng xác nhận không có ID overlap/near-duplicate, nhưng không được diễn giải là selector đã nhìn thấy final-test trước training.

Training v7 chỉ dùng 128 **synthetic examples đã được chọn**. Verification probes, discovery examples, stopping/gain-report rows, sealed HANS và MNLI validation đều không được đưa vào loss.

### 4. Khởi tạo round 1

V7 đọc initial synthetic pool và hypothesis archive từ `closed_loop.initial_pool_dir` trong resolved `config.yaml`. Input artifact phải có pool synthetic đã qua QC, hypothesis definitions, verification evidence và completion marker hợp lệ.

Cả 5 giả thuyết ban đầu được đưa vào tập `provisional-active`; mọi kết luận cũ được chuẩn hóa thành `unresolved` cho đến khi v7 tạo bằng chứng verification mới. Round 1 dùng initial pool trực tiếp và chưa chạy refresh. Refresh theo checkpoint bắt đầu trước round 2.

### 5. Refresh và vòng đời giả thuyết

Trước rounds 2–4, checkpoint continuation hiện tại được dùng để đánh giá lại discovery set. Discovery scorer là `uniform_gold_blind` với score cố định 0.5. Gold label HANS vẫn được dùng bên ngoài scorer để xác định prediction nào sai, nên mô tả đúng là **scorer gold-blind**, không phải toàn bộ discovery gold-free.

Quy tắc refresh:

1. Thu các lỗi hiện tại trên discovery set.
2. Tính capacity từ số giả thuyết active **trước verification**.
3. Chỉ tìm thêm 1–2 giả thuyết khi capacity còn trống và có lỗi discovery.
4. So khớp đề xuất với toàn bộ historical archive, không chỉ tập active.
5. Merge các đề xuất gần trùng; một giả thuyết đã resolved có thể được reactivate nếu evidence mới khớp với nó.
6. Verify lại candidates bằng probes mới và gán một trong ba trạng thái:
   - `certified`: đủ bằng chứng failure có hệ thống;
   - `unresolved`: chưa đủ bằng chứng để kết luận;
   - `refuted`: bằng chứng chống lại giả thuyết.
7. Chỉ `certified` và `unresolved` được giữ để sinh/chọn dữ liệu.

Sau round 4 có thêm một final audit bằng checkpoint cuối. Audit này chỉ cập nhật bằng chứng và trạng thái giả thuyết; nó không mở thêm round huấn luyện.

### 6. Target-only adaptive verification

V7 chỉ sinh và đánh giá **target probes**; protocol không tạo counterfactual/invariant control probes. Lịch adaptive dự kiến nhìn dữ liệu tại `n = [20, 30]`, với tối đa 30 target probes cho mỗi giả thuyết. Một outcome bằng 1 khi student cho đúng **predeclared wrong binary HANS label** trên target probe, tức failure mà giả thuyết dự đoán thật sự xuất hiện.

Với target error count `e_n` tại lần nhìn có `n` probes:

```text
p_hat = e_n / n
radius = sqrt(log(4*K*T/delta) / (2*n))
lower = p_hat - radius
upper = min(1, p_hat + radius)
```

Trong run chuẩn: `delta=0.05`, `tau=0.5`, `K=5`, verification `max_rounds=5`, hai looks, nên `T_eff = 5 × 2 = 10`.

Quyết định:

```text
certified  nếu lower >= tau
refuted    nếu upper <  tau
unresolved nếu chưa rơi vào hai trường hợp trên
```

Nếu chưa quyết định ở 20 probes, hệ thống tăng lên 30. Nếu generation/QC không tạo đủ `min_valid_probe_sets=20`, hypothesis không được certify ngay cả khi empirical failure rate cao; nó được đánh dấu unreliable/unresolved. Vì vậy số probes quan sát thực tế có thể nhỏ hơn schedule (run seed 42 có trường hợp `n=8` và final audit `n=0`). Phép bound dùng union correction theo số hypotheses và số lần nhìn; không được diễn giải nó như p-value hoặc confidence interval không hiệu chỉnh.

### 7. Sinh dữ liệu, QC và pool merge

Mỗi round, teacher sinh candidates cho các giả thuyết đang `certified` hoặc `unresolved`:

- mục tiêu: đủ ít nhất 32 examples sau QC;
- giới hạn thông thường: 96 generated examples, có thể sinh tối đa 384 nếu cần;
- giữ cấu trúc **complete pair** gồm hai members: counterfactual pair đổi quan hệ trung tâm và đổi gold label, còn invariant pair chỉ đổi chi tiết không liên quan và giữ gold label; v7 cấu hình 0 reasoning-contrast pairs;
- chạy các kiểm tra schema, nhãn, lexical/semantic constraints và consistency;
- merge với cumulative pool của các round đã có;
- loại near-duplicate xuyên vòng bằng pair-level Jaccard threshold `0.92`.

Pool sau merge là tập ứng viên; nó không đồng nghĩa với tập train. Chỉ các pair được selector chọn mới tiêu budget.

### 8. Chấm điểm và chọn 16 pairs mỗi vòng

Ở đầu mỗi vòng, model continuation hiện tại chấm lại toàn bộ eligible pool. Pair đã chọn ở các vòng trước không được chọn lại, nhưng embedding của chúng vẫn làm novelty anchors.

Với mỗi pair `i`:

```text
D_i = 0.65 * rank_pct(gold_NLL_i)
    + 0.35 * rank_pct(normalized_entropy_i)

S_i = D_i
    + 0.50 * rank_pct(novelty_i)
    + 0.35 * (sqrt(m_h + 1) - sqrt(m_h))
    + 0.50 / (1 + m_h)
```

Trong đó:

- các percentile ranks được tính lại theo eligible pool của **chính round hiện tại**, có xử lý ties;
- `gold_NLL` và entropy đến từ base member của pair;
- pair embedding là trung bình embedding hai members rồi L2-normalize;
- `novelty_i` đo khoảng cách tới các selected-pair anchors;
- `m_h` là số pairs của hypothesis `h` đã được chọn tích lũy, kể cả các lựa chọn trước đó trong cùng round.

Selector greedily lấy 16 pairs có score cao nhất. V7 không có hard quota cho từng hypothesis; hai hạng tử cuối chỉ tạo diminishing-return và balance mềm.

### 9. Hai nhánh huấn luyện công bằng

#### 9.1 Continuation branch

Sau khi chọn 16 pairs, model nhận đúng 32 examples mới của round đó:

- round 1: khởi tạo LoRA + AdamW từ base;
- rounds 2–4: nạp lại LoRA và **toàn bộ optimizer state** của round trước;
- mỗi round đúng một optimizer update hiệu dụng (`batch=4`, `grad_accum=8`, 32 examples, 1 epoch);
- trạng thái optimizer tiến liên tục `0 → 1 → 2 → 3 → 4`;
- tensor audit trước/sau update xác nhận checkpoint thật sự thay đổi.

Checkpoint continuation sau round `r` vừa là kết quả online ở thời điểm đó, vừa điều khiển discovery refresh và re-scoring ở round kế tiếp.

#### 9.2 Consolidation branch

Chỉ sau khi acquisition kết thúc:

- tạo LoRA mới từ cùng base model;
- train đúng một epoch trên union của toàn bộ 128 examples đã chọn;
- batch/accumulation giống continuation, cho tổng cộng 4 optimizer steps;
- mỗi example có đúng một exposure;
- không tham gia discovery, verification, scoring hay lựa chọn dữ liệu.

Vì hai nhánh dùng cùng selected IDs, cùng 128 examples, cùng một exposure/example và cùng 4 effective steps, so sánh này chủ yếu đo ảnh hưởng của **online continuation/order** so với **one-pass consolidated replay**.

### 10. Dừng và đánh giá

Acquisition dừng khi xảy ra một trong các điều kiện:

- không còn hypothesis active;
- không thể tạo đủ 32 examples hợp lệ cho một round;
- đạt 4 rounds / budget 128.

Độ chính xác trên stopping subset được ghi lại sau mỗi round để chẩn đoán. Controller v7 **không áp dụng small-gain early stopping**, vì vậy không được mô tả stopping curve như cơ chế quyết định dừng.

Sau acquisition và final audit, cả continuation và consolidation được đánh giá trên:

- gain-report subset;
- sealed HANS (15,000 examples), gồm overall và non-entailment accuracy;
- MNLI matched và mismatched;
- leakage audit chạy sau selection/training.

MNLI và HANS là các evaluation views riêng. Kết quả không phải một metric trộn chung hai bộ dữ liệu.

### 11. Kết quả run chuẩn seed 42

Output root:

```text
outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42
```

Stopping accuracy theo checkpoint continuation:

| Checkpoint | Accuracy |
|---|---:|
| Base trước acquisition | 63.2222% |
| Sau round 1 | 63.6667% |
| Sau round 2 | 63.8889% |
| Sau round 3 | 63.8889% |
| Sau round 4 | 64.1111% |

Phân bổ 64 selected pairs (128 examples):

| Hypothesis | Pairs |
|---|---:|
| Contextual Invalidation Misclassification | 6 |
| General-to-Specific Misclassification | 25 |
| Subject-Object Reversal Misclassification | 1 |
| Extra Context Entailment Failure | 1 |
| Entity Addition Misclassification | 6 |
| Conditional Statement Misclassification | 25 |

`Extra Context Entailment Failure` là giả thuyết mới được chấp nhận ở refresh trước round 4. Stop reason thực tế là `maximum_budget_reached_with_remaining_hypotheses`.

Final audit giữ 5 giả thuyết active:

| Hypothesis | Final state |
|---|---|
| Contextual Invalidation Misclassification | unresolved |
| General-to-Specific Misclassification | certified |
| Extra Context Entailment Failure | unresolved |
| Entity Addition Misclassification | certified |
| Conditional Statement Misclassification | unresolved |

Kết quả cuối:

| Branch | Gain report overall | Gain report non-ent. | Sealed HANS overall | Sealed HANS non-ent. | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|---:|---:|
| Continuation | 64.0702% | 35.6504% | **65.0000%** | **36.3296%** | **58.6347%** | **58.9605%** |
| Consolidation | 63.5088% | 34.4116% | 64.7200% | 35.5817% | 58.4412% | 58.6554% |

Trong run này continuation cao hơn consolidation, nhưng đây mới là seed 42; không được suy rộng thành kết luận thống kê đa-seed.

### 12. Pseudocode chuẩn

```python
cfg = resolve("configs/stage2_ours_verified_v7_dual_closed_loop.yaml")
pool, archive = load_initial_state(cfg.closed_loop.initial_pool_dir)
active = make_all_initial_hypotheses_provisional(archive)

continuation = init_lora_student(cfg.student)
optimizer = init_adamw(continuation, cfg.training)
selected_ids = set()
selected_pairs = []

for round_id in range(1, 5):
    if round_id > 1:
        errors = discover_current_errors(
            model=continuation,
            scorer="uniform_gold_blind",
            threshold=0.5,
        )
        proposals = propose_if_capacity(active, errors, max_active=5, max_new=2)
        candidates = merge_against_full_archive(archive, proposals)
        active = target_only_adaptive_verify(
            candidates,
            looks=[20, 30],
            tau=0.5,
            delta=0.05,
            K=5,
            effective_T=10,
        )

    generated = teacher_generate(active.states("certified", "unresolved"))
    accepted = quality_control(generated)
    pool = cross_round_pair_merge(pool, accepted, jaccard_threshold=0.92)

    eligible = pool.exclude(selected_ids)
    rescored = score_with_current_checkpoint(continuation, eligible)
    chosen = greedy_select_pairs(rescored, k=16, prior_anchors=selected_pairs)

    continuation, optimizer = train_one_round(
        continuation,
        optimizer,
        examples=flatten(chosen),  # exactly 32 new examples
        carry_full_optimizer_state=True,
    )
    selected_ids.update(ids(chosen))
    selected_pairs.extend(chosen)
    record_diagnostic_stopping_accuracy(continuation)

final_active = final_discovery_verification_audit(continuation, archive)

consolidation = init_fresh_lora_student(cfg.student)
consolidation = train_one_epoch(consolidation, flatten(selected_pairs))

evaluate_both(
    continuation,
    consolidation,
    suites=["gain_report", "sealed_hans", "mnli_matched", "mnli_mismatched"],
)
run_post_training_leakage_audit()
```

### 13. Artifacts và source-of-truth

Các file cần kiểm tra khi audit một run:

```text
<output_root>/.complete
<output_root>/config.yaml
<output_root>/rounds.jsonl
<output_root>/comparison.json
<output_root>/V7_COMPARISON.md
<output_root>/selected_training_examples.jsonl
<output_root>/hypothesis_archive.json
<output_root>/leakage_audit.json
<output_root>/training/round_001..004/...
<output_root>/refresh/round_002..005/...
<output_root>/branches/continuation/...
<output_root>/branches/consolidation/...
```

Code/config quyết định hành vi:

| Chức năng | Source |
|---|---|
| Entry point hai GPU | `run_ours_verified_v7_dual_closed_loop.sh` |
| Cấu hình v7 | `configs/stage2_ours_verified_v7_dual_closed_loop.yaml` |
| Orchestration | `scripts/run_stage2_closed_loop.py` |
| Closed-loop controller | `src/active_learning/closed_loop.py` |
| Frozen HANS split | `src/data/hans_split.py` |
| Pair scoring/selection | `src/benchmark/selection.py` |
| Student prompt, prediction và embedding | `src/models/qwen_nli.py` |
| LoRA/AdamW training | `src/models/lora_training.py` |
| Synthetic QC/leakage checks | `src/synthetic/validator.py` |
| Verification runner | `scripts/verify_hypotheses.py` |
| Hoeffding bounds | `src/verification/bounds.py` |

Checklist tối thiểu trước khi báo cáo:

- `.complete` tồn tại và báo `v7 dual fair closed-loop complete`;
- 4 rounds, mỗi round đúng 16 pairs / 32 examples;
- tổng selected IDs là 128 và không trùng;
- continuation optimizer continuity được xác nhận từ rounds 1–4;
- continuation và consolidation đều có 4 effective optimizer steps;
- consolidation dùng đúng union selected IDs của continuation;
- final audit không tạo thêm training round;
- gain report, sealed HANS, MNLI m/mm và leakage audit đều có artifact;
- mọi claim vượt quá seed 42 được ghi rõ là chưa được multi-seed xác nhận.

### 14. Khung dùng để viết paper

Phần Method nên đi theo đúng thứ tự phụ thuộc của thuật toán:

1. **Task and models:** three-way NLI student, binary HANS mapping và teacher dùng cho hypothesis/probe/synthetic generation.
2. **Frozen data protocol:** five-way HANS roles, stopping/report split và sealed-test firewall.
3. **Checkpoint-conditioned hypothesis loop:** discovery refresh, archive merge và ba hypothesis states.
4. **Target-only confidence-safe verification:** outcome definition, adaptive looks, Hoeffding bound và điều kiện underfill.
5. **Synthetic intervention pool:** pair semantics, QC và cross-round deduplication.
6. **Checkpoint-conditioned acquisition:** gold NLL, normalized entropy, contextual novelty và soft hypothesis balancing.
7. **Dual fair training protocol:** online continuation là closed-loop controller; fresh consolidation là controlled one-pass comparison.
8. **Stopping and evaluation:** budget-based stopping, gain report, sealed HANS, MNLI m/mm và leakage audit.

Khi viết paper, continuation phải được trình bày là **online v7 outcome chính**, còn consolidation là đối chứng dùng cùng acquisition trace và cùng fixed budget. Bảng chính nên báo cáo cả overall và non-entailment HANS accuracy, MNLI matched/mismatched, đủ số examples, số optimizer steps và stop reason.

Các claim được dữ liệu hiện tại hỗ trợ:

- v7 là một checkpoint-conditioned closed loop dưới fixed budget 128;
- target-only verification giữ lại hypothesis bằng lower confidence bound có union correction;
- continuation và consolidation dùng cùng 128 selected IDs, cùng một exposure/example và cùng 4 optimizer steps;
- trên run seed 42, continuation cao hơn consolidation ở các metrics trong bảng kết quả.

Các claim **không** nên viết nếu chưa có thí nghiệm bổ sung:

- superiority có ý nghĩa thống kê hoặc tổng quát qua nhiều seeds;
- từng component riêng lẻ gây ra mức tăng cuối nếu chưa có ablation;
- xác suất ba lớp là calibrated probability tuyệt đối;
- lợi thế chi phí hoặc lợi thế so với human annotation nếu chưa tổng hợp cost accounting tương ứng.
