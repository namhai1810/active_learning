# Failure-Driven Active Learning: flow, prompt và cơ sở toán học

Tài liệu này mô tả **đúng hành vi của nhánh `llm_main` trong code hiện tại**. Nó
được dùng như source-of-truth để chạy lại pipeline, kiểm tra implementation và
viết Method/Algorithm/Experimental Setup/Limitations cho paper.

Nhánh `oracle_hans_*`, ablation và offline theory validation là debug/reference,
không phải online main flow trừ khi được ghi rõ.

---

## 1. Tóm tắt phương pháp

Pipeline bắt đầu bằng một frozen, seed-independent HANS manifest. Chỉ bốn role
adaptive được đánh giá trước; final-test chỉ tồn tại dưới dạng ID placeholder
trong artifact mà pipeline nhìn thấy. Teacher LLM quan sát các
lỗi ở discovery split để đề xuất failure hypotheses có thể kiểm chứng. Mỗi
hypothesis được kiểm chứng bằng controlled probes mới, được sinh và QC hoàn toàn
trước khi nhìn thấy student outcome. Hypothesis chỉ được giữ nếu lower confidence
bound của controlled failure rate vượt ngưỡng verification.

Với mỗi hypothesis đã certified, teacher sinh các cặp NLI ba nhãn mới. Dữ liệu
được QC theo cả pair, lọc duplicate/leakage, rồi đưa vào các fresh-checkpoint
pilot interventions rời nhau. Gain pilot được đo bằng mức giảm HANS risk trên
`gain_eval`. Cuối cùng code dùng optimistic gain trừ acquisition cost để quyết
định STOP hay CONTINUE. CONTINUE chạy một fresh full LoRA intervention, checkpoint
đó trở thành starting checkpoint của round kế tiếp. Final-test chỉ được mở và
đánh giá sau khi toàn bộ adaptive loop đã kết thúc.

```text
Student checkpoint
    -> frozen deterministic five-way split before inference
    -> non-final HANS predictions
    -> [teacher critic estimates error probabilities
    -> teacher discovers failure hypotheses
    -> controlled probe generation and pre-outcome validation
    -> one-shot student verification
    -> verification LCB certification
    -> structured synthetic NLI generation and QC
    -> disjoint fresh-checkpoint pilots
    -> gain estimate and UCB
    -> cost-aware STOP/CONTINUE
       -> STOP: no training
       -> CONTINUE: one fresh full LoRA -> gain_eval -> updated checkpoint] x T
    -> one post-loop final-test evaluation
```

`run_main_active_learning.py` là multi-round implementation. Các stage scripts
riêng vẫn tồn tại để reproduce one-round/debug runs.

---

## 2. Ký hiệu

| Ký hiệu | Ý nghĩa trong code |
|---|---|
| \(h\) | Failure hypothesis do teacher sinh |
| \(K\) | Số hypothesis tối đa được tính trong union bound |
| \(T\) | Maximum round count được pre-register trong confidence bound |
| \(Z_{h,i}\) | Binary verification outcome của controlled set thứ \(i\) |
| \(n_h\) | Số valid controlled sets của hypothesis \(h\) |
| \(\widehat\nu_h\) | Empirical controlled-failure frequency |
| \(r_v(h)\) | Verification confidence radius |
| \(\tau_v\) | Verification threshold |
| \(X_{h,j}\) | Gain quan sát được ở pilot \(j\) |
| \(m_h\) | Số pilot thực sự chạy được |
| \(\widehat\Delta_h\) | Mean pilot gain |
| \(M\) | Bound giả định \(|X_{h,j}|\le M\) |
| \(b_h\) | Gain confidence radius |
| \(C(h)\) | Tổng component cost đã quy đổi về currency/unit cấu hình |
| \(\lambda\) | Cost coefficient `lambda_cost` |
| \(\widehat J(h)\) | Estimated net value |
| \(U(h)\) | Optimistic net value dùng để rank và stop |

---

## 3. Config và entry points

| Stage | Config | Entry point |
|---|---|---|
| Initial HANS evaluation | `configs/hans_qwen17b.yaml` | `scripts/evaluate_hans.py` |
| Hypothesis discovery | `configs/main_hypothesis_discovery.yaml` | `scripts/build_hypotheses.py` |
| Verification | `configs/main_verification.yaml` | `scripts/verify_hypotheses.py` |
| Synthetic generation | `configs/main_synthetic_generation.yaml` | `scripts/generate_synthetic_interventions.py` |
| Pilot gain | `configs/main_pilot_gain.yaml` | `scripts/run_main_pilot_gain.py` |
| Cost-aware decision | `configs/main_cost_aware.yaml` | `scripts/run_main_decision.py` |
| Multi-round main | `configs/main_multiround.yaml` | `scripts/run_main_active_learning.py` |
| Frozen Stage 2 data | `configs/stage2_benchmark.yaml` | `scripts/prepare_stage2_benchmark.py` |
| Fair Stage 2 run | `configs/stage2_benchmark.yaml` | `scripts/run_stage2_benchmark.py` |

Các main configs kế thừa `configs/main_common.yaml`; teacher identity phải được
freeze rõ ràng. Teacher lỗi làm pipeline dừng, không fallback ngầm sang HANS oracle.

```bash
python scripts/evaluate_hans.py --config configs/hans_qwen17b.yaml

python scripts/build_hypotheses.py \
  --config configs/main_hypothesis_discovery.yaml \
  --run-dir outputs/<existing_run>

python scripts/verify_hypotheses.py \
  --config configs/main_verification.yaml \
  --run-dir outputs/<existing_run>

python scripts/generate_synthetic_interventions.py \
  --config configs/main_synthetic_generation.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_pilot_gain.py \
  --config configs/main_pilot_gain.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_decision.py \
  --config configs/main_cost_aware.yaml \
  --run-dir outputs/<existing_run>
```

Mọi main stage kiểm tra:

```yaml
mode: llm_main
oracle_debug: false
```

để tránh trộn artifact main với oracle/debug.

---

## 4. Stage 0 — Student NLI và initial HANS evaluation

### 4.1 Student prompt

`QwenNLIStudent.format_prompt` dùng:

```text
System:
Classify natural-language inference. Choose exactly one label:
entailment, neutral, or contradiction.

User:
Premise: <premise>
Hypothesis: <hypothesis>
Label:
```

Nếu tokenizer có chat template, code dùng template với
`add_generation_prompt=True`. Code thử tắt thinking bằng
`enable_thinking=False`; nếu model không hỗ trợ argument thì gọi lại không có nó.

### 4.2 Prediction bằng candidate scoring

Student không generate free-form text. Với mỗi input, code chấm ba continuation:

```text
 entailment
 neutral
 contradiction
```

Với candidate label \(y\), score là mean token log-likelihood:

\[
s(y\mid x)=\frac{1}{|y|}\sum_{t=1}^{|y|}
\log p_\theta(y_t\mid x,y_{<t}).
\]

Prediction:

\[
\widehat y=\arg\max_{y\in
\{\text{entailment},\text{neutral},\text{contradiction}\}}s(y\mid x).
\]

Mỗi example vì thế luôn nhận một canonical MNLI label.

### 4.3 Mapping sang HANS

\[
g(y)=
\begin{cases}
\text{entailment}, & y=\text{entailment},\\
\text{non-entailment}, & y\in\{\text{neutral},\text{contradiction}\}.
\end{cases}
\]

`predictions.jsonl` chứa premise, hypothesis, HANS gold, binary prediction,
correctness, heuristic, subcase và raw three-class prediction.

### 4.4 Training objective

Training dùng cùng prompt. Target là label cộng EOS. Prompt tokens được mask bằng
`-100`, nên loss chỉ tính trên label tokens:

\[
\mathcal L(\theta)=-\sum_{t\in\text{label tokens}}
\log p_\theta(y_t\mid x,y_{<t}).
\]

Đây là standard per-example causal-LM cross-entropy với LoRA. Pair structure chỉ
điều khiển data selection; code không có contrastive hoặc pairwise loss.

---

## 5. Stage 1 — Deterministic five-way split

Input main method là một `predictions.jsonl` chưa có `split_role`. Stable IDs được
rank bằng:

\[
q_i=\operatorname{SHA256}(\text{split_seed}:\text{example_id}_i).
\]

| Role | Default fraction | Chức năng |
|---|---:|---|
| `hypothesis_discovery` | 0.125 | Critic và hypothesis discovery |
| `verification` | 0.125 | Context sinh verification probes mới |
| `intervention_pool` | 0.125 | Seed sinh synthetic training pairs |
| `gain_eval` | 0.125 | Applicability và gain evaluation |
| `final_test` | 0.500 | Final evaluation sau frozen CONTINUE |

Code kiểm tra:

1. ID tồn tại và unique.
2. Năm role pairwise disjoint.
3. Hợp của năm role phủ toàn bộ source IDs.
4. Mọi stage reconstruct đúng split và match `hans_split_v2.json`.
5. Artifact nguồn đã có `split_role` bị từ chối.

Đây là hash split, không stratify theo gold, heuristic hoặc subcase.

### Stage 2 sealed data firewall

`prepare_stage2_benchmark.py` tạo manifest trước inference, đánh giá chỉ non-final
roles và ghi final-test vào sealed artifact riêng. Round artifacts chỉ có final
IDs; không có input, label, prediction hoặc metric. `run_main_active_learning.py`
chỉ đọc sealed artifact sau khi mọi acquisition/training/STOP decision đã freeze.
Standalone legacy evaluation scripts vẫn có thể đánh giá full HANS và không được
dùng làm Stage 2 benchmark input.

Trong fixed-budget benchmark, các checkpoint trung gian chỉ được đánh giá post
hoc trên `gain_eval` để tạo accuracy-vs-examples/cost curves. Sealed `final_test`
được mở đúng một lần cho final checkpoint; MNLI matched/mismatched cũng chỉ được
đánh giá sau khi acquisition và training đã kết thúc. ID overlap làm run không
hợp lệ; lexical near-duplicate audit được log sau training và không được dùng để
thay đổi acquisition.

---

## 6. Stage 2 — Critic và failure-hypothesis discovery

### 6.1 Gold-blind error critic

`TeacherErrorProbabilityScorer` ước lượng:

\[
e_i=P(\text{student prediction is wrong}\mid
x_i,\widehat y_i,c_i),
\]

với \(c_i\) là optional confidence.

Critic chỉ nhận:

```text
id
premise
hypothesis
prediction
confidence, nếu source record có
```

Critic không nhận gold, correctness, HANS heuristic/subcase hoặc record ngoài
`hypothesis_discovery`. `evaluate_hans.py` hiện không ghi confidence, nên default
pipeline thường chạy critic không có signal này.

Payload yêu cầu standalone JSON:

```json
{
  "scores": [
    {
      "example_id": "one supplied ID",
      "error_probability": 0.0,
      "rationale": "short explanation"
    }
  ]
}
```

Response phải có đúng một score cho từng ID, không ID lạ/trùng và probability
finite trong `[0,1]`. Code không calibrate các probability này; chúng là
teacher-estimated scores dùng để sampling.

### 6.2 Error và correct pools

Sau critic scoring, code dùng ground-truth `correct` để tách:

```text
error pool   = prediction thực sự sai
correct pool = prediction thực sự đúng
```

Chỉ critic là gold-blind. Discovery tổng thể là supervised failure analysis trên
labeled split.

### 6.3 Error sampling

Ba strategy được hỗ trợ.

Normalization:

\[
w_i=\frac{e_i}{\sum_j e_j}.
\]

Threshold:

\[
i\text{ eligible}\iff e_i\ge\eta.
\]

Threshold không có fill policy; prompt có thể nhận ít hơn budget.

Strategy tên `exponential` thực tế dùng logistic weight:

\[
w_i=\sigma(\beta(e_i-\alpha))
=\frac{1}{1+\exp[-\beta(e_i-\alpha)]}.
\]

Weighted sampling without replacement dùng exponential-race priority:

\[
k_i=\frac{-\log U_i}{w_i},\qquad U_i\sim\operatorname{Uniform}(0,1),
\]

rồi chọn các key nhỏ nhất. Vì vậy tên `exponential` không có nghĩa
\(w_i=\exp(\beta e_i)\). Paper và code phải thống nhất điểm này.

Default:

```yaml
max_error_examples: 80
sampling_strategy: exponential
sampling_alpha: 0.5
sampling_beta: 10.0
```

### 6.4 Correct contrasts

Default `highest_error_score` lấy tối đa 20 correct records có critic error score
cao nhất. Đây là challenging false alarms, giúp teacher phân biệt failure pattern
thật với surface pattern trông khó nhưng student vẫn đúng.

### 6.5 Hypothesis prompt và schema

Teacher nhận selected errors/correct contrasts với:

```text
id, premise, hypothesis, gold, prediction, correct, optional confidence
```

Không truyền `heuristic` hoặc `subcase`. Teacher phải trả tối đa 5 hypotheses:

```json
{
  "hypotheses": [
    {
      "name": "non-empty string",
      "description": "non-empty string",
      "predicted_failure_behavior": "measurable behavior",
      "predicted_failure_labels": ["entailment or non-entailment"],
      "verification_criteria": ["outcome-independent rule"],
      "intervention_target": "what new NLI pairs should exercise",
      "support_example_ids": ["IDs shown to teacher"]
    }
  ]
}
```

Prompt yêu cầu compact, falsifiable hypotheses; criteria phải quyết định được từ
premise/hypothesis mà không nhìn student prediction. Support IDs phải thực sự đã
được đưa cho teacher. Main hypothesis bắt buộc có `source=llm_teacher`,
`heuristic=null`, non-empty behavior/target/criteria/support IDs.

Nếu selected error list rỗng, code không gọi hypothesis teacher và ghi
`no_discovery_errors`.

---

## 7. Stage 3 — Controlled failure verification

Verification records không được dùng trực tiếp để đo lại error rate. Chúng chỉ
làm context để teacher tạo **new controlled probe sets**.

### 7.1 Verification context

Với mỗi hypothesis, tối đa 24 records được chọn theo stable hash. Teacher chỉ thấy
`id`, `premise`, `hypothesis` và `gold`; không thấy prediction, correctness,
heuristic hoặc subcase.

### 7.2 Probe-set structure

Default sinh 50 sets/hypothesis, tối đa 8 sets/call. Mỗi set có đúng ba members:

1. **Target**: thỏa hypothesized mechanism; có exact MNLI gold label và một
   predeclared binary HANS failure label khác binary gold.
2. **Counterfactual/positive control**: mechanism-relevant minimal change mà
   correct behavior giúp phân biệt proposed mechanism.
3. **Invariant control**: thay irrelevant entity/surface detail nhưng giữ nguyên
   correct label.

Teacher phải state labels trước khi student được query và không được copy/lightly
paraphrase context. Mỗi probe có:

```text
premise
hypothesis
expected_label                 # exact three-class MNLI label
expected_failure_label         # target only; binary HANS label
label_rationale
constraint_rationale
```

Target failure label phải thuộc hypothesis-declared labels và khác
`mnli_to_hans(expected_label)`. Controls phải đặt failure label bằng `null`.

### 7.3 Pre-outcome validation

Một teacher call riêng independently solve exact NLI label và kiểm tra target /
control function, ambiguity, grammar, originality. Response cho mỗi probe gồm:

```text
validated_label
label_valid
constraint_satisfied
unambiguous
grammatical
original
suitable_control
reasons
```

Prompt yêu cầu reject thay vì repair và nhấn mạnh chưa có student prediction.
“Independent” ở code nghĩa là call/prompt/seed khác và outcome-blind; generator
và validator vẫn dùng cùng configured teacher model. Không nên gọi đây là
independent-model validation trong paper.

### 7.4 Fail-closed finalization

Một set bị reject toàn bộ nếu bất kỳ member nào:

- thiếu validation decision;
- validated label khác generated expected label;
- có một quality flag bằng `false`;
- copy hoặc near-copy verification context;
- duplicate trong set;
- duplicate/near-duplicate probe set trước.

Near duplicate dùng token-set Jaccard:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|},
\]

với default threshold `0.92`. Code không repair probe rồi thử lại.

### 7.5 Freeze-before-observe invariant

Cho mọi candidate hypotheses, code hoàn thành:

```text
generation -> validation -> deterministic finalization
```

trước khi instantiate student. Sau đó student được query đúng một lần trên frozen
accepted collection. Không có teacher call sau khi student outcomes đã lộ ra.
Điều này ngăn failure-conditioned retry và adaptive probe cherry-picking.

### 7.6 Controlled binary outcome

Với set \(i\) của hypothesis \(h\):

```text
T_hi = target maps to the predeclared wrong HANS label
C_hi = counterfactual prediction equals exact MNLI label
I_hi = invariant prediction equals exact MNLI label
```

\[
Z_{h,i}=\mathbf 1\{T_{h,i}\land C_{h,i}\land I_{h,i}\}.
\]

Target được đánh giá ở binary HANS level, còn controls phải đúng exact three-class
MNLI label. Generic target error không đúng predeclared failure label cho \(Z=0\).
Target fail nhưng một control sai cũng cho \(Z=0\). Vì vậy \(Z\) đo một
control-aware reproducible failure, không chỉ raw error rate.

### 7.7 Verification LCB

\[
\widehat\nu_h=\frac{1}{n_h}\sum_{i=1}^{n_h}Z_{h,i}.
\]

Union-adjusted Hoeffding radius:

\[
r_v(h)=\sqrt{\frac{\log(4KT/\delta)}{2n_h}}.
\]

\[
LCB_v(h)=\widehat\nu_h-r_v(h).
\]

Certification:

\[
h\text{ certified}\iff LCB_v(h)\ge\tau_v.
\]

Code còn buộc `n_h >= min_valid_probe_sets`; nếu không, `certified=false` với
status `unreliable_probe_design` dù numeric LCB vượt threshold.

Default:

```yaml
delta: 0.05
tau_v: 0.5
n_probe_sets: 50
min_valid_probe_sets: 40
K: 5
T: 5
```

Với 50 valid sets:

\[
r_v\approx0.244775,
\qquad
\widehat\nu_h\ge0.744775
\]

mới certified. Với đúng 40 valid sets:

\[
r_v\approx0.273666,
\qquad
\widehat\nu_h\ge0.773666.
\]

### 7.8 Ý nghĩa lý thuyết

LCB có diễn giải frequentist khi outcomes bounded và probe sets đủ độc lập. One-shot
generation, controls, scenario separation và pre-outcome validation giảm
adaptivity nhưng không chứng minh probes là iid. Certification không tự động là
causal claim.

---

## 8. Stage 4 — Synthetic intervention generation

Chỉ hypothesis có `certified=true` mới được dùng.

### 8.1 Seed selection

Seed chỉ đến từ `intervention_pool`; tối đa 24 seeds/hypothesis được chọn bằng
stable hash. Teacher nhận `id`, premise, hypothesis và `binary_gold`; không nhận
student outcome, heuristic hoặc subcase. Seeds chỉ là structural context và prompt
cấm copy/light paraphrase.

### 8.2 Pair schedule

Default cho mỗi certified hypothesis:

```yaml
counterfactual_pairs_per_hypothesis: 80
invariant_pairs_per_hypothesis: 80
reasoning_contrast_pairs_per_hypothesis: 0
```

Tổng 160 pairs = 320 examples trước QC. Mỗi pair có stable `pair_id`, `pair_type`
và đúng hai member roles.

**Counterfactual pair** có `base` và `relation_changed`: chỉ failure-relevant
relation đổi và correct three-class label phải đổi.

**Invariant pair** có `base` và `irrelevant_change`: chỉ irrelevant content đổi và
correct label phải giữ nguyên.

**Reasoning-contrast pair** có `base` và `reasoning_contrast`: hai members yêu cầu
contrasting reasoning steps. Implementation có sẵn nhưng default count bằng 0.

### 8.3 Explicit three-class labels

Mỗi member được gán trực tiếp một trong:

```text
entailment
neutral
contradiction
```

Không convert tự động HANS `non-entailment` thành `contradiction`, vì binary
non-entailment không phân biệt neutral và contradiction. Mỗi member/pair phải có
label rationale, targeting rationale, pair rationale và cited seed IDs.

### 8.4 Teacher QC

Một call riêng đánh giá:

```text
label_consistent
unambiguous
grammatical
relevant
pair_structure_valid
semantically_consistent
diverse
```

Prompt yêu cầu reject thay vì repair. QC chỉ thấy generated examples và hypothesis;
held-out text không được gửi vào generation/QC prompt. Generator và QC hiện vẫn
dùng cùng teacher model nên lỗi giữa hai pass có thể tương quan.

### 8.5 Deterministic pair-level filtering

Code tiếp tục kiểm tra:

1. Pair đủ đúng hai required member roles.
2. Hai members có metadata, seed IDs và pair rationale nhất quán.
3. Counterfactual labels phải khác nhau.
4. Invariant labels phải giống nhau.
5. Mọi teacher quality flag phải `true`.
6. Premise/hypothesis length trong configured bounds.
7. Không exact duplicate trong controlled pair.
8. Không duplicate/near-duplicate generated examples của pair trước.
9. Không exact/near overlap leakage references.

Nếu một member fail, cả pair bị reject.

Length filter hiện đếm số **unique tokens** do dùng `frozenset`, không phải tổng
token occurrences. Cần sửa nếu paper định nghĩa sentence length bằng tokenizer
length hoặc word count thông thường.

### 8.6 Leakage index

Sau khi teacher generation/QC hoàn tất, code mới tạo local signatures từ:

```text
intervention_pool
hypothesis_discovery
verification
generated verification probes
gain_eval
```

Leakage veto dùng exact SHA256 của normalized pair và token-set Jaccard. Raw
held-out text không được gửi lại cho teacher.

`final_test` cố ý không nằm trong leakage index để không materialize nó trước
decision. Hệ quả là accidental synthetic/final-test overlap không được local veto
phát hiện.

### 8.7 Data-volume requirement

Để chạy đủ default design cho một hypothesis:

\[
4\times32+128=256
\]

accepted examples là cần thiết. Generation tạo 320 examples, có rejection margin
64. Code không buộc đạt 256; thiếu dữ liệu có thể dẫn tới ít pilot hơn hoặc full
intervention nhỏ hơn requested size.

---

## 9. Stage 5 — Outcome-independent gain-eval applicability

Trước pilots, teacher operationalize hypothesis trên `gain_eval`. Teacher chỉ nhận
`id`, premise, hypothesis và gold; không nhận prediction, correctness, heuristic
hoặc subcase. Với mỗi record, teacher trả `applicable: boolean` cùng reason và một
batch-level `testable` flag.

Mục đích là chọn evaluation subgroup theo predeclared criteria mà không phụ thuộc
student outcome. Nếu bất kỳ batch nào trả `testable=false`, whole hypothesis trở
thành untestable với empty applicable IDs.

Applicability được phép nhìn HANS gold. Nó outcome-independent đối với student
nhưng không label-blind; paper cần mô tả đúng điều này.

---

## 10. Stage 6 — Pilot interventions và gain estimation

### 10.1 Default design

```yaml
m: 4
pilot_train_size: 32
gain_eval_size: 64
M: 1.0
delta: 0.05
K: 5
T: 1
```

`pilot_train_size` phải chẵn để giữ complete pairs.

### 10.2 Disjoint deterministic selection

Synthetic pairs và applicable evaluation records được stable-hash rank. Mỗi pilot
lấy một contiguous non-overlapping train block và eval block. Trong cùng một
hypothesis:

- pilot train sets không overlap;
- pilot eval sets không overlap;
- complete pairs luôn được giữ nguyên;
- pilot examples bị loại khỏi full reserve.

Giữa các hypothesis, gain-eval IDs có thể overlap vì selection được xây riêng.

### 10.3 Fresh-checkpoint invariant

Mỗi pilot:

1. derive deterministic seed;
2. load fresh student từ đúng starting checkpoint;
3. đo risk trước training;
4. train fresh LoRA adapter;
5. đo risk sau training trên cùng eval batch;
6. lưu gain rồi release model.

Không pilot nào bắt đầu từ adapter của pilot trước. Checkpoint descriptor và
training config được lưu; decision stage kiểm tra chúng khớp full recipe.

### 10.4 Risk và pilot gain

\[
R(f;D)=1-\operatorname{Accuracy}_{HANS}(f;D).
\]

\[
X_{h,j}=R(f_0;D_{h,j})-R(f_{h,j};D_{h,j}),
\]

trong đó \(f_0\) là starting checkpoint, \(f_{h,j}\) là post-pilot model và
\(D_{h,j}\) là pilot eval subset.

```text
X > 0: intervention giảm risk
X = 0: không đổi
X < 0: intervention làm tăng risk
```

Code kiểm tra risk trong `[0,1]` và \(|X|\le M\); không clip bound violations.

### 10.5 Feasible pilot count

\[
m_h=\min\left(
m,
\left\lfloor\frac{N_{train,h}}{\text{pilot_train_size}}\right\rfloor,
\left\lfloor\frac{N_{eval,h}}{\text{gain_eval_size}}\right\rfloor
\right).
\]

Nếu \(m_h<m\), shortage được log. Nhưng nếu \(m_h\ge1\), code vẫn tạo estimate
với actual \(m_h\). Chỉ \(m_h=0\) mới cho `insufficient_pilots`. Nếu theorem yêu
cầu fixed \(m\), nên fail closed hoặc loại candidate khi thiếu pilot.

### 10.6 Mean gain và UCB

\[
\widehat\Delta_h=\frac{1}{m_h}\sum_{j=1}^{m_h}X_{h,j}.
\]

\[
b_h=M\sqrt{\frac{2\log(2KT/\delta)}{m_h}}.
\]

Gain interval:

\[
[\widehat\Delta_h-b_h,\widehat\Delta_h+b_h].
\]

Optimistic gain dùng cho decision:

\[
UCB_\Delta(h)=\widehat\Delta_h+b_h.
\]

### 10.7 Statistical limitation

Bound giả định bounded, sufficiently independent pilot gains. Disjoint batches
giảm direct reuse nhưng không chứng minh independence: data cùng teacher, LoRA
runs cùng base/recipe, eval subsets là deterministic partitions, và across-hypothesis
evaluation overlap có thể tồn tại. Interval là conditional concentration bound,
không phải causal guarantee.

---

## 11. Stage 7 — Cost-aware STOP/CONTINUE

### 11.1 Full reserve

Với mỗi hypothesis, code lấy complete pairs chưa dùng trong pilots, stable-hash
rank và chọn tối đa `full_intervention_size: 128`. Size phải chẵn. Nếu chỉ còn ít
hơn 128 examples, code lấy số đang có. Vì vậy đây là **maximum full size**, không
phải exact required size. Empty reserve làm hypothesis không eligible.

### 11.2 Cost definition

`cost.type: full_pipeline` lưu component ledger gồm teacher input/output tokens,
generation/QC, verification, pilots, LoRA training và evaluation. Ledger tách:

1. `incurred_exploration`: resource đã tiêu trước decision và vẫn phải tính khi STOP;
2. `projected_intervention`: marginal full-training/evaluation resource chưa commit;
3. realized cumulative cost: incurred plus measured full training/evaluation
   cost khi CONTINUE (GPU timing nếu runtime hỗ trợ), cộng final evaluation sau loop.

Teacher tokens và pilot GPU time được đo khi provider/runtime cung cấp; full-run
GPU time và USD rates là configured assumptions/projections. Shared calls được
phân bổ một lần. Synthetic examples mặc định có zero human-annotation items;
human cost chỉ khác zero khi có human work thực sự được log/cấu hình.

### 11.3 Eligibility

Một hypothesis vào decision set khi:

1. verification certified;
2. gain status là `estimated`, \(m_h>0\);
3. \(\widehat\Delta_h\) và \(b_h\) tồn tại;
4. có non-empty validated full reserve;
5. pilot checkpoint và training recipe khớp current full recipe.

### 11.4 Net-value equations

\[
\widehat J(h)=\widehat\Delta_h-\lambda C(h).
\]

\[
U(h)=\widehat\Delta_h+b_h-\lambda C(h).
\]

Candidates được sort theo \(U(h)\) giảm dần, tie-break bằng hypothesis ID.

### 11.5 Stopping rule

\[
h^*=\arg\max_{h\in\mathcal H_{eligible}}U(h).
\]

\[
\text{action}=
\begin{cases}
\text{STOP}, & \mathcal H_{eligible}=\varnothing,\\
\text{STOP}, & \max_h U(h)\le0,\\
\text{CONTINUE with }h^*, & \max_h U(h)>0.
\end{cases}
\]

CONTINUE không bảo đảm realized gain dương; nó chỉ nói positive net value chưa bị
loại dưới optimistic bound và assumptions.

### 11.6 Numeric behavior

Với pilot count nhỏ, union-adjusted UCB có thể rất rộng. Phải report actual
radius và không diễn giải CONTINUE là guaranteed profitability. Vì `C(h)` dùng
currency cấu hình, không được thay số example count vào equation này.

STOP/CONTINUE phụ thuộc actual pilot observations, configured rates/projections và
remaining budget. Báo cáo phải lưu từng component thay vì chỉ một scalar tổng.

---

## 12. Stage 8 — Frozen decision, full intervention, final test

### 12.1 Freeze

Code ghi `decision/candidates.json` và `decision/decision.json` trước training.
Bytes của decision được giữ để xác nhận post-decision work không sửa quyết định.

### 12.2 STOP

STOP không load student, không train và không evaluate final-test trong decision
script.

### 12.3 CONTINUE

1. Lấy non-pilot full reserve của selected hypothesis.
2. Xác nhận actual training count bằng frozen selection plan; cost scalar được
   kiểm tra riêng bằng component ledger.
3. Load fresh student từ cùng starting checkpoint như pilots.
4. Không promote/reuse pilot adapter.
5. Train đúng một full LoRA với cùng recipe.
6. Đo before/after risk trên full applicable `gain_eval` subset.
7. Ghi metrics nhưng không recompute decision.
8. Trong multi-round mode, ghi `next_checkpoint.json` và chưa mở final-test.
9. Round kế tiếp dùng adapter vừa lưu làm starting checkpoint.
10. Sau khi loop kết thúc mới mở sealed final-test và evaluate đúng post-loop.
11. Xác nhận frozen decision bytes không thay đổi.

Observed full gain:

\[
\Delta^{full}_{obs}=R(f_0;D_h^{gain})-R(f_h^{full};D_h^{gain}).
\]

Final-test không được dùng để chọn hypothesis, tune \(lambda\), thay decision hoặc
chọn lại adapter.

---

## 13. Prompt infrastructure và reproducibility

Mỗi teacher request gồm:

```text
task
prompt_version
seed
system_prompt
payload
request_id
```

`request_id` là SHA256 của serialized request. Teacher calls/responses, versions,
seeds và selection IDs được lưu làm audit trail.

Hugging Face teacher:

- dùng chat template nếu có;
- JSON-serialize payload với sorted keys;
- không truncate input âm thầm;
- prompt vượt `max_input_tokens` làm stage fail;
- `do_sample=False`;
- response phải là standalone JSON object;
- markdown/prose ngoài JSON hoặc schema sai làm stage fail;
- không retry, repair JSON hoặc oracle fallback.

Caveats:

1. Greedy decoding làm seed thường không tạo sampling diversity. Payload/batch
   index khác vẫn có thể cho output khác, nhưng seed không hoạt động như sampling.
2. `prompt_version` là metadata/request identity; prompt text hard-code trong
   Python. Đổi version string không tự chọn template implementation khác.

| Chức năng | Default prompt version |
|---|---|
| Error critic | `discovery_error_probability_v1` |
| Failure discovery | `failure_discovery_v1` |
| Probe generation | `controlled_verification_probes_v1` |
| Probe validation | `verification_probe_validation_v1` |
| Synthetic generation | `targeted_synthetic_nli_pairs_v2` |
| Synthetic QC | `synthetic_pair_quality_control_v2` |
| Gain applicability | `hypothesis_operationalization_v1` |

Paper-level reproducibility nên version hóa actual prompt templates bằng immutable
files hoặc registry thay vì chỉ ghi free-form version string.

---

## 14. Artifact dependency graph

```text
predictions.jsonl
    ├── hypotheses.json
    ├── hypothesis_evidence.json
    ├── hans_split_v2.json
    └── teacher_calls/discovery.jsonl
            │
            ▼
verification/
    ├── generated_probes.jsonl
    ├── probes.jsonl
    ├── rejected_probes.jsonl
    ├── probe_set_outcomes.jsonl
    └── results.json
            │
            ▼
synthetic/
    ├── generated.jsonl
    ├── accepted.jsonl
    ├── rejected.jsonl
    └── manifest.json
            │
            ▼
gain/
    ├── pilot_observations.jsonl
    ├── gain_eval_applicability.json
    ├── gain_estimates.json
    └── pilots/<hypothesis>/<pilot>/adapter/
            │
            ▼
decision/
    ├── candidates.json
    ├── decision.json
    ├── full_intervention/<hypothesis>/adapter/   # CONTINUE only
    ├── post_intervention_metrics.json           # CONTINUE only
    ├── final_test_predictions.jsonl             # CONTINUE only
    └── final_test_metrics.json                  # CONTINUE only
```

Các `teacher_calls/*.jsonl` là audit trail quan trọng cho prompt analysis và paper
appendix; không nên chỉ giữ aggregate metrics.

---

## 15. End-to-end pseudo-code đúng implementation

```text
INPUT:
    starting student f0
    unsplit HANS predictions D
    teacher LLM
    split fractions/seeds
    verification delta, tau_v, K, T
    pilot m, train/eval sizes, M
    cost coefficient lambda

1. Split stable IDs:
       D_disc, D_ver, D_int, D_gain, D_final

2. DISCOVERY
   For each x in D_disc:
       critic estimates e(x) without gold/correctness
   Use gold to separate actual errors and correct controls.
   Sample errors by e(x); select challenging correct contrasts.
   Ask teacher for at most K falsifiable hypotheses H.

3. VERIFICATION DESIGN
   For each h in H:
       select context from D_ver
       generate target + counterfactual + invariant sets
       validate labels/constraints in a separate pre-outcome call
       reject whole set on any QC/duplicate failure
   Freeze all accepted sets for all h.

4. VERIFICATION MEASUREMENT
   Load f0 only after all designs are frozen.
   Query f0 once on accepted probes.
   Z = target predeclared failure AND both controls correct.
   For each h:
       nu_hat = mean(Z)
       r_v = sqrt(log(4 K T / delta) / (2 n_h))
       certify iff nu_hat - r_v >= tau_v
       and n_h >= min_valid_probe_sets

5. SYNTHETIC GENERATION
   For each certified h:
       select seeds from D_int
       generate new structured three-class NLI pairs
       run teacher QC
   Build local leakage signatures after generation/QC.
   Reject complete pairs on QC/schema/label/duplicate/leakage failures.

6. GAIN APPLICABILITY
   For each certified h:
       select D_gain records satisfying h without student outcomes.

7. PILOTS
   For each certified h:
       construct disjoint complete-pair train batches
       construct disjoint applicable eval batches
       For each feasible pilot j:
           load fresh f0
           risk_before = 1 - HANS_accuracy(f0, eval_j)
           train fresh LoRA on train_j
           risk_after = 1 - HANS_accuracy(f_hj, eval_j)
           X_hj = risk_before - risk_after
       delta_hat_h = mean_j X_hj
       b_h = M sqrt(2 log(2 K T / delta) / m_h)

8. DECISION
   For every certified h with valid gain and full reserve:
       exclude pilot train IDs
       C(h) = actual full example count
       J_hat(h) = delta_hat_h - lambda C(h)
       U(h) = delta_hat_h + b_h - lambda C(h)
   If no candidate or max_h U(h) <= 0:
       freeze STOP and exit
   Else:
       h* = argmax_h U(h)
       freeze CONTINUE(h*)

9. FULL INTERVENTION, CONTINUE ONLY
   Load fresh f0.
   Train one LoRA on non-pilot full reserve for h*.
   Evaluate full gain on applicable D_gain.
   Materialize D_final and evaluate f0 + full adapter.
   Never recompute the frozen decision.
```

---

## 16. Invariants cần giữ khi sửa code

### Data roles

- Discovery prompt chỉ dùng discovery role.
- Verification context chỉ dùng verification role.
- Synthetic seeds chỉ dùng intervention role.
- Pilot/full gain chỉ dùng gain-eval role.
- Final-test không ảnh hưởng generation, verification, ranking hoặc stopping.
- Verification probes không bao giờ là training data.

### Outcome independence

- Verification generation/QC hoàn tất trước student predictions.
- Không retry probe vì student không fail như mong đợi.
- Gain applicability không nhìn student prediction/correctness.
- \(\lambda\), budgets và confidence parameters fix trước full outcome.
- Decision freeze trước full training.

### Interventions

- Complete synthetic pairs không bị split.
- Pilot training sets rời nhau.
- Pilot examples bị loại khỏi full reserve.
- Mỗi pilot và full run bắt đầu từ cùng checkpoint.
- Pilot adapter không được promote.
- Full run dùng cùng LoRA recipe nếu extrapolate từ pilots.

### Validation

- Schema violation fail fast.
- Một member fail làm reject cả probe set hoặc synthetic pair.
- Không sửa/đổi label sau QC.
- Không map binary non-entailment sang three-class label tùy ý.

---

## 17. Điều code bảo đảm và không chứng minh

### Bảo đảm bằng validation/control flow

- deterministic role assignment và role disjointness;
- manifest consistency;
- strict teacher-response schemas;
- pre-outcome frozen probes;
- control-aware verification outcome;
- minimum valid-probe requirement;
- complete-pair preservation;
- pilot/full train-ID separation;
- same-checkpoint/same-recipe checks;
- frozen decision trước full/final evaluation;
- không reuse pilot adapter.

### Không tự chứng minh

- critic probabilities được calibrated;
- probes là iid;
- pilot gains là independent draws;
- controlled failure có đúng causal mechanism được mô tả;
- synthetic pairs chỉ thay đúng targeted mechanism;
- concentration coverage đúng dưới dependencies do cùng LLM sinh;
- CONTINUE bảo đảm positive realized gain;
- STOP loại trừ mọi unseen useful intervention;
- candidate-set-relative STOP loại trừ mọi future multi-step repair;
- final-test được physical isolation khỏi experimenter.

---

## 18. Mismatch/risk cần kiểm tra khi viết paper

1. **Adaptive bounds vẫn có assumptions.** Outer loop tồn tại, nhưng \(K\times T\)
   union adjustment chỉ hợp lệ nếu mỗi constituent interval conditionally valid
   theo prior history.
2. **`exponential` thực chất là logistic weighting.** Equation và tên phải thống
   nhất.
3. **Independent validation dùng cùng teacher.** Có pre-outcome independence,
   không có model independence.
4. **Critic không calibrated.** Không nên gọi score là calibrated probability nếu
   chưa có calibration study.
5. **Partial pilots vẫn eligible.** Actual \(m_h<m\) vẫn được estimate nếu
   \(m_h\ge1\).
6. **Full size có thể nhỏ hơn requested size.** Config hiện là maximum budget.
7. **USD cost là assumed.** Quantities measured/projected được tách rõ, nhưng rates
   vẫn do config đặt và projection error có thể đổi decision.
8. **UCB có thể quá lỏng.** Phải report actual radius và sensitivity.
9. **Sealed artifact không phải OS access control.** Code không truyền final content
   vào adaptive stages, nhưng experimenter vẫn quản lý filesystem.
10. **Leakage detector là lexical.** Token-set Jaccard bỏ word order/multiplicity
    và có thể bỏ sót semantic paraphrase.
11. **Same-teacher QC có correlated-error risk.** Nên cân nhắc second model,
    ensemble hoặc human audit.
12. **Target/control label resolution bất đối xứng.** Target fail ở binary HANS
    level, controls đúng ở exact MNLI level; cần ghi rõ.

---

## 19. Cấu trúc Method gợi ý cho paper

1. **Problem setup**: student, failure distribution, component intervention cost và
   adaptive finite-horizon objective.
2. **Data-role separation**: five-way split và no-cross-role-use protocol.
3. **Failure discovery**: gold-blind critic, error sampling, challenging correct
   contrasts và hypothesis schema.
4. **Controlled verification**: target/counterfactual/invariant sets, pre-outcome
   QC, binary \(Z\), LCB certification.
5. **Targeted intervention construction**: structured three-class pairs, teacher
   QC, deterministic filtering, leakage veto.
6. **Pilot gain estimation**: fresh checkpoint, disjoint pilots, risk reduction,
   gain UCB.
7. **Cost-aware action selection**: \(\widehat J(h)\), \(U(h)\), argmax và STOP.
8. **Multi-round update and held-out evaluation**: checkpoint transition, disjoint
   gain shards và post-loop sealed final-test protocol.
9. **Assumptions and limitations**: boundedness, independence, generated-data
   dependence, non-causality và finite-horizon conditional scope.

Ngôn ngữ nên dùng:

```text
"conditional on boundedness and sufficient independence assumptions"
"controlled reproducibility certificate"
"optimistic eligibility rule"
"finite-horizon, candidate-set-relative stopping decision"
```

Không nên claim khi chưa có bằng chứng bổ sung:

```text
"causally verified failure mechanism"
"guaranteed beneficial intervention"
"globally safe stopping"
"independent validation" nếu hiểu là independent model
"unconditional adaptive confidence guarantee"
```

---

## 20. Checklist trước research run

### Trước discovery

- [ ] Student checkpoint/adapter đã freeze.
- [ ] Source predictions chưa có `split_role`.
- [ ] Teacher path và exact model revision được lưu.
- [ ] Split fractions/seeds và run seed đã freeze.
- [ ] Không tune theo final-test subgroup.

### Trước verification

- [ ] Candidate count không vượt \(K\).
- [ ] \(\delta,\tau_v,K,T\), probe count và minimum valid count pre-register.
- [ ] Generator/validator identity được mô tả đúng.
- [ ] Giữ one-shot/pre-outcome probe policy.

### Trước synthetic generation

- [ ] Chỉ certified hypotheses được dùng.
- [ ] Training labels là exact three-class MNLI.
- [ ] Pair counts đủ cho pilots plus full reserve sau expected rejection.
- [ ] QC model/human audit policy đã freeze.

### Trước pilots

- [ ] \(m\), train/eval sizes, \(M,\delta,K,T\) đã freeze.
- [ ] Chính sách cho phép/cấm partial pilots đã pre-register.
- [ ] Checkpoint descriptor và LoRA recipe đã freeze.
- [ ] Applicability selection không nhìn student outcomes.

### Trước decision

- [ ] \(\lambda\) và cost definition đã freeze.
- [ ] Xác định full size là exact hay maximum.
- [ ] Kiểm tra numeric UCB không vacuous.
- [ ] Pilot IDs bị loại khỏi full reserve.
- [ ] Không tune rule từ full/final outcomes.

### Sau decision

- [ ] STOP không trigger training.
- [ ] CONTINUE chỉ train một fresh full intervention.
- [ ] Không reuse pilot adapter.
- [ ] Final-test chỉ dùng sau frozen decision/training.
- [ ] Báo cáo negative gains, shortages và rejected counts.
- [ ] Giữ đầy đủ teacher call/response audit trail.

---

## 21. Offline theory validation không thuộc online flow

`scripts/run_theory_validation.py` là oracle-debug, evaluation-only. Nó dùng một
separate disjoint reference dataset để đánh giá verification behavior, empirical
gain-UCB coverage và audit một online STOP decision.

Safe-stop audit chỉ xét robustly valid hypotheses trong candidate set đã biết:

\[
\max_h\left(\Delta_h^{reference}-\lambda C(h)\right)\le0.
\]

Passing audit không loại trừ unseen/future hypotheses, hypothesis interactions
hoặc multi-step interventions. Nó hỗ trợ assumptions về mặt thực nghiệm, không
chứng minh theorem và không phải online main step.

---

## 22. Source-code map

| Thành phần | File |
|---|---|
| Student prompt/candidate scoring | [`src/models/qwen_nli.py`](src/models/qwen_nli.py) |
| LoRA objective/training | [`src/models/lora_training.py`](src/models/lora_training.py) |
| Five-way split | [`src/data/hans_split.py`](src/data/hans_split.py) |
| Split/manifest guards | [`src/data/pipeline_split.py`](src/data/pipeline_split.py) |
| Critic scoring | [`src/hypotheses/error_scorer.py`](src/hypotheses/error_scorer.py) |
| Discovery sampling/prompt | [`src/hypotheses/llm_generator.py`](src/hypotheses/llm_generator.py) |
| Hypothesis schema | [`src/hypotheses/types.py`](src/hypotheses/types.py) |
| Teacher request/provider | [`src/teacher/base.py`](src/teacher/base.py) |
| Probe generation/QC/outcomes | [`src/verification/generated_probes.py`](src/verification/generated_probes.py) |
| Verification LCB | [`src/verification/bounds.py`](src/verification/bounds.py) |
| Gain applicability | [`src/verification/llm_operationalizer.py`](src/verification/llm_operationalizer.py) |
| Synthetic generation | [`src/synthetic/generator.py`](src/synthetic/generator.py) |
| Synthetic schema | [`src/synthetic/types.py`](src/synthetic/types.py) |
| Synthetic QC/leakage | [`src/synthetic/validator.py`](src/synthetic/validator.py) |
| Pilot/full interventions | [`src/intervention/synthetic.py`](src/intervention/synthetic.py) |
| Gain estimate/UCB | [`src/gain/estimation.py`](src/gain/estimation.py) |
| Cost-aware stopping | [`src/decision/cost_aware.py`](src/decision/cost_aware.py) |
| Full orchestration/final test | [`scripts/run_main_decision.py`](scripts/run_main_decision.py) |

Khi code thay đổi, phải cập nhật đồng thời formula, config defaults, role
permissions, prompt schemas, artifact schemas và research claims trong tài liệu
này.




