# Confidence-Safe Failure-Driven Synthetic Acquisition (Protocol v5)

## 0. Phạm vi và trạng thái tài liệu

Tài liệu này là mô tả kỹ thuật chuẩn của pipeline `ours_verified` v5 cho Stage 2. Mục tiêu là đủ chính xác để dùng làm cơ sở viết phần Method, Algorithm, Experimental Setup, Ablation và Limitations của paper, đồng thời đủ cụ thể để audit implementation.

Tài liệu phân biệt rõ ba loại phát biểu:

1. **Định nghĩa protocol**: hành vi được code và config quy định.
2. **Bảo đảm có điều kiện**: mệnh đề thống kê chỉ đúng dưới các giả định được nêu.
3. **Kết quả thực nghiệm**: số liệu quan sát được ở seed 42, không phải bảo đảm tổng quát.

Entry point chuẩn hiện tại là:

```bash
POOL_CUDA_DEVICE=0 RUN_CUDA_DEVICE=1 ./run_ours_verified_v5.sh
```

Protocol v5 cố định seed 42 và maximum budget 128. Discovery và verification được tái sử dụng từ artifact v2 đã hoàn tất; v5 sinh lại synthetic pool, chấm toàn bộ pool bằng student, chọn dữ liệu bằng difficulty cộng contextual novelty, huấn luyện CE theo bốn round và áp dụng stopping protocol kế thừa từ v4.

Tại thời điểm audit tài liệu này, run v5 seed 42 đã hoàn tất:

- synthetic pool: 480 examples được sinh, 476 examples/238 complete pairs được chấp nhận;
- training: dùng đủ 128 examples qua 4 round, không early-stop;
- HANS final accuracy: 0.6482;
- HANS final non-entailment accuracy: 0.3648;
- MNLI matched/mismatched: 0.5881/0.5894.

Các số trên chỉ là kết quả seed 42. Không được viết chúng thành guarantee của thuật toán. Phiên bản cũ của tài liệu được giữ ở `Flow_legacy.md` để truy vết lịch sử; nó không phải source-of-truth cho v5.

---

## 1. Tóm tắt đóng góp của flow mới

Flow cũ có hai điểm dễ làm mất hypothesis hữu ích và làm giảm chất lượng acquisition:

- chỉ giữ hypothesis đã được Hoeffding LCB certify, vì vậy một hypothesis có empirical failure rate cao nhưng interval còn rộng có thể bị loại như thể nó đã bị bác bỏ;
- selector cũ cân bằng hoặc phân bổ bằng các quota/score dựa mạnh vào verification controls, thay vì trực tiếp ưu tiên các synthetic target mà student hiện tại thấy khó và loại redundancy ở mức biểu diễn.

Flow v5 thay bằng hai nguyên tắc:

### 1.1 Confidence-safe retention

Một hypothesis chỉ bị loại khi dữ liệu verification đủ mạnh để **bác bỏ** nó:

\[
\operatorname{UCB}_h < \tau_v.
\]

Hypothesis đã certify và hypothesis chưa ngã ngũ đều được giữ:

\[
\mathcal H_{\mathrm{keep}}
=\mathcal H_{\mathrm{certified}}\cup\mathcal H_{\mathrm{unresolved}}.
\]

Cách làm này không hạ ngưỡng Hoeffding và không tuyên bố unresolved là đúng. Nó chỉ tránh đồng nhất “chưa đủ bằng chứng để certify” với “đã có bằng chứng để loại”.

### 1.2 Student-aware entropy-novelty acquisition

Sau khi QC, student base checkpoint chấm mọi synthetic example đúng một lần. Với mỗi complete pair, selector:

- dùng NLL của gold label để phát hiện target mà student đang gán xác suất thấp cho nhãn đúng;
- dùng entropy để nhận diện vùng student còn phân vân;
- dùng embedding của chính student để giảm trùng lặp ngữ nghĩa;
- trước hết chọn một pair cho mỗi retained hypothesis như một coverage constraint;
- sau đó greedy chọn các pair còn lại bằng difficulty cộng novelty.

Controls vẫn quan trọng để tạo pair và làm diagnostic, nhưng không còn là hard certification gate và không được dùng để làm một target khó trở nên “dễ” trong difficulty score.

---

## 2. End-to-end flow

```text
Frozen HANS manifest (seed-independent)
    |
    +--> discovery split
    |      -> base-student predictions + gold correctness
    |      -> gold-blind teacher critic estimates error probability
    |      -> diversity-aware evidence sampling
    |      -> teacher proposes exactly five falsifiable hypotheses
    |
    +--> verification split + new controlled probes
    |      -> target / counterfactual control / invariant control
    |      -> independent QC call before student outcomes
    |      -> adaptive looks at n = 20, 30, 40, 50, 60
    |      -> simultaneous Hoeffding interval
    |      -> state = certified / unresolved / refuted
    |
    +--> confidence-safe retention
    |      -> keep certified + unresolved
    |      -> exclude only UCB < tau_v
    |
    +--> fresh synthetic generation from intervention seeds
    |      -> counterfactual and invariant complete pairs
    |      -> teacher QC + deterministic structural checks
    |      -> duplicate/leakage filtering
    |
    +--> base-student acquisition scoring over the accepted pool
    |      -> gold NLL + normalized entropy
    |      -> contextual embeddings
    |      -> one coverage pair per retained hypothesis
    |      -> greedy difficulty + novelty until 64 pairs
    |
    +--> cumulative LoRA CE training, 32 examples/round
    |      -> frozen 900-example stopping_eval after each round
    |      -> paired gain diagnostics + practical patience rule
    |      -> at most 4 rounds / 128 examples
    |
    +--> checkpoint fixed
           -> one held-out 2,850-example gain report
           -> one sealed 15,000-example HANS final evaluation
           -> MNLI matched/mismatched evaluation
           -> leakage audit and report
```

Hai stage GPU trong shell script là tuần tự, không chạy đồng thời. `POOL_CUDA_DEVICE` dành cho teacher generation; `RUN_CUDA_DEVICE` dành cho student scoring/training/evaluation. Khi chỉ expose một physical GPU qua `CUDA_VISIBLE_DEVICES`, process nhìn nó như local `cuda:0`.

---

## 3. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| \(h\) | một failure hypothesis |
| \(K\) | số hypothesis tối đa trong simultaneous bound; ở đây \(K=5\) |
| \(\tau_v\) | verification threshold; ở đây \(0.5\) |
| \(Z_{h,i}\) | target-failure outcome nhị phân của probe set thứ \(i\) |
| \(n_h\) | số valid probe sets đã đánh giá cho \(h\) |
| \(\widehat\nu_h\) | empirical target-failure rate |
| \(r_h\) | Hoeffding confidence radius |
| \(L_h,U_h\) | lower/upper confidence bound của \(h\) |
| \(x=(p,q,y)\) | một NLI example gồm premise, hypothesis, gold label |
| \(s_y(x)\) | mean token log-likelihood score của candidate label \(y\) |
| \(p_y(x)\) | softmax probability trên ba candidate label |
| \(NLL(x)\) | negative log probability của gold label |
| \(H(x)\) | entropy chuẩn hóa của phân phối ba nhãn |
| \(e(x)\) | contextual embedding L2-normalized của example |
| \(D_j\) | difficulty của synthetic pair \(j\) |
| \(N(j\mid S)\) | novelty của pair \(j\) so với tập pair đã chọn \(S\) |
| \(G_t\) | paired accuracy gain sau training round \(t\) |

Trong HANS, nhãn evaluation là binary `entailment`/`non-entailment`. Trong synthetic training và MNLI, label space là ba nhãn `entailment`, `neutral`, `contradiction`. Code không tự ý coi hai không gian nhãn này là giống nhau.

---

## 4. Frozen data protocol và information firewall

### 4.1 Five-way HANS split

Toàn bộ 30,000 HANS examples được partition một lần bằng hash với `split_seed = 20261011`:

| Role | Số examples | Được dùng để làm gì |
|---|---:|---|
| `hypothesis_discovery` | 3,750 | phát hiện lỗi và sinh hypothesis |
| `verification` | 3,750 | context cho controlled verification |
| `intervention_pool` | 3,750 | seed/context để sinh dữ liệu can thiệp |
| `gain_eval` | 3,750 | stopping và held-out gain report |
| `final_test` | 15,000 | final HANS evaluation duy nhất |

Frozen manifest hash của benchmark hiện tại là:

```text
871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0
```

Đây là global deterministic hash split, không phải stratified split. Cân bằng ở stopping subset được thực hiện sau đó và không làm thay đổi five-way manifest.

### 4.2 Tách `gain_eval`

`gain_eval` được chia prediction-blind thành:

- `stopping_eval`: 900 examples, đúng 30 examples cho mỗi một trong 30 HANS subcases;
- `gain_report`: 2,850 examples còn lại, chỉ mở sau khi checkpoint đã được chọn.

Membership được xếp hạng bằng SHA-256 dựa trên manifest hash, namespace và ID. Selection chỉ đọc `id`, `subcase`, `heuristic`, `gold`, không đọc prediction/correctness. Partition hash là:

```text
2be4c5e62e2983b565a254b894fce2ff4ed8358adad9371629a8900bc1cddaff
```

### 4.3 Final-test firewall

Trong adaptive loop, final-test text không được cấp cho discovery, verification, synthetic generation, acquisition hoặc stopping. Chỉ final IDs nằm trong frozen manifest. Sau khi final checkpoint được cố định, code mới materialize sealed final rows và đánh giá đúng một lần.

Sau evaluation, code audit ID overlap giữa selected training rows và final-test. Near-duplicate protection đối với final-test không được dùng trước selection vì làm vậy sẽ mở held-out text; do đó paper phải mô tả chính xác rằng pre-training leakage index bao phủ các non-final roles, còn final-test được audit post hoc theo firewall.

---

## 5. Student model và NLI scoring

Student mặc định là `Qwen/Qwen3-1.7B`. Cùng một formatted chat prompt được dùng cho inference, difficulty scoring, embedding extraction và training-token construction.

Với mỗi input \(x\), model không generate nhãn tự do. Code nối lần lượt ba candidate strings và tính mean conditional log-likelihood trên các token của candidate:

\[
s_y(x)=\frac{1}{|T_y|}\sum_{k\in T_y}
\log P_\theta(t_k\mid \operatorname{prompt}(x),t_{<k}).
\]

Mean theo token làm giảm thiên lệch do candidate label có độ dài token khác nhau. Phân phối ba nhãn dùng trong acquisition là:

\[
p_y(x)=\frac{\exp s_y(x)}
{\sum_{c\in\mathcal Y}\exp s_c(x)},
\qquad
\mathcal Y=\{E,N,C\}.
\]

Prediction là \(\arg\max_y s_y(x)\). Đây là probability được chuẩn hóa trên đúng ba verbalized candidates, không phải xác suất calibrated tuyệt đối của toàn vocabulary.

---

## 6. Discovery: từ lỗi quan sát đến hypothesis có thể kiểm chứng

### 6.1 Critic và vai trò của gold label

Discovery tổng thể **có dùng gold label** để biết student đúng hay sai. Tuy nhiên teacher critic được thiết kế gold-blind: mỗi record critic nhận premise, hypothesis, student prediction và optional model confidence, nhưng không nhận gold label. Critic trả một error probability \(e_i\in[0,1]\).

Sau critic scoring, pipeline mới dùng gold correctness đã có trong benchmark để tách error pool và correct-contrast pool. Vì vậy không được viết rằng “toàn bộ discovery là gold-free”; phát biểu đúng là “critic call không thấy gold label”.

Ở seed 42, base student sai 1,343 và đúng 2,407 trên 3,750 discovery examples.

### 6.2 Weighted evidence sampling

Critic score được đổi thành sampling weight qua logistic transform:

\[
w_i=\sigma\bigl(\beta(e_i-\alpha)\bigr),
\qquad \alpha=0.5,\;\beta=10.
\]

Exponential-race key được tính như sau:

\[
k_i=\frac{-\log U_i}{w_i},\qquad U_i\sim\operatorname{Uniform}(0,1),
\]

và lấy các key nhỏ nhất. Đây là weighted sampling without replacement; không nên mô tả sai thành “sample theo \(\exp(e_i)\)”. Diversity bucket được dùng để tránh evidence bị thống trị bởi một pattern gần như giống nhau.

Teacher nhận tối đa:

- 80 observed-error examples;
- 20 correct but critic-high-score contrast examples.

Các correct contrasts giúp teacher phân biệt một cơ chế lỗi thật với các false alarm có hình thức bề mặt tương tự.

### 6.3 Hypothesis schema và diversity

Teacher `Qwen3-14B` phải trả đúng năm hypothesis theo schema, gồm tên, mô tả, predicted failure behavior/labels, verification criteria, intervention target và unique support IDs. Candidate hypotheses có token-Jaccard tối đa 0.78. Invalid JSON, thiếu field hoặc duplicate support IDs được retry; duplicate IDs có thể được chuẩn hóa khi an toàn và phải được log.

Kết quả discovery là đề xuất cơ chế có thể bị falsify, chưa phải bằng chứng rằng cơ chế đúng. Verification stage chịu trách nhiệm kiểm tra bước đó.

## 7. Controlled verification và Hoeffding states

### 7.1 Cấu trúc một probe set

Mỗi verification set gồm ba premise-hypothesis pairs khác nhau:

1. **Target**: hiện thực hóa cơ chế được hypothesis dự đoán. Gold binary HANS label và predicted wrong label được khai báo trước khi student inference.
2. **Counterfactual control**: thay đổi quan hệ quyết định sao cho exact MNLI label được validator xác nhận.
3. **Invariant control**: thay đổi chi tiết không liên quan, exact MNLI label phải được giữ theo thiết kế.

Ba pair phải pairwise-distinct. Probes là dữ liệu verification-only và không bao giờ được đưa vào training pool.

Target-failure variable của protocol v2/v5 là:

\[
Z_{h,i}=
\mathbf 1\{\widehat y_{h,i}^{\mathrm{target}}
=y_{h}^{\mathrm{wrong,predeclared}}\}.
\]

Nó đo việc student có lặp lại đúng kiểu sai được hypothesis dự đoán hay không. Counterfactual/invariant correctness và strict-joint outcome vẫn được tính để diagnostic, nhưng không gate target-only certification.

Strict-joint diagnostic là:

\[
J_{h,i}=Z_{h,i}\,
\mathbf 1\{\widehat y_{h,i}^{cf}=y_{h,i}^{cf}\}\,
\mathbf 1\{\widehat y_{h,i}^{inv}=y_{h,i}^{inv}\}.
\]

Việc bỏ \(J\) khỏi certification giải quyết một mismatch: controls được tạo để kiểm tra specificity và cung cấp cấu trúc training, nhưng lỗi label/QC hoặc độ khó của control không nên phủ nhận trực tiếp bằng chứng rằng target failure có tồn tại.

### 7.2 Generation, QC và retry

Teacher sinh probe theo structured JSON. Một reviewer call riêng kiểm tra labels và cấu trúc trước khi student outcomes được tính. Reviewer dùng cùng checkpoint teacher `Qwen3-14B`, nên “independent reviewer” ở đây nghĩa là separate call/role, **không phải một model độc lập**.

Các safeguard chính:

- tối đa 24 context records mỗi hypothesis;
- generation tối đa 5 attempts/batch;
- nếu multi-set response vẫn invalid, batch được giảm dần tới một set;
- validator tối đa 5 attempts/batch và recursively split multi-set batch;
- schema bắt buộc trả đúng một decision cho mọi supplied probe ID;
- lexical anchors kiểm tra scenario adherence;
- exact/near duplicate được lọc với token-Jaccard threshold 0.92;
- invalid singleton chỉ được skip trong allowance đã định; nếu không, stage fail-closed.

Các lỗi thực tế từng gặp gồm malformed standalone JSON, thiếu decision, ba roles dùng pair trùng nhau và near-duplicate với probe trước. Retry correction, batch splitting, checkpointing và exact-ID validation được thêm để tránh mất hàng giờ chạy rồi hỏng toàn stage.

### 7.3 Adaptive looks

Protocol cấu hình tối đa 60 valid sets/hypothesis, minimum 20, với looks:

\[
\mathcal L=\{20,30,40,50,60\}.
\]

Sau mỗi look, student outcomes mới được dùng để quyết định certify, refute hoặc tiếp tục. Teacher call sinh batch tiếp theo không nhận student outcomes trong prompt. Tuy nhiên toàn bộ 60 probes không được frozen trước look đầu: quyết định có sinh tiếp hay không vẫn phụ thuộc lịch sử outcomes. Vì vậy repeated-look factor phải xuất hiện trong confidence correction.

### 7.4 Simultaneous Hoeffding interval

Với \(n_h\) valid target outcomes:

\[
\widehat\nu_h=\frac{1}{n_h}\sum_{i=1}^{n_h}Z_{h,i}.
\]

Code dùng union-adjusted Hoeffding radius:

\[
r_h=\sqrt{\frac{\log(4KT_{\mathrm{eff}}/\delta)}{2n_h}},
\]

với \(K=5\), configured verification rounds bằng 1, 5 adaptive looks nên \(T_{\mathrm{eff}}=5\), và \(\delta=0.05\). Bounds dùng cho decision là:

\[
L_h=\widehat\nu_h-r_h,
\qquad
U_h=\min(1,\widehat\nu_h+r_h).
\]

Lower bound trong artifact có thể âm; code không clip nó về 0 trước certification.

### 7.5 Ba trạng thái quyết định

Với \(\tau_v=0.5\):

\[
\operatorname{state}(h)=
\begin{cases}
\text{certified}, & L_h\ge\tau_v,\\
\text{refuted}, & U_h<\tau_v,\\
\text{unresolved}, & L_h<\tau_v\le U_h.
\end{cases}
\]

Điểm quan trọng nhất của v5 là:

\[
\mathcal H_{\mathrm{retain}}
=\{h:L_h\ge\tau_v\}\cup
\{h:L_h<\tau_v\le U_h\}
=\{h:U_h\ge\tau_v\}.
\]

Chỉ \(U_h<\tau_v\) mới đủ bằng chứng để loại. Retention này khác “accept unresolved”: unresolved vẫn được ghi đúng trạng thái và chỉ được cấp cơ hội generation/acquisition.

### 7.6 Bảo đảm không loại nhầm và giới hạn của nó

Trên simultaneous confidence event

\[
\forall h,t:\quad |\widehat\nu_{h,t}-\nu_h|\le r_{h,t},
\]

nếu một hypothesis thật sự có \(\nu_h\ge\tau_v\), ta có

\[
U_{h,t}\ge\nu_h\ge\tau_v,
\]

nên nó không bị refute. Theo union-adjusted Hoeffding construction, xác suất event thất bại được kiểm soát bởi \(\delta\), với điều kiện outcomes bounded và concentration bound vẫn valid có điều kiện theo lịch sử adaptive.

Đây là phát biểu xác suất có giả định, không phải “không bao giờ bỏ hypothesis tốt” theo nghĩa tuyệt đối. Nó cũng chỉ nói về true target-failure rate theo probe-generating distribution; nó không bảo đảm downstream training gain dương.

---

## 8. Verification seed 42 được tái sử dụng trong v5

Kết quả thực tế từ artifact v2:

| Hypothesis | \(n\) | \(\widehat\nu\) | \(r\) | LCB | UCB | State | CF acc. | INV acc. | Joint rate |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Subject-Object Reversal | 20 | 0.0000 | 0.4359 | -0.4359 | 0.4359 | refuted | 0.9000 | 0.5500 | 0.0000 |
| General-to-Specific | 20 | 1.0000 | 0.4359 | 0.5641 | 1.0000 | certified | 0.5500 | 0.5000 | 0.0500 |
| Contextual Invalidation | 60 | 0.5500 | 0.2517 | 0.2983 | 0.8017 | unresolved | 0.8333 | 0.7500 | 0.3833 |
| Conditional Statement | 30 | 0.8667 | 0.3559 | 0.5107 | 1.0000 | certified | 0.8667 | 0.6000 | 0.3333 |
| Entity Addition | 30 | 0.8667 | 0.3559 | 0.5107 | 1.0000 | certified | 1.0000 | 0.6667 | 0.6000 |

Do đó v5 giữ bốn hypothesis: ba certified cộng `Contextual Invalidation` unresolved; chỉ loại `Subject-Object Reversal` vì UCB 0.4359 thấp hơn 0.5.

Ví dụ `Contextual Invalidation` cho thấy vì sao \(\widehat\nu\) cao chưa đồng nghĩa LCB cao. Dù empirical rate là 0.55, radius ở \(n=60\) vẫn là 0.2517, nên LCB chỉ 0.2983. Certified-only flow loại nó; confidence-safe flow giữ nó vì UCB 0.8017 chưa cho phép bác bỏ.

---

## 9. Fresh synthetic intervention generation

### 9.1 Generation schedule

v5 tái sử dụng discovery/verification nhưng **sinh fresh synthetic pool**. Config đặt:

```yaml
counterfactual_pairs_per_hypothesis: 120
invariant_pairs_per_hypothesis: 120
stage2_generation_limit_examples: 480
stage2_min_accepted_examples: 128
stage2_max_generated_examples: 1600
```

Tên hai config keys có hậu tố `per_hypothesis`, nhưng wrapper Stage 2 coi 120 và 120 là **global pair schedule** rồi chia đều qua retained hypotheses. Với bốn retained hypotheses, round đầu cấp cho mỗi hypothesis:

- 30 counterfactual pairs;
- 30 invariant pairs;
- tổng 60 pairs = 120 examples.

Toàn pool ban đầu là 240 pairs = 480 examples. Cần mô tả behavior này theo code, không diễn giải literal rằng mỗi hypothesis nhận 240 pairs.

### 9.2 Pair semantics

Mỗi pair có đúng hai members và cùng `hypothesis_id`, `pair_id`, `pair_type`:

- counterfactual: `base` và `relation_changed`; hai gold labels phải khác nhau;
- invariant: `base` và `irrelevant_change`; hai gold labels phải giống nhau.

`reasoning_contrast_pairs_per_hypothesis = 0` trong protocol này. Pair rationales, label rationales, targeting rationale và permitted seed IDs là các trường bắt buộc.

Mỗi hypothesis dùng tối đa 24 intervention seed examples. Mỗi teacher call xoay 4 seeds, yêu cầu tối đa 4 pairs và retry tối đa 3 lần. Việc xoay subset nhỏ giảm xu hướng copy nguyên HANS templates và giữ structured response trong output-token limit.

### 9.3 Fail-closed QC

Một separate teacher QC call trả đúng một decision/example với bảy boolean flags:

1. `label_consistent`;
2. `unambiguous`;
3. `grammatical`;
4. `relevant`;
5. `pair_structure_valid`;
6. `semantically_consistent`;
7. `diverse`.

Tất cả phải true. QC xử lý tối đa 8 examples, tức 4 pairs/call, và retry tối đa 3 lần. Teacher không được sửa nhãn trong QC; questionable pair bị reject.

Deterministic validator còn kiểm tra:

- unique IDs và complete pair roles;
- consistent pair metadata, seed IDs và rationale;
- số unique normalized tokens của riêng premise và riêng hypothesis đều trong [3, 80];
- counterfactual labels đổi và invariant labels giữ nguyên;
- exact duplicate giữa hai controlled members bị reject;
- exact/near duplicate với pair đã sinh và non-final leakage references bị reject;
- near-duplicate threshold là token-Jaccard 0.92.

Controlled members được phép gần nhau về lexical content vì bản chất intervention. Vì vậy validator không áp near-duplicate test giữa hai members cùng pair; chỉ exact duplicate trong pair là lỗi.

Leakage index bao phủ `intervention_pool`, `hypothesis_discovery`, `verification`, generated verification probes và `gain_eval`. Raw leakage-reference text/signatures không được gửi trong teacher prompt.

### 9.4 Underfill recovery

Nếu accepted examples chưa đạt 128, generator lên kế hoạch round bổ sung. Gọi deficit là

\[
d=128-N_{accepted},
\]

observed acceptance rate là \(a=N_{accepted}/\max(1,N_{generated})\), và

\[
\widetilde a=\max(0.25,a).
\]

Số examples dự kiến cho round mới là:

\[
m=\max\left(
64,\;2d,\;2\left\lceil\frac{d}{2\widetilde a}\right\rceil,\;4|\mathcal H_{retain}|
\right),
\]

sau đó clip theo remaining generation capacity, làm tròn xuống số chẵn và chia gần đều cho counterfactual/invariant, đồng thời mỗi retained hypothesis phải nhận ít nhất một pair của mỗi loại. Nếu không đạt minimum trước cap 1,600, stage fail thay vì âm thầm chạy pool thiếu.

### 9.5 Pool v5 seed 42 quan sát được

Round 0 đã đủ nên không cần underfill round:

| Hypothesis | Generated | Accepted | Accepted pairs |
|---|---:|---:|---:|
| General-to-Specific | 120 | 116 | 58 |
| Contextual Invalidation | 120 | 120 | 60 |
| Conditional Statement | 120 | 120 | 60 |
| Entity Addition | 120 | 120 | 60 |
| **Total** | **480** | **476** | **238** |

Bốn rejected examples thuộc hai complete pairs của `General-to-Specific`. Validation và selection luôn xử lý ở complete-pair granularity.

## 10. Confidence-safe entropy-novelty selector

### 10.1 Vì sao phải chấm toàn bộ accepted synthetic pool

Selector cần biết difficulty tương đối và khoảng cách biểu diễn giữa mọi candidate pair trước khi chọn subset. Vì vậy base student chạy qua toàn bộ 476 accepted examples hai lượt forward logic:

1. candidate-label scoring để lấy \(p_y\), NLL và entropy;
2. hidden-state extraction để lấy contextual embedding.

Đây là acquisition cost một lần, không lặp sau mỗi selected pair hoặc mỗi LoRA round. Sau khi ranking 128 examples được lập, bốn round chỉ lấy các đoạn liên tiếp của kế hoạch đó.

### 10.2 Gold NLL

Với gold label \(y^*\):

\[
\operatorname{NLL}(x)=-\log\max(p_{y^*}(x),10^{-12}).
\]

NLL cao nghĩa là student gán xác suất thấp cho đáp án đúng. Nó bao phủ cả hai tình huống:

- confidently wrong: gold probability rất nhỏ;
- weakly correct/uncertain: gold probability chưa cao.

Khác entropy, NLL dùng gold label và trực tiếp đo độ khó supervised. Nó không phải training loss đã backprop; ở đây chỉ là acquisition score từ frozen base student.

### 10.3 Normalized entropy

Với ba nhãn:

\[
H(x)=-\frac{1}{\log 3}\sum_{y\in\mathcal Y}
p_y(x)\log\max(p_y(x),10^{-12}).
\]

Do chia cho \(\log 3\), \(H\in[0,1]\). Entropy cao khi student phân vân giữa các nhãn. Một confidently-wrong example có thể entropy thấp nhưng NLL cao, nên kết hợp hai đại lượng tránh bỏ mất loại hard example này.

### 10.4 Robust global NLL normalization

Difficulty chỉ dùng member `base` của mỗi pair. Gọi \(q_{0.1}\) và \(q_{0.9}\) là linear quantiles của target/base NLL trên toàn accepted pool. Code clip và scale:

\[
\widetilde{NLL}_j=
\frac{\operatorname{clip}(NLL_j,q_{0.1},q_{0.9})-q_{0.1}}
{q_{0.9}-q_{0.1}}.
\]

Nếu hai quantiles bằng nhau, mọi normalized NLL được đặt bằng 0.5. Global normalization giữ score có thể so sánh giữa hypotheses; quantile clipping ngăn vài outlier NLL chi phối toàn ranking.

Pair difficulty là:

\[
D_j=
\frac{0.65\,\widetilde{NLL}_j+0.35\,H_j}{0.65+0.35}
=0.65\,\widetilde{NLL}_j+0.35\,H_j.
\]

Chỉ `base` member được dùng ở đây. Nếu average cả base và control, một control rất dễ có thể che một failure-targeting base rất khó.

### 10.5 Contextual embedding

Với cùng formatted NLI prompt, student trả final hidden layer \(h_{x,t}\) cho mỗi non-padding token. Example embedding là mask-aware mean pooling rồi L2 normalize:

\[
\bar e(x)=\frac{\sum_t m_t h_{x,t}}{\sum_t m_t},
\qquad
e(x)=\frac{\bar e(x)}{\|\bar e(x)\|_2}.
\]

Embedding được chuyển về CPU theo batch để không giữ một feature matrix lớn trên GPU. Đây là representation của student cho toàn premise-hypothesis prompt, không phải lexical bag-of-words và cũng không phải một embedding model bên ngoài.

Với hai members \(x_j^{(1)},x_j^{(2)}\), pair embedding là normalized mean:

\[
e_j=\operatorname{norm}_2
\left(\frac{e(x_j^{(1)})+e(x_j^{(2)})}{2}\right).
\]

### 10.6 Novelty

Với tập pair đã chọn \(S\):

\[
N(j\mid S)=
\begin{cases}
1, & S=\varnothing,\\
\operatorname{clip}_{[0,1]}
\left(\dfrac{1-\max_{k\in S}e_j^\top e_k}{2}\right), & \text{otherwise}.
\end{cases}
\]

Selector so candidate với **mọi pair đã chọn**, lấy maximum cosine similarity, rồi thưởng distance tới nearest semantic neighbor. Do vectors normalized, cosine nằm trong [-1,1] và phép chia 2 ánh xạ cosine distance về [0,1]. Novelty được tính lại sau mỗi greedy choice vì \(S\) thay đổi.

Embedding novelty không bảo đảm các câu khác nhau về mọi khía cạnh; nó chỉ chống redundancy theo representation hiện tại của student. Lexical duplicate/leakage đã được validator xử lý trước đó.

### 10.7 Coverage phase

Budget 128 examples tương ứng 64 indivisible pairs. Trước greedy fill, selector chọn đúng một pair cho mỗi retained hypothesis. Với mỗi hypothesis, nó lấy candidate có \(D_j\) lớn nhất; ties được phá deterministic bằng SHA-256 của seed, phase và pair ID.

Đây là minimum coverage constraint:

\[
\forall h\in\mathcal H_{retain}:\quad
|S\cap\mathcal P_h|\ge1.
\]

Nó **không** phải equal quota. Nếu số available pair slots nhỏ hơn số retained hypotheses, selector raise error thay vì phá coverage guarantee.

### 10.8 Greedy fill

Sau coverage, mỗi bước chọn:

\[
j^*=\arg\max_{j\notin S}
\left[D_j+\lambda_N N(j\mid S)\right],
\qquad \lambda_N=0.50.
\]

Tie-breaking vẫn deterministic bằng seeded SHA-256. Cả hai members của pair được chọn cùng nhau. Thuật toán dừng khi đạt 64 pairs.

Không có expected-gain model, hard label quota, hard pair-type quota hoặc equal hypothesis quota trong v5. Phân bổ nổi lên từ observed student difficulty và novelty sau minimum coverage.

Nếu có \(M\) candidate pairs, cần chọn \(P\) pairs và embedding dimension là \(d\), naive greedy novelty có worst-case \(O(MP^2d)\) dot-product work. Với pool 238 pairs và \(P=64\), phần GPU forward thường đắt hơn phần greedy CPU.

### 10.9 Selection seed 42 quan sát được

Selected plan có 64 pairs/128 examples:

| Hypothesis | Selected pairs | Verification state |
|---|---:|---|
| Contextual Invalidation | 1 | unresolved |
| General-to-Specific | 12 | certified |
| Conditional Statement | 50 | certified |
| Entity Addition | 1 | certified |

Pair types: 25 counterfactual và 39 invariant. Label counts trên 128 examples: 23 entailment, 71 neutral, 34 contradiction.

Diagnostics:

- mean selected target NLL: 7.9730;
- mean normalized target entropy: 0.1630;
- mean embedding novelty at selection time: 0.0226;
- global target-NLL q10/q90: xấp xỉ \(1.92\times10^{-13}\) và 7.7683.

Phân bổ 50/12/1/1 chứng minh selector không hard-code cân bằng hypothesis. Nó cũng là một limitation cần report: difficulty term đã tập trung mạnh vào `Conditional Statement`; coverage chỉ ngăn một hypothesis biến mất hoàn toàn, không ngăn concentration. Mean novelty thấp cho thấy accepted pool hoặc student representation chứa nhiều candidates rất gần nhau; novelty weight 0.5 không đủ tạo cân bằng cơ chế ở seed này.

---

## 11. Training protocol

Experiment B chia maximum budget 128 thành bốn stages:

\[
B_{round}=32,\qquad R_{max}=4.
\]

Full 128-example plan được lập một lần từ base checkpoint. Round \(t\) lấy slice \([32(t-1),32t)\). Student không re-score pool sau mỗi adapter update, nên stopping không thể retrospective thay đổi thứ tự acquisition.

LoRA training là cumulative: adapter cuối round trước là starting checkpoint của round sau. Không train bốn adapter độc lập từ base.

v5 command truyền `--ours-objective ce`, do đó objective thực tế là per-example causal-LM cross-entropy:

\[
\mathcal L_{CE}=-\sum_{x\in\mathcal B}
\sum_{t\in \operatorname{label}(x)}
\log P_\theta(t\mid \operatorname{prompt}(x),t_{<t}).
\]

Prompt tokens được mask bằng `-100`; loss chỉ áp trên gold label tokens và EOS. Cả hai members của selected pair đóng góp CE như hai examples có trọng số bằng nhau. Pair structure chỉ được dùng để generation, QC và indivisible selection; v5 CE không có contrastive hoặc consistency loss.

Hyperparameters:

| Tham số | Giá trị |
|---|---:|
| Epochs/round | 1 |
| Microbatch | 4 examples |
| Gradient accumulation | 8 |
| Effective batch | 32 examples |
| Learning rate | \(10^{-5}\) |
| Weight decay | 0 |
| Max grad norm | 1.0 |
| LoRA rank / alpha | 8 / 16 |
| LoRA dropout | 0 |
| Target modules | q/k/v/o, gate/up/down projections |
| Gradient checkpointing | true |
| Max sequence length | 512 |
| Model dtype | float32 |

Với 32 examples, batch 4, accumulation 8 và một epoch, mỗi round ghi một optimizer step.

---

## 12. Adaptive stopping

### 12.1 Paired marginal gain

Sau mỗi round, checkpoint trước và sau được so sánh trên cùng 900 `stopping_eval` IDs. Với example \(i\):

\[
\Delta_{t,i}=
\mathbf1\{\widehat y_{t,i}=y_i\}
-\mathbf1\{\widehat y_{t-1,i}=y_i\}
\in\{-1,0,1\}.
\]

Mean marginal gain và sample variance là:

\[
\bar G_t=\frac1n\sum_i\Delta_{t,i},
\qquad
s_t^2=\frac1{n-1}\sum_i(\Delta_{t,i}-\bar G_t)^2.
\]

Code tính empirical-Bernstein diagnostic với tối đa \(L=4\) looks:

\[
r_t^{EB}=
\sqrt{\frac{2s_t^2\log(3L/\delta)}{n}}
+\frac{6\log(3L/\delta)}{n},
\]

\[
LCB_t=\max(-1,\bar G_t-r_t^{EB}),
\qquad
UCB_t=\min(1,\bar G_t+r_t^{EB}).
\]

Trong v5/v4, interval này là diagnostic; decision rule thực tế là `practical_paired_gain_patience_v1` vì 900 examples làm conservative EB UCB quá rộng để trở thành gate hữu ích.

### 12.2 Practical small-gain look

Một look được coi là `small` khi đồng thời:

\[
\bar G_t\le0.003,
\]

\[
\max_g G^{heuristic}_{t,g}\le0.005,
\qquad
\max_c G^{label}_{t,c}\le0.005,
\]

\[
\min_g G^{heuristic}_{t,g}\ge-0.01,
\qquad
\min_c G^{label}_{t,c}\ge-0.01.
\]

Tức là overall marginal gain nhỏ, không group nào còn cải thiện đáng kể, và cũng không group nào regression quá 1 percentage point. Sau minimum 2 rounds, hai small looks liên tiếp (`patience = 2`) gây early stop.

`maximum_small_gain_ucb = 0.02` vẫn được lưu trong config/result nhưng **không được dùng** bởi practical rule; nó chỉ có hiệu lực với `paired_empirical_bernstein_ucb_v1`. Tương tự, `minimum_optimistic_gain_per_gpu_hour = 0.0` vô hiệu hóa cost-efficiency stop.

Một nuance quan trọng: regression tệ hơn -0.01 làm điều kiện `small` thành false, nên rule tiếp tục thay vì dừng vì harm. Do đó đây là diminishing-return rule, không phải harm-aware safety stop.

Nếu stop, checkpoint sau round hiện tại là final checkpoint. Nếu không stop, protocol dùng checkpoint cuối ở budget 128; không chọn “best round” bằng stopping set.

### 12.3 Trace seed 42

| Round | Cumulative examples | Stopping accuracy | Marginal gain | EB UCB | Action |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0.6322 | — | — | base |
| 1 | 32 | 0.6356 | 0.00333 | 0.05207 | continue |
| 2 | 64 | 0.6422 | 0.00667 | 0.05219 | continue |
| 3 | 96 | 0.6456 | 0.00333 | 0.04624 | continue |
| 4 | 128 | 0.6444 | -0.00111 | 0.03911 | maximum budget reached |

Round 4 là small look đầu tiên, nên patience chưa đủ để stop; đồng thời đã đạt maximum budget. Kết quả là run dùng đủ 128 examples và trạng thái `complete`, không phải `complete_adaptive_stop`.

## 13. Final evaluation và reporting

Sau khi checkpoint được cố định, pipeline thực hiện:

1. `gain_report` trên 2,850 examples đúng một lần;
2. sealed HANS final-test trên 15,000 examples đúng một lần;
3. MNLI validation matched và mismatched;
4. post-training final-test leakage audit;
5. ghi `result.json`, curves, predictions và markdown report.

Metrics HANS gồm overall accuracy, entailment/non-entailment accuracy, accuracy theo heuristic và theo 30 subcases. MNLI báo accuracy riêng trên matched/mismatched.

### 13.1 Kết quả v5 seed 42

| Evaluation role | Examples | Overall accuracy | Non-entailment accuracy |
|---|---:|---:|---:|
| Held-out gain report | 2,850 | 0.6375 | 0.3551 |
| Sealed HANS final | 15,000 | 0.6482 | 0.3648 |

MNLI:

| Split | Examples | Accuracy |
|---|---:|---:|
| matched | 9,815 | 0.5881 |
| mismatched | 9,832 | 0.5894 |

Final leakage audit pass: không có ID overlap hoặc near duplicate được phát hiện. Audit được thực hiện sau selection/training và không dùng để thay đổi training set.

Chỉ có một seed nên các giá trị trên không cho phép ước lượng variance giữa seeds hoặc tuyên bố statistical superiority. Báo cáo hiện tại có thể hiển thị `± 0.0000` vì aggregation trên đúng một run; con số đó không phải confidence interval có ý nghĩa.

---

## 14. Cost accounting

Result lưu cả raw usage và assumed monetary cost. Với seed 42 v5:

| Resource | Measured value |
|---|---:|
| Teacher input tokens | 688,889 |
| Teacher output tokens | 619,450 |
| Generated items | 480 |
| Verified synthetic items | 480 |
| LoRA training runs | 4 |
| LoRA training GPU seconds | 12.0079 |
| Evaluation/acquisition items | 41,573 |
| Evaluation/acquisition GPU seconds | 3,152.7314 |

Với rates trong config, assumed total cost là 6.1913 USD. Đây là **normalized cost model**, không nhất thiết là hóa đơn thực tế của local GPUs hoặc Hugging Face inference. Các rates phải được report cùng số cost; không được nêu 6.1913 USD mà thiếu giả định.

Token usage của v5 artifact có thể bao gồm reused teacher-call provenance trong pool directory. Vì v5 shell script không gọi lại discovery/verification teacher, cần phân biệt:

- **marginal wall-clock/API cost của v5 rerun**: fresh synthetic generation/QC, student acquisition, training và evaluation;
- **full-pipeline accounted cost**: provenance của cả discovery, verification và synthetic stages.

Cost-axis learning curves loại post-hoc final-test evaluation và held-out gain-report evaluation, đúng theo flags trong result.

---

## 15. Flow cũ và flow v5

| Thành phần | Flow cũ | Flow v5 | Lý do thay đổi |
|---|---|---|---|
| Verification metric | ban đầu strict-joint; sau đó target-only | target-only | controls không nên phủ nhận target failure |
| Retention | certified-only: \(LCB\ge\tau_v\) | keep nếu \(UCB\ge\tau_v\) | không loại hypothesis chỉ vì interval rộng |
| Unresolved state | bị loại như non-certified | được giữ, vẫn gắn nhãn unresolved | tách “không đủ bằng chứng” khỏi “bị bác bỏ” |
| Synthetic pool | chỉ hypotheses certified | certified + unresolved | bảo toàn coverage của plausible mechanisms |
| Allocation | hard balance hoặc verification-conditioned primal-dual | minimum coverage + emergent greedy allocation | tránh hard-code hypothesis distribution |
| Controls | từng tham gia certification/specificity weight | diagnostic + pair construction | target difficulty là acquisition signal chính |
| Utility | average/item NLL hoặc stratum utility | base-target gold NLL + entropy | giữ confidently-wrong và uncertain targets |
| Diversity | lexical Jaccard hoặc score cũ | contextual student embedding novelty | giảm semantic redundancy theo student |
| Ranking | có thể phụ thuộc quota/policy phức tạp | plan một lần, deterministic tie-break | reproducible và dễ audit |
| Training objective | từng thử CE và pairwise | CE cho v5 main run | cô lập hiệu quả acquisition; pairwise là ablation |
| Stopping | conservative EB UCB gate ở v3 | practical paired-gain patience ở v4/v5 | UCB trên n=900 quá rộng để dừng hữu ích |
| Gain evaluation | cùng gain set cho adaptation/report | 900 stopping + 2,850 report | giảm adaptive reuse bias |

Protocol v3 từng dùng control specificity:

\[
S_h=LCB_h-rac12[(1-A_h^{cf})+(1-A_h^{inv})].
\]

v5 không dùng đại lượng này trong selector. Controls vẫn được report để người đọc đánh giá specificity, nhưng không hạ sampling mass chỉ vì control accuracy thấp.

---

## 16. Các vấn đề đã gặp và cách giải quyết

### 16.1 `nu_hat` cao nhưng LCB thấp

Nguyên nhân là confidence radius lớn ở sample nhỏ và còn union correction qua hypotheses/looks. Giải pháp không phải hạ threshold hoặc bỏ Hoeffding, mà là dùng ba-state decision và chỉ refute bằng UCB. Khi cần certification mạnh hơn, tăng valid probes sẽ giảm radius theo \(O(n^{-1/2})\), nhưng teacher generation/QC cost tăng gần tuyến tính theo số probes.

### 16.2 Certified-only loại hypothesis có tiềm năng

`Contextual Invalidation` có \(\widehat\nu=0.55\) nhưng LCB 0.2983. Flow cũ loại nó; v5 giữ vì UCB 0.8017. Confidence-safe retention giải quyết false-discard risk theo bound, nhưng không khẳng định hypothesis sẽ có training gain.

### 16.3 Controls làm certification quá khắt khe

Strict-joint rate thấp khi target failure tái diễn nhưng một control sai. v5 certify bằng target failure; control accuracies và joint metric là diagnostics. Synthetic generator vẫn dùng counterfactual/invariant structure để tạo supervision có kiểm soát.

### 16.4 Teacher trả invalid JSON

Các lỗi gồm truncated/malformed JSON, thiếu supplied IDs, duplicate IDs và wrong schema. Giải pháp: standalone-JSON parser, correction prompt có error message, bounded retries, smaller batches, recursive split/reduce-to-singleton, atomic checkpoint và persistent response cache.

### 16.5 Near-duplicate probe/data

Teacher có xu hướng lặp template. Giải pháp: rotating seed subset, exact hash, token-Jaccard threshold, explicit avoid context, validation against prior generated items và non-final split roles.

### 16.6 Hard quota làm acquisition thiếu tự nhiên

Equal quota có thể ép chọn easy/low-value samples. v5 chỉ giữ one-pair coverage, sau đó để difficulty+novelty quyết định. Trade-off là allocation có thể quá tập trung, như 50/64 pairs vào một hypothesis ở seed 42.

### 16.7 NLL một mình bỏ qua vùng phân vân; entropy một mình bỏ confidently-wrong

Kết hợp 0.65 NLL + 0.35 entropy. NLL biết gold label; entropy không cần biết correct class và đo độ phẳng distribution. Hai tín hiệu bổ sung chứ không tương đương.

### 16.8 Lexical diversity không phản ánh semantic redundancy

Selector cũ dùng token overlap. v5 dùng contextual embedding của chính student cho novelty. Deterministic validator vẫn giữ lexical duplicate check như hàng rào leakage, còn acquisition novelty hoạt động ở representation space.

### 16.9 Stopping confidence bound quá bảo thủ

Với 900 paired examples, EB UCB vẫn quanh 4–5 percentage points trong run này. v5 giữ interval để diagnostic nhưng dùng predeclared practical patience rule. Đây là engineering compromise; không được mô tả practical rule như một statistical confidence guarantee.

---

## 17. Điều phương pháp bảo đảm và không bảo đảm

### 17.1 Bảo đảm theo implementation

- deterministic split membership và seeded tie-breaking;
- verification probes không training-eligible;
- complete-pair generation, validation và selection;
- refuted hypothesis không xuất hiện trong v5 pool;
- mỗi retained hypothesis có ít nhất một selected pair nếu budget đủ và pool hợp lệ;
- final-test không tham gia adaptive acquisition/stopping;
- final checkpoint được đánh giá một lần trên sealed HANS final set;
- malformed artifacts, incomplete pairs, incompatible manifests và schema violations fail closed.

### 17.2 Bảo đảm có điều kiện

Với bounded target outcomes, appropriate conditional concentration validity và simultaneous event xác suất ít nhất \(1-\delta\), confidence-safe retention không refute hypothesis có \(\nu_h\ge\tau_v\).

### 17.3 Không được tuyên bố

- Không bảo đảm downstream accuracy luôn tăng.
- Không bảo đảm unresolved hypothesis là đúng hoặc hữu ích.
- Không bảo đảm teacher labels/QC hoàn toàn chính xác.
- Không bảo đảm reviewer độc lập về model; generation và QC dùng cùng teacher checkpoint trong separate calls.
- Không chứng minh causal mechanism; hypotheses là operational failure descriptions.
- Không chứng minh entropy hoặc NLL đã calibrated.
- Không bảo đảm contextual novelty tương ứng linguistic/mechanism diversity.
- Không bảo đảm practical stopping là statistically optimal hoặc harm-aware.
- Không được tuyên bố equal data access với random/uncertainty MNLI baselines; đó là cross-source baselines.
- Không được chọn/tune thêm trên seed 42 rồi vẫn gọi seed 42 là confirmatory evidence mà không disclosure.

---

## 18. Pseudocode đúng implementation

```text
INPUT:
    frozen HANS manifest M
    base student f0
    completed seed-42 discovery/verification artifacts
    maximum training budget B = 128
    verification threshold tau = 0.5

1. Load five discovered hypotheses H.
2. For each h in H, read adaptive target-failure outcomes:
       nu_hat_h = mean(Z_h)
       r_h = sqrt(log(4*K*T_eff/delta) / (2*n_h))
       L_h = nu_hat_h - r_h
       U_h = min(1, nu_hat_h + r_h)
       if L_h >= tau: state_h = certified
       else if U_h < tau: state_h = refuted
       else: state_h = unresolved
3. H_keep = {h : state_h in {certified, unresolved}}.
4. For every h in H_keep:
       generate fresh counterfactual and invariant pairs
       run teacher QC and deterministic validation
       reject incomplete, invalid, duplicate, or leaking pairs
5. If accepted examples < 128:
       generate bounded underfill rounds until >= 128 or fail at cap.
6. Run f0 over every accepted example:
       compute three-label probabilities
       NLL_i = -log p_i[gold_i]
       entropy_i = -sum_y p_i[y] log p_i[y] / log 3
       embedding_i = L2(mean_pool(final_hidden(prompt_i)))
7. For every complete pair j:
       target = its base member
       compute robust normalized target NLL
       D_j = 0.65*NLLnorm_j + 0.35*entropy_j
       e_j = L2((e_member1 + e_member2)/2)
8. Coverage:
       select highest-D pair once for every h in H_keep.
9. Greedy fill until 64 pairs:
       novelty(j,S) = clipped (1 - max cosine(e_j,e_k))/2
       select argmax_j [D_j + 0.50*novelty(j,S)]
       use seeded SHA-256 ties
10. Freeze the 128-example ordered plan.
11. For t = 1..4:
       take next 32 examples
       continue LoRA CE training from previous adapter
       evaluate paired marginal gain on fixed 900-example stopping_eval
       compute EB diagnostic and group gains
       apply practical patience rule after minimum round 2
       if stop: break
12. Fix the last checkpoint (not the best stopping checkpoint).
13. Evaluate once on 2,850 gain_report, 15,000 sealed HANS final,
    MNLI matched and mismatched.
14. Audit leakage; write result and report.
```

---

## 19. Cách trình bày trong paper

### 19.1 Method claim phù hợp

Có thể mô tả novelty chính như sau:

> We introduce confidence-safe hypothesis retention, which distinguishes statistically refuted failure mechanisms from those that remain unresolved, and a student-aware complete-pair acquisition rule combining gold-label difficulty, predictive entropy, and contextual representation novelty.

Không nên viết “verification proves the hypothesis” hoặc “the method never removes a good hypothesis”. Câu đúng hơn:

> Under the simultaneous concentration event, a hypothesis whose controlled target-failure rate exceeds the predeclared threshold is not eliminated by the UCB-based refutation rule.

### 19.2 Method subsections đề xuất

1. Problem formulation and data firewall.
2. Failure evidence discovery.
3. Controlled target-failure verification.
4. Confidence-safe hypothesis retention.
5. Controlled synthetic pair generation and QC.
6. Difficulty- and representation-aware pair acquisition.
7. Cumulative adaptation and budget stopping.
8. Evaluation protocol and leakage audit.

### 19.3 Ablations bắt buộc để tách đóng góp

Để chứng minh thành phần nào tạo gain, nên giữ pool/checkpoint/budget cố định và so sánh:

- certified-only vs certified+unresolved retention;
- NLL-only vs entropy-only vs NLL+entropy;
- difficulty-only vs difficulty+embedding novelty;
- no coverage vs one-pair coverage;
- CE vs pairwise objective trên cùng selected IDs;
- contextual novelty vs lexical-Jaccard diversity;
- v5 selector vs uncertainty MNLI, với disclosure cross-source data access.

Vì v5 hiện được phát triển trên seed 42, seed 43/44 hoặc một frozen confirmatory grid phải được chạy sau khi hyperparameters cố định. Nếu tiếp tục sửa dựa trên seed 42, paper phải gọi seed 42 là development seed.

### 19.4 Primary và secondary endpoints

Khuyến nghị:

- primary: HANS final overall và non-entailment accuracy;
- secondary: HANS per-heuristic/subcase, held-out gain report, MNLI matched/mismatched;
- efficiency: measured GPU seconds, teacher tokens, generated/accepted items;
- diagnostic: retention states, selected hypothesis/pair-type/label distributions, NLL/entropy/novelty summaries.

---

## 20. Reproducibility, artifacts và commands

### 20.1 Chạy toàn bộ v5

```bash
conda activate active_lr
cd /home/llm/hainn/hainn/active_learning
POOL_CUDA_DEVICE=0 RUN_CUDA_DEVICE=1 ./run_ours_verified_v5.sh
```

Hai GPU được dùng cho hai stage tuần tự. Để chỉ build pool:

```bash
POOL_CUDA_DEVICE=0 ./run_ours_verified_v5.sh --pool-only
```

Sau khi pool đã có `.complete`, chạy benchmark/report:

```bash
RUN_CUDA_DEVICE=1 ./run_ours_verified_v5.sh --skip-pool
```

Chỉ tạo lại report từ completed result:

```bash
./run_ours_verified_v5.sh --report-only
```

Script từ chối overwrite completed run. Incomplete run directory được move vào `logs/.../partial_runs` trước khi chạy lại benchmark.

### 20.2 Artifacts chính

| Artifact | Path |
|---|---|
| Reused hypotheses | `outputs/stage2_acquisition_ours_verified_v2/seed_42/hypotheses.json` |
| Reused verification result | `outputs/stage2_acquisition_ours_verified_v2/seed_42/verification/results.json` |
| v5 accepted pool | `outputs/stage2_acquisition_ours_verified_v5/seed_42/synthetic/accepted.jsonl` |
| v5 pool provenance | `outputs/stage2_acquisition_ours_verified_v5/seed_42/synthetic/manifest.json` |
| v5 benchmark result | `outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/result.json` |
| Round/stopping trace | `outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/rounds.jsonl` |
| Final report | `reports/stage2_ours_verified_v5_ce/seed_42_budget_128/STAGE2_REPORT.md` |
| Logs | `logs/stage2_ours_verified_v5_ce/seed_42_budget_128/` |

Persistent teacher response cache và per-hypothesis generation/QC checkpoints cho phép resume mà không lặp các call đã hoàn tất. Atomic final writes và `.complete` marker ngăn downstream benchmark đọc pool dang dở.

---

## 21. Source-code map

| Chức năng | File |
|---|---|
| v5 entry point | `run_ours_verified_v5.sh` |
| v5 generation config | `configs/stage2_generation_ours_confidence_safe.yaml` |
| v5 benchmark config | `configs/stage2_benchmark_ours_verified_v5.yaml` |
| Discovery config | `configs/stage2_ours_verified_v2_discovery.yaml` |
| Verification config | `configs/stage2_ours_verified_v2_verification.yaml` |
| Hypothesis generation/sampling | `src/hypotheses/llm_generator.py` |
| Critic error scoring | `src/hypotheses/error_scorer.py` |
| Probe generation/QC | `src/verification/generated_probes.py` |
| Hoeffding calculation | `src/verification/bounds.py` |
| Synthetic orchestration/retention | `scripts/generate_synthetic_interventions.py` |
| Synthetic generation | `src/synthetic/generator.py` |
| Synthetic QC/leakage | `src/synthetic/validator.py` |
| Synthetic schema | `src/synthetic/types.py` |
| Student scoring/embeddings | `src/models/qwen_nli.py` |
| Pair selector | `src/benchmark/selection.py` |
| Stopping subset | `src/benchmark/stopping_subset.py` |
| Benchmark/training/stopping/eval | `scripts/run_stage2_benchmark.py` |
| Suite grid | `scripts/run_stage2_suite.py` |
| Reporting | `scripts/report_stage2_benchmark.py` |

---

## 22. Audit checklist trước khi dùng số liệu trong paper

- [ ] Config snapshot trong run directory khớp v5 config hash/content.
- [ ] Frozen HANS manifest hash đúng giá trị đã khai báo.
- [ ] Verification metric là `target_failure`, controls là diagnostic-only.
- [ ] Với protocol v6, `probe_design=target_only`, `probes_per_set=1`, và
  strict-joint/control diagnostics phải là `not_available` thay vì giá trị 0.
- [ ] Retained set bằng certified cộng unresolved; refuted set bị loại.
- [ ] `accepted.jsonl` chỉ chứa complete pairs và ít nhất 128 examples.
- [ ] Selector strategy là `confidence_safe_entropy_novelty_v1`.
- [ ] Selected budget chẵn và mọi selected pair có đủ hai members.
- [ ] Training objective ghi `per_example_causal_lm_cross_entropy`.
- [ ] Stopping chỉ đọc 900 `stopping_eval`, không đọc `gain_report`/final-test.
- [ ] Final-test evaluations bằng một cho final checkpoint.
- [ ] Leakage audit pass và không dùng audit để thay training/selection.
- [ ] Report rõ seed 42 là development evidence nếu đã tune trên seed này.
- [ ] So sánh với MNLI baselines không claim equal data access.
- [ ] Mọi cost đều kèm rates/assumptions và raw resource usage.

---

## 23. Ghi chú diễn giải kết quả hiện tại

v5 đã đạt mục tiêu kỹ thuật là không loại `Contextual Invalidation` chỉ vì LCB chưa vượt threshold, tạo đủ fresh candidates, chọn 128 examples không shortage và hoàn tất benchmark không leakage. Tuy nhiên selector cuối cùng chỉ cấp một pair cho unresolved hypothesis và tập trung 50/64 pairs vào `Conditional Statement`. Vì vậy run này chứng minh retention/coverage machinery hoạt động, nhưng chưa chứng minh unresolved hypothesis đóng góp đáng kể cho gain.

---

## 24. Protocol v6 closed loop và target-only verification

Protocol v6 thay thế flow one-shot của v5 bằng tối đa bốn vòng, mỗi vòng train
32 examples và tổng budget tối đa vẫn là 128. Sau mỗi checkpoint, discovery set
được chấm lại; tối đa hai mechanism mới được đề xuất khi tổng candidate active
nhỏ hơn năm. Candidate mới được deduplicate với archive; một mechanism đã
resolved được tái kích hoạt nếu discovery phát hiện cùng lỗi quay trở lại.

Verification v6 dùng `probe_design: target_only`. Mỗi probe set chỉ chứa một
target premise-hypothesis pair với exact three-class gold label và một
predeclared wrong binary HANS label. Counterfactual và invariant controls không
được sinh ở verification stage; chúng chỉ được sinh sau verification dưới dạng
complete training pairs. Vì vậy biến kiểm định là

\[
Z_{h,i}=\mathbf 1\{\widehat y_{h,i}^{\mathrm{HANS}}
=y_h^{\mathrm{wrong,predeclared}}\},
\]

trong khi strict-joint statistic và control accuracies được ghi rõ là
`not_available`. Quy tắc trạng thái không đổi: certify nếu
\(LCB_h\ge\tau_v\), refute/resolve nếu \(UCB_h<\tau_v\), và giữ lại nếu khoảng
tin cậy còn cắt threshold. Probe verification luôn có
`training_eligible=false`.

Với retained hypotheses, generator tạo counterfactual/invariant synthetic
training pairs riêng. Student hiện tại chấm toàn bộ eligible pool. Với target
member của pair \(j\):

\[
D_j=0.65R_{\mathrm{NLL}}(j)+0.35R_{H}(j),
\]

trong đó \(R\) là tie-aware percentile rank theo vòng. Greedy score là

\[
S_j=D_j+0.50R_{\mathrm{novelty}}(j)
+0.35(\sqrt{m_h+1}-\sqrt{m_h})
+\frac{0.50}{1+m_h}.
\]

Hai số hạng cuối là soft diminishing-return và new/under-trained hypothesis
bonus. Không có hard per-hypothesis quota. Sau vòng cuối, một fresh target-only
audit được chạy trước khi mở gain report và sealed final test.

Stopping accuracy tăng từ 0.6322 lên đỉnh 0.6456 ở round 3 rồi giảm nhẹ xuống 0.6444 ở round 4; protocol vẫn dùng round 4 vì checkpoint selection là last checkpoint và patience chưa hết. Đây là hành vi đúng theo predeclared flow, đồng thời chỉ ra hai hướng ablation tương lai: harm-aware stopping và best-checkpoint selection phải được thiết kế/freeze trước khi chạy confirmatory seeds, không được chọn hậu nghiệm trên final-test.

Tóm lại, claim mạnh và chính xác nhất của flow mới là: **confidence-safe retention bảo vệ plausible hypotheses khỏi bị loại chỉ vì thiếu power, còn entropy-NLL-contextual-novelty acquisition chọn complete synthetic pairs theo trạng thái thực của student thay vì hard-coded allocation**. Hiệu quả downstream phải được xác nhận bằng multi-seed controlled ablations.

### 24.1. Preserved seed-42 diagnostic run

Run v6 seed 42, configured budget 128, hoàn tất ngày 2026-10-19 được giữ lại
vĩnh viễn như một diagnostic artifact; không được xóa hoặc thay thế âm thầm.
Observed HANS final accuracy là `0.638533`, nhưng row này không phải một
fixed-budget comparison hợp lệ cho intended cumulative v6. Post-run audit cho
thấy round-4 adapter không giữ được expected cumulative LoRA updates từ ba
round trước. Vì vậy observed accuracy phải luôn đi kèm nhãn
`preserved diagnostic / not fixed-budget comparable`.

Điều này không làm mất giá trị của lifecycle trace: protocol đã refute
Subject-Object Reversal, mở candidate slot, discover Extra Context Entailment
Failure và phân bổ selected pairs cân bằng hơn v5. Checkpoint, checksum và ghi
chú diễn giải được lưu tại
`outputs/preserved_checkpoints/ours_verified_v6_closed_loop/seed_42_budget_128_diagnostic_20261019`;
bảng so sánh có v6 nằm tại
`reports/stage2_seed42_comparison_with_v6/STAGE2_COMPARISON.md`.

---

## 25. Protocol v7: một closed loop, hai training branches công bằng

v7 giữ nguyên discovery, target-only verification, confidence-safe retention,
synthetic QC và selector của v6. Thay đổi duy nhất có chủ đích là sửa tính liên
tục của training và tạo một đối chứng one-pass trên **chính xác cùng acquisition
trace**. Closed loop chỉ chạy một lần; do đó hai branch có cùng hypothesis
archive, verification probes, generated pool, thứ tự vòng và tập 128 selected
example IDs.

### 25.1 Stateful continuation branch (cách 1)

Gọi \(A_r\) là LoRA weights và \(O_r\) là toàn bộ AdamW state sau vòng \(r\),
bao gồm first/second moments và step counter. Vòng kế tiếp bắt đầu từ

\[
(A_{r-1},O_{r-1})
\xrightarrow{\;32\text{ fresh selected examples}\;}
(A_r,O_r).
\]

Với batch size 4, gradient accumulation 8 và một epoch, mỗi vòng có 8
microbatches và đúng một optimizer step. Vì vậy audit bắt buộc là

\[
O_0.step=0,\quad O_1.step=1,\quad O_2.step=2,\quad
O_3.step=3,\quad O_4.step=4.
\]

Trước mỗi vòng \(r>1\), serialized parent adapter được so tensor-by-tensor với
adapter vừa load trong memory. Tên tensor PEFT được canonicalize để
`.default.` không tạo false mismatch; shape/key set phải trùng và sai khác trị
tuyệt đối tối đa chỉ được nằm trong tolerance cho dtype casting. Sau training,
adapter vừa save được kiểm tra lại theo cùng quy tắc. AdamW checkpoint còn lưu
ordered trainable-parameter names; bất kỳ thay đổi parameter order nào đều làm
run fail. Do đó lỗi silent reinitialization của v6 không thể tiếp tục tạo một
artifact có vẻ hoàn tất: v7 sẽ fail-fast ở đúng vòng mất continuity.

### 25.2 Fresh consolidation branch (cách 3)

Sau khi closed loop kết thúc và selected IDs đã frozen, v7 load lại checkpoint
gốc, khởi tạo một LoRA mới với seed 42 và train đúng một epoch trên hợp của bốn
batch acquisition:

\[
\mathcal D_{128}=\mathcal B_1\cup\mathcal B_2\cup
\mathcal B_3\cup\mathcal B_4,
\qquad
A_{\mathrm{base}}\xrightarrow{\;\mathcal D_{128}\;}A_{\mathrm{consolidated}}.
\]

128 examples tạo 32 microbatches và đúng 4 AdamW steps. Branch này không điều
khiển discovery/verification; nó chỉ trả lời câu hỏi sạch: *nếu training recipe
one-pass của các baseline cũ được giữ nguyên, tập 128 examples do closed-loop
acquisition chọn có tốt hơn hay không?* Vì vậy consolidation là primary row để
so sánh acquisition với các one-pass baseline cũ. Continuation là secondary row
để đo giá trị của online continual optimization.

Hai branch có cùng số unique training examples (128), cùng một lần exposure mỗi
example, cùng objective CE, cùng LR, cùng effective batch size và cùng tổng bốn
optimizer steps. Chúng khác nhau có chủ đích ở data ordering/schedule: continuation
chia bốn online segments và giữ AdamW moments, còn consolidation shuffle/train
một lần trên toàn bộ frozen set. Verification probes vẫn luôn
`training_eligible=false`.

### 25.3 v7 artifacts và command

```bash
conda activate active_lr
cd /home/llm/hainn/hainn/active_learning
TEACHER_GPU=1 STUDENT_GPU=0 ./run_ours_verified_v7_dual_closed_loop.sh
```

| Artifact | Path |
|---|---|
| Shared acquisition trace | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/rounds.jsonl` |
| Frozen selected examples | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/selected_training_examples.jsonl` |
| Continuation result | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json` |
| Consolidation result | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json` |
| Direct branch report | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/V7_COMPARISON.md` |
| Full seed-42 table | `reports/stage2_seed42_comparison_with_v7/STAGE2_COMPARISON.md` |

### 25.4 Change log v6 → v7

| Thành phần | v6 diagnostic | v7 hiện tại |
|---|---|---|
| Closed-loop controller | Round adapter được truyền theo path nhưng completed artifact không giữ được expected cumulative updates | Round \(r+1\) bắt buộc load adapter của round \(r\), có tensor-level audit trước khi train |
| Optimizer | AdamW được tạo lại mỗi lần gọi training; không có persisted moments/step audit | Lưu và load toàn bộ AdamW state, ordered parameter names và step counter |
| Continuity failure | Có thể chỉ phát hiện sau khi run kết thúc | Fail-fast nếu adapter, parameter order hoặc optimizer step không khớp |
| Online training | Intended cumulative nhưng artifact seed 42 không chứng minh được continuity | 32 fresh examples/round, một optimizer step/round, state chain `0→1→2→3→4` |
| Final comparison | Chỉ có final online checkpoint, không comparable với one-pass baseline do continuity failure | Hai branch trên cùng selected IDs: continuation và fresh one-pass consolidation |
| Acquisition cost | Một closed loop | Vẫn chỉ một closed loop; không lặp teacher generation cho branch thứ hai |
| Primary comparison | Không dùng v6 để xếp hạng | Consolidation là primary apples-to-apples row; continuation là secondary online-training row |
| v6 artifact | Preserved diagnostic | Vẫn được giữ nguyên, không overwrite hoặc xóa khỏi bảng lịch sử |

Những thành phần **không thay đổi** từ v6 gồm: checkpoint-conditioned
discovery, tối đa 1–2 novel hypotheses mỗi refresh, target-only verification,
adaptive Hoeffding certification, confidence-safe retention, synthetic
generation/QC, NLL–entropy difficulty, contextual embedding novelty và soft
diminishing-return selector. Vì vậy v7 cô lập thay đổi training/checkpoint
continuity thay vì đồng thời thay thuật toán acquisition.

### 25.5 Completed run ledger: seed 42, budget 128

Run được launch lúc `2026-10-19 22:52 CEST` bằng GPU mapping
teacher=`physical GPU 1`, student=`physical GPU 0`, sau đó hoàn tất thành công
vào ngày `2026-10-20`. Marker `.complete`, `comparison.json`, hai branch result
và các báo cáo đều đã được tạo.

Kết quả lifecycle và continuity:

| Round | Fresh examples | Cumulative examples | AdamW transition | Parent adapter matched | Saved adapter matched | Stopping accuracy | Stopping non-entailment |
|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 32 | 32 | `0 → 1` | N/A | yes | 0.636667 | 0.331111 |
| 2 | 32 | 64 | `1 → 2` | yes | yes | 0.638889 | 0.340000 |
| 3 | 32 | 96 | `2 → 3` | yes | yes | 0.638889 | 0.340000 |
| 4 | 32 | 128 | `3 → 4` | yes | yes | 0.641111 | 0.344444 |

Ở cả bốn vòng, adapter save audit kiểm tra đủ `392/392` tensors, không có
missing key, extra key hoặc shape mismatch; maximum absolute difference bằng
`0.0`. Round 2--4 đều load đúng parent adapter và đúng AdamW state từ vòng
trước. Vì vậy v7 đã chứng minh được chuỗi stateful continuation

\[
(A_0,O_0)\rightarrow(A_1,O_1)\rightarrow(A_2,O_2)
\rightarrow(A_3,O_3)\rightarrow(A_4,O_4)
\]

thay vì chỉ truyền một adapter path trên danh nghĩa như artifact v6.

Closed loop hoàn tất `4/4` rounds và dùng đủ `128/128` examples. Stop reason là
`maximum_budget_reached_with_remaining_hypotheses`: run dừng vì đã dùng hết
budget, không phải vì mọi hypothesis đã được giải quyết. Trong refresh trước
round 4, hệ thống phát hiện một novel hypothesis
`llm_cb3f3dbe6e3a4fb9`, verify nó ở trạng thái unresolved và vẫn cấp một pair
exploration nhờ soft diminishing-return selector. Final audit còn hai
hypotheses certified và ba hypotheses unresolved; verification probes tiếp tục
có `training_eligible=false`.

Kết quả final của hai branch dùng cùng 128 selected IDs:

| Branch | Optimizer steps | HANS accuracy | HANS non-entailment | MNLI matched | MNLI mismatched | Vai trò |
|---|---:|---:|---:|---:|---:|---|
| Continuation | 4 | **0.650000** | **0.363296** | **0.586347** | **0.589605** | Kết quả online closed-loop chính của stateful training |
| Consolidation | 4 | 0.647200 | 0.355817 | 0.584412 | 0.586554 | Primary apples-to-apples acquisition comparison với one-pass baselines |

Continuation hơn consolidation `0.0028`, tức `0.28` điểm phần trăm HANS. Vì
hai branch dùng đúng cùng selected examples, cùng CE objective, cùng tổng bốn
optimizer steps và cùng một lần exposure cho mỗi example, chênh lệch này phản
ánh training schedule/state continuity chứ không phải khác acquisition pool.

Completion checklist:

- [x] v7 config snapshot được ghi vào output directory.
- [x] Cả bốn round lưu và audit đúng adapter chain.
- [x] AdamW state chain hoàn tất đúng `0 → 1 → 2 → 3 → 4`.
- [x] Final fresh hypothesis audit hoàn tất.
- [x] Continuation branch được đánh giá trên gain report, sealed HANS và MNLI.
- [x] Consolidation được train fresh từ base trên cùng frozen 128 IDs,
  optimizer `0 → 4`.
- [x] Consolidation branch được đánh giá trên đúng các evaluation sets trên.
- [x] `comparison.json`, `V7_COMPARISON.md` và `.complete` đã được tạo.
- [x] Full seed-42 comparison đã được tạo và vẫn chứa preserved v6 diagnostic
  row.

Chỉ hai file `branches/*/result.json` sau `.complete` được dùng làm kết quả v7;
`rounds.jsonl` là audit trail để chứng minh checkpoint continuity và hypothesis
lifecycle.

### 25.6 Continuity incident sau round 1 và resume fix

Run đầu tiên dừng có chủ đích trước khi score/train round 2. Tensor audit phát
hiện adapter round 1 trên disk vẫn có hash hợp lệ, nhưng adapter vừa load trong
student trên logical `cuda:1` lệch tối đa `0.044175`. Vì check xảy ra ngay sau
load và trước probability scoring/training, **không có invalid round-2 update
nào được tạo**. Round-1 adapter, AdamW state và completed round-2
verification/synthetic artifacts vẫn hợp lệ.

Root cause được tái hiện độc lập với cùng checkpoint:

| Load placement | Adapter result |
|---|---|
| logical `cuda:0` | 392 tensors matched, max difference `0.0` |
| logical `cuda:1`, PEFT inferred load device | mismatch, max difference khoảng `0.04417` |
| logical `cuda:1`, adapter loaded explicitly through CPU | 392 tensors matched, max difference `0.0` |

Trong PEFT 0.21.2, `PeftModel.from_pretrained` gọi `infer_device()` khi không
truyền `torch_device`; hàm này chọn logical `cuda:0` dù base student nằm trên
logical `cuda:1`. Với cấu hình này, model có thể giữ LoRA tensors vừa khởi tạo
thay vì serialized values mà không ném hard load error. Fix v7 truyền
`torch_device="cpu"`; `load_state_dict` sau đó copy từng adapter tensor tới
device thật của target module. Cách này cũng an toàn khi model dùng device map.

Resume support được thêm để không lặp teacher cost:

- validate config snapshot và từ chối resume nếu config thay đổi;
- restore 32 selected examples, round-1 adapter và AdamW step 1;
- validate artifact paths, unique IDs, cumulative counts và contiguous rounds;
- reuse completed round-2 verification plus 96 accepted synthetic examples;
- append log thay vì overwrite log cũ;
- tiếp tục từ scoring/selection/training round 2.

Validation sau fix lúc `2026-10-20 08:24 CEST`:

- full test suite: `205 passed`;
- real Qwen/PEFT load trên logical `cuda:1`: `392/392` tensors matched,
  maximum absolute difference `0.0`;
- dry-run round-2 refresh restore: 5 active hypotheses, 5 states và 96
  synthetic examples được đọc lại, không gọi teacher.

Lệnh resume đã được dùng để tiếp tục artifact hợp lệ sau incident:

```bash
TEACHER_GPU=1 STUDENT_GPU=0 ./run_ours_verified_v7_dual_closed_loop.sh --resume
```

`--overwrite` không được dùng cho incident này vì sẽ xóa các artifact hợp lệ và
lặp toàn bộ teacher generation/QC. Run sau resume hiện đã hoàn tất; command trên
chỉ được giữ lại như provenance, không cần chạy lại.
