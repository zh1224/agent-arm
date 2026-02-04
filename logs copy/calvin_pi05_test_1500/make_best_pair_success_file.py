import json
import argparse
from collections import defaultdict

def pick_best_pair_for_each_subtask(raw, alpha=1.0, beta=1.0, min_total=5, min_success=2):
    """
    raw: dict like { "prev__subtask": {"success":[...], "fail":[...]}, ... }

    选择规则：
    - 对每个 subtask，收集所有 pair_key (*__subtask)
    - 计算平滑成功率 p_hat = (s+alpha)/(s+f+alpha+beta)
    - 优先在满足 (total>=min_total 且 s>=min_success) 的候选里选 p_hat 最大的
    - 若没有任何候选满足门槛，则退化为：选 total 最大（再按 p_hat）
    """
    by_subtask = defaultdict(list)

    for pair_key, v in raw.items():
        if "__" not in pair_key:
            continue
        prev, subtask = pair_key.split("__", 1)

        s_list = v.get("success", []) or []
        f_list = v.get("fail", []) or []
        s = len(s_list)
        f = len(f_list)
        total = s + f
        p_hat = (s + alpha) / (total + alpha + beta) if total > 0 else 0.0

        by_subtask[subtask].append({
            "pair_key": pair_key,
            "prev": prev,
            "s": s,
            "f": f,
            "total": total,
            "p_hat": float(p_hat),
        })

    best = {}
    report = {}

    for subtask, cands in by_subtask.items():
        # 先过滤掉完全没数据的
        cands = [c for c in cands if c["total"] > 0]
        if not cands:
            continue

        strong = [c for c in cands if (c["total"] >= min_total and c["s"] >= min_success)]

        if strong:
            # 主策略：p_hat 最大；并列时用 total、s 打破平局
            chosen = max(strong, key=lambda x: (x["p_hat"], x["total"], x["s"]))
        else:
            # 兜底：没有足够样本，选 total 最大，其次 p_hat
            chosen = max(cands, key=lambda x: (x["total"], x["p_hat"], x["s"]))

        best[subtask] = chosen["pair_key"]
        report[subtask] = chosen

    return best, report

def build_output(raw, best_map, report):
    out = {}
    for subtask, pair_key in best_map.items():
        v = raw.get(pair_key, {})
        out[subtask] = {
            "best_pair_key": pair_key,
            "success": v.get("success", []) or [],
            "n_success": report[subtask]["s"],
            "n_fail": report[subtask]["f"],
            "p_hat": report[subtask]["p_hat"],
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", type=str, default="./subtask_init_states.json", help="path to subtask_init_states.json")
    ap.add_argument("--out_json", type=str, default="./best.json", help="path to write best-per-subtask json")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    ap.add_argument("--beta", type=float, default=1.0, help="Laplace smoothing beta")
    ap.add_argument("--min_total", type=int, default=3, help="min (success+fail) to be considered reliable")
    ap.add_argument("--min_success", type=int, default=2, help="min success count to be considered reliable")
    ap.add_argument("--print_topk", type=int, default=60, help="print top-k subtasks by p_hat")
    args = ap.parse_args()

    with open(args.in_json, "r") as f:
        raw = json.load(f)

    best_map, report = pick_best_pair_for_each_subtask(
        raw,
        alpha=args.alpha,
        beta=args.beta,
        min_total=args.min_total,
        min_success=args.min_success,
    )

    out = build_output(raw, best_map, report)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    # 打印一些汇总，方便你确认
    items = sorted(report.items(), key=lambda kv: kv[1]["p_hat"], reverse=True)
    print(f"[OK] wrote {len(out)} subtasks to {args.out_json}")
    print("Top candidates by p_hat:")
    for subtask, info in items[: args.print_topk]:
        print(f"  {subtask:35s}  best={info['pair_key']:45s}  "
              f"s={info['s']:4d} f={info['f']:4d} total={info['total']:4d} p_hat={info['p_hat']:.3f}")

if __name__ == "__main__":
    main()
