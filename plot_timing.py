"""
对比 sync 和 async 模式各阶段耗时，找出性能差异来源
用法：
  python bench_async.py  # 先分别跑串行和异步，生成 timing_sync.json / timing_async.json
  python plot_timing.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']


def load(path):
    with open(path) as f:
        return json.load(f)


def summary(log, mode_filter, key):
    vals = [x[key] for x in log if x.get("mode") == mode_filter and key in x]
    if not vals:
        return None
    return {"data": vals, "mean": np.mean(vals), "median": np.median(vals),
            "p95": np.percentile(vals, 95), "n": len(vals)}


def print_stats(label, s):
    if s is None:
        print(f"  {label}: 无数据")
        return
    print(f"  {label}: mean={s['mean']:.3f}ms  median={s['median']:.3f}ms  p95={s['p95']:.3f}ms  (n={s['n']})")


def plot(sync_log, async_log):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Sync vs Async 各阶段耗时对比", fontsize=14)

    def hist(ax, data_list, labels, colors, title, xlabel):
        for d, lbl, c in zip(data_list, labels, colors):
            if d:
                ax.hist(d, bins=60, alpha=0.6, label=lbl, color=c)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("次数")
        ax.legend()

    def line(ax, data_list, labels, colors, title):
        for d, lbl, c in zip(data_list, labels, colors):
            if d:
                ax.plot(d, alpha=0.7, label=lbl, color=c, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("ms")
        ax.legend()

    # ---- decode ----
    sync_gpu_dec  = summary(sync_log,  "sync",  "gpu_ms")
    async_wait_dec = summary(async_log, "async", "wait_ms")

    line(axes[0][0],
         [sync_gpu_dec["data"] if sync_gpu_dec else [],
          async_wait_dec["data"] if async_wait_dec else []],
         ["sync gpu_ms", "async wait_ms"], ["steelblue", "tomato"],
         "Decode: GPU时间 vs Wait时间（折线）")

    hist(axes[1][0],
         [sync_gpu_dec["data"] if sync_gpu_dec else [],
          async_wait_dec["data"] if async_wait_dec else []],
         ["sync gpu_ms", "async wait_ms"], ["steelblue", "tomato"],
         "Decode: GPU时间 vs Wait时间（分布）", "ms")

    # ---- prefill ----
    sync_gpu_pre   = summary(sync_log,  "sync",  "gpu_ms")   # is_prefill=True 的子集
    async_wait_pre = summary(async_log, "async", "wait_ms")

    sync_gpu_pre_d  = [x["gpu_ms"]  for x in sync_log  if x.get("mode") == "sync"  and x.get("is_prefill")]
    async_wait_pre_d = [x["wait_ms"] for x in async_log if x.get("mode") == "async" and x.get("is_prefill")]

    line(axes[0][1],
         [sync_gpu_pre_d, async_wait_pre_d],
         ["sync gpu_ms(prefill)", "async wait_ms(prefill)"], ["cornflowerblue", "salmon"],
         "Prefill: GPU时间 vs Wait时间（折线）")

    hist(axes[1][1],
         [sync_gpu_pre_d, async_wait_pre_d],
         ["sync gpu_ms(prefill)", "async wait_ms(prefill)"], ["cornflowerblue", "salmon"],
         "Prefill: GPU时间 vs Wait时间（分布）", "ms")

    # ---- schedule + run_async ----
    sync_sched  = [x["schedule_ms"] for x in sync_log  if x.get("mode") == "sync"]
    async_sched = [x["schedule_ms"] for x in async_log if x.get("mode") == "async_schedule"]
    async_run   = [x["run_async_ms"] for x in async_log if x.get("mode") == "async_schedule"]

    line(axes[0][2],
         [sync_sched, async_sched, async_run],
         ["sync schedule", "async schedule", "async run_async"],
         ["green", "orange", "purple"],
         "Schedule / run_async 耗时（折线）")

    hist(axes[1][2],
         [sync_sched, async_sched, async_run],
         ["sync schedule", "async schedule", "async run_async"],
         ["green", "orange", "purple"],
         "Schedule / run_async 耗时（分布）", "ms")

    plt.tight_layout()
    plt.savefig("timing_comparison.png", dpi=150)
    print("\n图已保存到 timing_comparison.png")
    plt.show()


if __name__ == "__main__":
    import os

    sync_path  = "timing_sync.json"
    async_path = "timing_async.json"

    if not os.path.exists(sync_path) or not os.path.exists(async_path):
        print(f"请先运行 bench_async.py 生成 {sync_path} 和 {async_path}")
        exit(1)

    sync_log  = load(sync_path)
    async_log = load(async_path)

    print("=" * 55)
    print("统计摘要")
    print("=" * 55)

    print("\n[Decode 阶段]")
    print_stats("sync  gpu_ms ", summary(sync_log,  "sync",  "gpu_ms"))
    print_stats("async wait_ms", summary(async_log, "async", "wait_ms"))

    print("\n[Prefill 阶段]")
    sg = {"data": [x["gpu_ms"]  for x in sync_log  if x.get("mode") == "sync"  and x.get("is_prefill")]}
    ag = {"data": [x["wait_ms"] for x in async_log if x.get("mode") == "async" and x.get("is_prefill")]}
    if sg["data"]: sg.update(mean=np.mean(sg["data"]), median=np.median(sg["data"]), p95=np.percentile(sg["data"],95), n=len(sg["data"]))
    if ag["data"]: ag.update(mean=np.mean(ag["data"]), median=np.median(ag["data"]), p95=np.percentile(ag["data"],95), n=len(ag["data"]))
    print_stats("sync  gpu_ms ", sg if sg["data"] else None)
    print_stats("async wait_ms", ag if ag["data"] else None)

    print("\n[调度耗时]")
    print_stats("sync  schedule_ms ", summary(sync_log,  "sync",           "schedule_ms"))
    print_stats("async schedule_ms ", summary(async_log, "async_schedule",  "schedule_ms"))
    print_stats("async run_async_ms", summary(async_log, "async_schedule",  "run_async_ms"))

    print("\n[Postprocess]")
    print_stats("sync  postprocess_ms", summary(sync_log,  "sync",  "postprocess_ms"))
    print_stats("async postprocess_ms", summary(async_log, "async", "postprocess_ms"))

    plot(sync_log, async_log)
