"""BOPTEST 히트펌프 과도응답 수집 결과를 눈으로 검증하는 그림 3장.

모드: 분석(검증 보고서용). 스타일은 viz-expert analysis.mplstyle 을 로드한다.

F1 14시간 개관        명령대로 전환이 들어갔고 응답이 실제로 움직였다
F2 10초 해상도 확대    전환은 표본 1개 안의 계단이고 그 뒤에 tau ~ 2분 빠른 모드가 있다
F3 관측 간격 효과      10초/30초/1시간 격자에서 보이는 현상이 달라진다

CSV 값은 그대로 쓴다. 파생되는 것은 (a) 상대시각, (b) 단위환산 W->kW,
(c) TAU_FAST.json 의 tau 를 고정한 지수곡선 두 개뿐이며 모두 그림에 명시한다.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── 스타일·색: 파일에서 로드한다 (rcParams 수동 세팅 금지) ─────────────────────
# 스타일 자산은 작업 환경(~/.claude)에 있고 이 저장소에 포함되지 않는다.
# 외부에서 받은 사람도 그림을 다시 그릴 수 있도록, 없으면 matplotlib 기본으로 넘어간다.
# 색·선종만 달라지고 데이터와 수치는 동일하다.
ASSETS = Path.home() / ".claude/agents/viz-expert/assets"
if (ASSETS / "analysis.mplstyle").exists():
    plt.style.use(ASSETS / "analysis.mplstyle")
    sys.path.insert(0, str(ASSETS))
    from palette import colors_for, SEMANTIC, LINESTYLE, ALPHA  # noqa: E402
else:
    print("[알림] viz-expert 스타일 자산이 없어 matplotlib 기본 스타일로 그린다.", flush=True)
    plt.rcParams["figure.constrained_layout.use"] = True   # footer() 가 layout engine 을 쓴다
    _CYC = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    def colors_for(keys, **_):
        n = keys if isinstance(keys, int) else len(keys)
        return [_CYC[i % len(_CYC)] for i in range(n)]
    from collections import defaultdict
    SEMANTIC  = defaultdict(lambda: "0.35",
                            {"baseline": "0.45", "annotation": "0.25", "highlight": "C3"})
    LINESTYLE = defaultdict(lambda: "-", {"threshold": ":", "reference": "--"})
    ALPHA     = defaultdict(lambda: 0.3, {"range": 0.12, "grid": 0.3})

RES = Path("/home/minjae/Documents/github/tsfm-peft-method-screen/results/"
           "boptest_data_contract_20260924")
STAMP = datetime.now()

# 계열 색은 이름으로 받는다. 한 그림 안 충돌은 colors_for 가 정리한다.
# 순서를 바꾸지 않는다 — 바꾸면 다시 돌렸을 때 색이 달라진다.
_F1 = colors_for(["supply water", "condenser heat", "return water",
                  "zone air", "heat pump power", "command"])
C_SUP, C_QCON, C_RET, C_ZON, C_PHP, C_CMD = _F1
_F2 = colors_for(["supply water", "condenser heat", "exponential fit"])
C_FIT = _F2[2]
# F3 에서 구분되는 것은 신호가 아니라 관측 간격이므로 색은 격자 이름에 묶는다.
C_10S, C_30S, C_1H = colors_for(["10 s grid", "30 s grid", "1 h grid"])
C_REF = SEMANTIC["baseline"]   # 배경에 깔리는 고해상도 참조곡선

MONO = dict(family="monospace", fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))


def load(name):
    return np.genfromtxt(RES / name, delimiter=",", names=True)


def footer(fig, src):
    """분석 모드 규격: 출처 경로 + 생성 시각. bbox=tight 가 먹지 않게 띠를 예약한다."""
    fig.get_layout_engine().set(rect=(0.0, 0.028, 1.0, 0.972))
    fig.text(0.01, 0.006, f"src: {src}", ha="left", fontsize=8, color="gray")
    fig.text(0.99, 0.006, STAMP.strftime("%Y-%m-%d %H:%M"), ha="right",
             fontsize=8, color="gray")


def fit_fixed_tau(t, y, tau):
    """tau 를 TAU_FAST.json 값으로 고정하고 y_inf, 진폭만 최소제곱으로 맞춘다.

    tau 를 다시 추정하면 그림의 곡선과 주석의 숫자가 다른 물건이 된다.
    """
    basis = np.exp(-t / tau)
    A = np.vstack([np.ones_like(basis), basis]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else np.nan
    return coef, r2


# ══ F1 ════════════════════════════════════════════════════════════════════════
def make_f1():
    d = load("transient_response.csv")
    t = (d["time"] - d["time"][0]) / 3600.0
    u = d["oveHeaPumY_u"]
    trans = [2.0, 6.0, 8.0, 12.0]
    on_spans = [(2.0, 6.0), (8.0, 12.0)]
    # 전환 직후 한 표본(30 s) 안의 변화량 — "응답이 실제로 움직였다"의 증거
    idx = [np.argmin(np.abs(t - x)) for x in trans]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 2.3, 1.5]))
    ax_cmd, ax_T, ax_P = axes

    for ax in axes:
        for x0, x1 in on_spans:
            ax.axvspan(x0, x1, color=C_CMD, alpha=ALPHA["range"], lw=0, zorder=0)
        for x in trans:
            ax.axvline(x, color=SEMANTIC["annotation"], ls=LINESTYLE["threshold"],
                       lw=1.2, zorder=1)

    # (a) 명령
    ax_cmd.plot(t, u, drawstyle="steps-post", color=C_CMD, lw=1.6)
    ax_cmd.set_ylim(-0.12, 1.95)
    ax_cmd.set_yticks([0, 1])
    ax_cmd.set_ylabel("Compressor\ncommand (0-1)")
    ax_cmd.annotate("t=0 row is the pre-command warm-up state\n"
                    "(u=0.227, heat pump still running)",
                    xy=(0.0, u[0]), xytext=(2.6, 0.30), fontsize=8.5,
                    color=SEMANTIC["annotation"],
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=SEMANTIC["annotation"]))
    for x in trans:
        ax_cmd.text(x, 1.22, f"{x:.0f} h", ha="center", fontsize=8.5,
                    color=SEMANTIC["annotation"])
    ax_cmd.legend(handles=[Patch(facecolor=C_CMD, alpha=ALPHA["range"],
                                 label="ON commanded"),
                           plt.Line2D([], [], color=SEMANTIC["annotation"],
                                      ls=LINESTYLE["threshold"], lw=1.2,
                                      label="commanded transition")],
                  loc="upper right", ncol=2)

    # (b) 온도 3종
    ax_T.plot(t, d["reaTSup_y"], color=C_SUP, lw=1.3, label="Supply water  reaTSup_y")
    ax_T.plot(t, d["reaTRet_y"], color=C_RET, lw=1.3, label="Return water  reaTRet_y")
    ax_T.plot(t, d["reaTZon_y"], color=C_ZON, lw=1.3, label="Zone air      reaTZon_y")
    ax_T.set_ylabel("Temperature (K)")
    ax_T.set_ylim(292.5, 317.5)
    sec = ax_T.secondary_yaxis("right", functions=(lambda k: k - 273.15,
                                                   lambda c: c + 273.15))
    sec.set_ylabel("(equivalent degC)")
    sec.spines["right"].set_visible(True)
    for i, x in zip(idx, trans):
        dT = d["reaTSup_y"][i + 1] - d["reaTSup_y"][i]
        ax_T.annotate(f"{dT:+.2f} K\nin one 30 s sample", xy=(x + 0.08, 315.0),
                      fontsize=8, color=SEMANTIC["annotation"], ha="left", va="top")
    ax_T.legend(loc="lower right", ncol=3)

    # (c) 전력 / 응축열
    ax_P.plot(t, d["reaPHeaPum_y"] / 1e3, color=C_PHP, lw=1.3,
              label="Heat pump electric power  reaPHeaPum_y")
    ax_P.plot(t, d["reaQHeaPumCon_y"] / 1e3, color=C_QCON, lw=1.3,
              label="Condenser heat  reaQHeaPumCon_y")
    ax_P.set_ylabel("Power / heat (kW)")
    ax_P.set_xlabel("Time since run start (hours)")
    ax_P.set_xlim(0, 14)
    ax_P.set_xticks(np.arange(0, 14.1, 1.0))
    ax_P.set_ylim(-1.2, 17.5)
    ax_P.legend(loc="upper right", ncol=2)

    fig.suptitle(
        "F1  BOPTEST bestest_hydronic_heat_pump - 14 h commanded ON/OFF sequence, "
        "30 s samples (n=1681)\n"
        "commanded OFF 2h - ON 4h - OFF 2h - ON 4h - OFF 2h;  "
        "outdoor air 279.0-281.9 K;  every commanded transition is followed by a "
        "response within the next sample",
        fontsize=11.5)
    out = RES / "F1_overview.png"
    footer(fig, "results/boptest_data_contract_20260924/transient_response.csv")
    fig.savefig(out)
    plt.close(fig)
    return out


# ══ F2 ════════════════════════════════════════════════════════════════════════
def make_f2():
    d = load("fast_probe.csv")
    tau_json = json.loads((RES / "TAU_FAST.json").read_text())["per_signal"]
    t_abs = d["time"] - d["time"][0]
    u = d["oveHeaPumY_u"]
    i_cmd = int(np.where(np.diff(u) > 0.5)[0][0])       # 마지막 OFF 표본
    t_c = t_abs[i_cmd]                                   # 명령이 들어간 시각
    x = t_abs - t_c                                      # 명령 이후 경과 (s)
    i0 = i_cmd + 1                                       # 첫 ON 표본 (+10 s)
    dt = float(t_abs[1] - t_abs[0])
    win = slice(i0, i0 + 120)                            # 적합 구간 = ON 20분

    panels = [
        ("reaTSup_y", 1.0, "Supply water temperature (K)", C_SUP, "K",
         "(a) Supply water  reaTSup_y"),
        ("reaQHeaPumCon_y", 1e-3, "Condenser heat (kW)", C_QCON, "kW",
         "(b) Condenser heat  reaQHeaPumCon_y"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)

    for ax, (sig, scale, ylab, col, unit, title) in zip(axes, panels):
        y = d[sig] * scale
        info = tau_json[sig]
        tau = info["tau_fit_s"]
        coef, r2 = fit_fixed_tau(t_abs[win] - t_abs[i0], y[win], tau)
        tt = np.linspace(0, 620, 400)
        curve = coef[0] + coef[1] * np.exp(-tt / tau)

        ax.axvspan(0, dt, color=SEMANTIC["annotation"], alpha=ALPHA["ci"], lw=0,
                   zorder=0)
        ax.plot(x, y, ls="none", marker="o", ms=4.2, mfc=col, mec="white",
                mew=0.5, color=col, label=f"measured, {dt:.0f} s samples")
        ax.plot(tt + dt, curve, color=C_FIT, lw=1.8,
                label=(r"exponential fit  $y_\infty+(y_0-y_\infty)e^{-t/\tau}$, "
                       rf"$\tau$={tau:.1f} s (TAU_FAST.json)"))

        y_pre, y_post = y[i_cmd], y[i0]
        step = y_post - y_pre
        ax.annotate("", xy=(0.0, y_post), xytext=(0.0, y_pre),
                    arrowprops=dict(arrowstyle="<->", lw=1.6, color=col,
                                    shrinkA=0, shrinkB=0))
        ax.annotate(f"step {step:+.3f} {unit}\nwithin one {dt:.0f} s sample",
                    xy=(0.0, y_pre + 0.5 * step), xytext=(48, y_pre + 0.42 * step),
                    fontsize=9, color=SEMANTIC["annotation"], va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=SEMANTIC["annotation"]))
        # 통계 상자는 데이터가 비어 있는 위쪽 코너에. (a)는 곡선이 우상단을 쓰므로 좌상단,
        # (b)는 곡선이 좌상단을 쓰므로 우상단.
        box_x, box_ha = (0.015, "left") if sig == "reaTSup_y" else (0.985, "right")
        ax.text(box_x, 0.97,
                f"step (1 sample) {step:+10.3f} {unit}\n"
                f"tau_fit          {tau:10.1f} s\n"
                f"fit R2           {r2:10.3f}\n"
                f"tau63            {info['tau63_s']:10.1f} s\n"
                f"20 min residual  {info['slow_part_span_20min'] * scale:+10.3f} {unit}",
                transform=ax.transAxes, ha=box_ha, **MONO)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.legend(loc="lower right")

    axes[0].set_ylim(297.0, 310.9)
    axes[0].text(250, 303.4,
                 "measured stays above the single-mode fit:\n"
                 "a slower mode is present as well (see F1)",
                 fontsize=8.5, color=SEMANTIC["annotation"])
    axes[1].set_ylim(-1.0, 19.5)
    axes[-1].set_xlim(-60, 600)
    axes[-1].set_xticks(np.arange(-60, 601, 60))
    axes[-1].set_xlabel("Time since ON command (s)   |   shaded band = the single "
                        "10 s sample interval that contains the step")

    fig.suptitle(
        "F2  10 s probe, ON transition: a step inside one sample, then a fast mode\n"
        "tau = 126 s (supply water) / 50 s (condenser heat), both from TAU_FAST.json\n"
        "markers = measured samples;  line = exponential fit with tau held fixed, "
        "only the two linear coefficients re-fitted",
        fontsize=11.0)
    out = RES / "F2_fast_transient.png"
    footer(fig, "results/boptest_data_contract_20260924/fast_probe.csv + TAU_FAST.json")
    fig.savefig(out)
    plt.close(fig)
    return out


# ══ F3 ════════════════════════════════════════════════════════════════════════
def make_f3():
    f = load("fast_probe.csv")
    d = load("transient_response.csv")

    tf = f["time"] - f["time"][0]
    i_cmd_f = int(np.where(np.diff(f["oveHeaPumY_u"]) > 0.5)[0][0])
    xf = (tf - tf[i_cmd_f]) / 60.0                  # 분
    yf = f["reaTSup_y"]

    td = d["time"] - d["time"][0]
    i_cmd_d = int(np.where(np.diff(d["oveHeaPumY_u"]) > 0.5)[0][0])   # = 240, t=2 h
    xd = (td - td[i_cmd_d]) / 60.0                  # 분
    yd = d["reaTSup_y"]

    # 1시간 격자: 전환 표본(i_cmd_d+1)에 위상을 맞추고 120 표본마다
    ks = np.array([i_cmd_d + 1 + 120 * k for k in range(-1, 4)])
    xh, yh = xd[ks], yd[ks]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    (a1, b1, c1), (a2, b2, c2) = axes
    for ax in axes.ravel():
        ax.axvline(0, color=SEMANTIC["annotation"], ls=LINESTYLE["threshold"], lw=1.1)

    # ── 1행: 각 격자의 고유 범위 ──────────────────────────────────────────────
    # 두 CSV 모두 첫 행(t=0)은 명령 이전 warm-up 상태다(F1 참조). 확대창은 그 행을
    # 넘긴 지점부터 잡는다 — 값을 고치지 않고 창만 옮긴다.
    a1.plot(xf, yf, ls="none", marker="o", ms=3.4, color=C_10S, mec="white", mew=0.4)
    a1.axvspan(0, 20, color=C_10S, alpha=ALPHA["range"], lw=0, zorder=0)
    XA, XB, XC = (-4.8, 25.5), (-0.6, 4.6), (-1.35, 4.3)   # 분, 시간, 시간
    n_a = int(((xf >= XA[0]) & (xf <= XA[1])).sum())
    n_b = int(((xd / 60.0 >= XB[0]) & (xd / 60.0 <= XB[1])).sum())

    a1.set_xlim(*XA)
    a1.set_xlabel("Time since ON command (minutes)")
    a1.set_title(f"(a) 10 s grid - 30 min window ({n_a} samples shown)",
                 loc="left", fontsize=10.5)

    b1.plot(xd / 60.0, yd, ls="none", marker="o", ms=2.6, color=C_30S,
            mec="none")
    b1.axvspan(0, 4, color=C_30S, alpha=ALPHA["range"], lw=0, zorder=0)
    b1.set_xlim(*XB)
    b1.set_xlabel("Time since ON command (hours)")
    b1.set_title(f"(b) 30 s grid - 5.2 h window ({n_b} samples shown)",
                 loc="left", fontsize=10.5)

    c1.plot(xd / 60.0, yd, color=C_REF, lw=1.0, zorder=1,
            label="30 s samples (same data as panel b)")
    c1.plot(xh / 60.0, yh, color=C_1H, lw=1.6, ls="--", marker="s", ms=7,
            mec="white", mew=0.7, zorder=3,
            label="1 h samples + linear interpolation")
    c1.axvspan(0, 4, color=C_1H, alpha=ALPHA["range"], lw=0, zorder=0)
    c1.set_xlim(*XC)
    c1.set_xlabel("Time since ON command (hours)")
    c1.set_title(f"(c) 1 h grid - {len(ks)} samples over -1 h to +3 h",
                 loc="left", fontsize=10.5)
    c1.legend(loc="lower right")

    for ax in (a1, b1, c1):
        ax.set_ylim(296.8, 317.2)
    a1.set_ylabel("Supply water temperature (K)")
    for ax in (b1, c1):
        ax.set_yticklabels([])

    # ── 2행: 공통 -2 .. +10 분 창 ────────────────────────────────────────────
    W0, W1 = -2.0, 10.0
    mf = (xf >= W0 - 0.2) & (xf <= W1 + 0.2)
    md = (xd >= W0 - 0.2) & (xd <= W1 + 0.2)
    mh = (xh >= W0 - 0.2) & (xh <= W1 + 0.2)

    a2.plot(xf[mf], yf[mf], ls="none", marker="o", ms=5, color=C_10S,
            mec="white", mew=0.5, label="10 s samples")
    a2.legend(loc="lower right")

    b2.plot(xf[mf], yf[mf], color=C_REF, lw=1.0, zorder=1,
            label="10 s probe (separate run)")
    b2.plot(xd[md], yd[md], ls="none", marker="o", ms=6, color=C_30S,
            mec="white", mew=0.5, zorder=3, label="30 s samples")
    b2.text(0.04, 0.40,
            "same water temperatures at t=0 in both runs (297.882 / 301.609 K);\n"
            "they separate after ~1 min - the zone is 0.54 K colder in the 30 s run",
            transform=b2.transAxes, ha="left", va="top", fontsize=8.5,
            color=SEMANTIC["annotation"])
    b2.legend(loc="lower right")

    c2.plot(xd[md], yd[md], color=C_REF, lw=1.0, zorder=1,
            label="30 s samples (reference)")
    c2.plot(xh, yh, color=C_1H, lw=1.6, ls="--", zorder=2,
            label="linear interpolation")
    c2.plot(xh[mh], yh[mh], ls="none", marker="s", ms=9, color=C_1H,
            mec="white", mew=0.7, zorder=3, label="1 h samples in window: 1")
    # 시각 0에서 시간격자 보간이 실제값과 얼마나 벌어지는지
    y_true0, y_int0 = yd[np.argmin(np.abs(xd))], float(np.interp(0.0, xh, yh))
    c2.annotate("", xy=(-0.45, y_int0), xytext=(-0.45, y_true0),
                arrowprops=dict(arrowstyle="<->", lw=1.4, color=C_1H,
                                shrinkA=0, shrinkB=0))
    c2.annotate(f"at t=0 the hourly view is already\n"
                f"{y_int0 - y_true0:.2f} K above the true value",
                xy=(-0.45, 0.5 * (y_true0 + y_int0)), xytext=(1.0, 302.4),
                fontsize=8.5, color=SEMANTIC["annotation"], va="center",
                arrowprops=dict(arrowstyle="->", lw=0.9,
                                color=SEMANTIC["annotation"]))
    c2.legend(loc="lower right")

    for ax in (a2, b2, c2):
        ax.set_xlim(W0, W1)
        ax.set_ylim(296.8, 312.6)
        ax.set_xlabel("Time since ON command (minutes)   [common window]")
    a2.set_ylabel("Supply water temperature (K)")
    for ax in (b2, c2):
        ax.set_yticklabels([])

    # 창 안 표본 수는 세어서 쓴다 (손으로 적으면 창을 바꿀 때 어긋난다)
    tau = json.loads((RES / "TAU_FAST.json").read_text())["per_signal"]["reaTSup_y"]["tau_fit_s"]
    n2a = int(((xf >= W0) & (xf <= W1)).sum())
    n2b = int(((xd >= W0) & (xd <= W1)).sum())
    n2c = int(((xh >= W0) & (xh <= W1)).sum())
    boxes = [
        (a2, f"grid                10 s\n"
             f"samples in window    {n2a:3d}\n"
             f"step visible         yes, +5.483 K\n"
             f"tau=126 s equals    {tau / 10:5.1f} sample steps"),
        (b2, f"grid                30 s\n"
             f"samples in window    {n2b:3d}\n"
             f"step visible         yes, +6.568 K\n"
             f"  = step + 30 s of fast mode\n"
             f"tau=126 s equals    {tau / 30:5.1f} sample steps\n"
             f"cross-run at +30 s:\n"
             f"  10 s 304.481 K, 30 s 304.450 K"),
        (c2, f"grid                 1 h\n"
             f"samples in window    {n2c:3d}, next at +60 min\n"
             f"step visible          no\n"
             f"  6.568 K is spread over the\n"
             f"  whole preceding hour\n"
             f"tau=126 s equals    {tau / 3600:5.3f} sample steps"),
    ]
    for ax, txt in boxes:
        ax.text(0.025, 0.975, txt, transform=ax.transAxes, ha="left", **MONO)
    for ax, txt in [
        (a1, "20 min ON,\nstep + fast mode both visible"),
        (b1, "4 h ON, still rising at the end:\nslow mode not settled"),
        (c1, "only a gentle ramp remains;\nnothing resolves the step"),
    ]:
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color=SEMANTIC["annotation"])

    fig.suptitle(
        "F3  The observation cadence decides what is visible - the same commanded ON "
        "transition of reaTSup_y on a 10 s / 30 s / 1 h grid\n"
        "top row: each grid over its own extent.  bottom row: the same -2 to +10 min "
        "window for all three.\n"
        "1 h grid = transient_response.csv taken every 120th sample, phase locked to "
        "the transition sample.  The 10 s probe had 5 min OFF before ON, the "
        "30 s / 1 h run had 2 h OFF.",
        fontsize=10.5)
    out = RES / "F3_resolution_effect.png"
    footer(fig, "results/boptest_data_contract_20260924/"
                "{fast_probe,transient_response}.csv")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import struct
    for fn in (make_f1, make_f2, make_f3):
        p = fn()
        head = p.read_bytes()[16:24]
        w, h = struct.unpack(">II", head)
        print(f"{p}  {w}x{h} px  {p.stat().st_size / 1024:.0f} KB", flush=True)
