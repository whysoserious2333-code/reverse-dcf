# -*- coding: utf-8 -*-
"""
反推 DCF · 预期翻译器 (Reverse DCF Expectations Calculator) — v0.4
------------------------------------------------------------------------
自动拉取改用 Alpha Vantage 免费档（仅美股）+ 自带 key（bring-your-own-key）。
非美 / 手动模式完整保留。自动拉取只做【预填】，所有字段仍可手动覆盖。

⚠ AV 免费现金流表【不含 SBC 行】。需要 SBC 负担口径时，在面板里手动填 SBC，从基年扣除。
⚠ AV 免费 25 次/天。每只票约 4 次调用 → 约 6 只/天。公开时让用户各自填 key，额度算自己的。

运行：pip install -r requirements.txt ; streamlit run reverse_dcf_app.py
AV 免费 key：alphavantage.co/support/#api-key（秒拿，不要信用卡）
"""

import datetime
import time
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

try:
    import requests
except ImportError:
    requests = None

# ============================================================
# 1. 反推引擎
# ============================================================
def dcf_ev(fcff0, cagr, wacc, n, terminal="gordon", g_term=0.025,
           exit_multiple=15.0, base_metric=None):
    pv = sum(fcff0 * (1 + cagr) ** t / (1 + wacc) ** t for t in range(1, n + 1))
    fcff_n = fcff0 * (1 + cagr) ** n
    if terminal == "gordon":
        tv = fcff_n * (1 + g_term) / (wacc - g_term)
    elif terminal == "exit_fcff":
        tv = fcff_n * exit_multiple
    elif terminal == "exit_metric":
        tv = base_metric * (1 + cagr) ** n * exit_multiple
    else:
        raise ValueError("unknown terminal")
    pv_tv = tv / (1 + wacc) ** n
    return pv + pv_tv, pv, pv_tv, fcff_n, tv


def solve_cagr(ev_target, fcff0, wacc, n, **kw):
    f = lambda g: dcf_ev(fcff0, g, wacc, n, **kw)[0] - ev_target
    if f(-0.90) > 0 or f(5.0) < 0:
        return None
    return brentq(f, -0.90, 5.0, xtol=1e-7)


def margin_path(m0, mT, n, k):
    return [m0 + (mT - m0) * t / k if t <= k else mT for t in range(1, n + 1)]


def dcf_ev_rev(rev0, cagr, wacc, n, m0, mT, k,
               terminal="gordon", g_term=0.025, exit_multiple=6.0):
    ms = margin_path(m0, mT, n, k)
    pv, fcffs = 0.0, []
    for i, t in enumerate(range(1, n + 1)):
        f = rev0 * (1 + cagr) ** t * ms[i]
        fcffs.append(f)
        pv += f / (1 + wacc) ** t
    rev_n = rev0 * (1 + cagr) ** n
    fcff_n = rev_n * mT
    if terminal == "gordon":
        tv = fcff_n * (1 + g_term) / (wacc - g_term)
    elif terminal == "exit_fcff":
        tv = fcff_n * exit_multiple
    elif terminal == "exit_rev":
        tv = rev_n * exit_multiple
    else:
        raise ValueError("unknown terminal")
    pv_tv = tv / (1 + wacc) ** n
    return pv + pv_tv, pv, pv_tv, fcff_n, rev_n, fcffs


def solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT, k, **kw):
    f = lambda g: dcf_ev_rev(rev0, g, wacc, n, m0, mT, k, **kw)[0] - ev_target
    if f(-0.90) > 0 or f(5.0) < 0:
        return None
    return brentq(f, -0.90, 5.0, xtol=1e-7)


def crossover_year(m0, mT, k):
    return 0.0 if m0 >= 0 else k * (-m0) / (mT - m0)


def capm_wacc(rf, erp, beta, kd, tax, wd):
    ke = rf + beta * erp
    return (1 - wd) * ke + wd * kd * (1 - tax), ke


# ============================================================
# 2. Alpha Vantage 取数（免费档 / 仅美股 / 自带 key）
# ============================================================
AV_BASE = "https://www.alphavantage.co/query"


def _av_num(x):
    if x is None:
        return 0.0
    s = str(x).strip()
    if s in ("", "None", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _av_get(function, key, symbol, _tries=2):
    for attempt in range(_tries):
        r = requests.get(AV_BASE, params={"function": function, "symbol": symbol, "apikey": key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        msg = None
        for flag in ("Error Message", "Information", "Note"):  # AV 出错也回 200，信息在 JSON 里
            if isinstance(data, dict) and data.get(flag):
                msg = str(data[flag]); break
        if not msg:
            return data
        low = msg.lower()
        rate = ("per second" in low or "spreading out" in low or
                "thank you for using" in low or "frequency" in low or "per day" in low)
        if rate and attempt < _tries - 1:
            time.sleep(2.0)
            continue
        if rate:
            raise RuntimeError("AV 限速：每秒最多 1 次、每天 25 次。已自动放慢重试仍失败——"
                               "稍等几秒重试，或当日 25 次已用尽（明日再试 / 换 key）。")
        raise RuntimeError(msg[:200])
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_av(ticker, key):
    isn = _av_get("INCOME_STATEMENT", key, ticker).get("annualReports", [])
    time.sleep(1.1)
    bs = _av_get("BALANCE_SHEET", key, ticker).get("annualReports", [])
    time.sleep(1.1)
    cf = _av_get("CASH_FLOW", key, ticker).get("annualReports", [])
    time.sleep(1.1)
    ov = _av_get("OVERVIEW", key, ticker)
    if not isn or not bs or not cf:
        raise RuntimeError("无财报数据——可能是非美标的（免费档仅美股）、ticker 拼写有误，或当日额度已用尽。")
    return cf, isn, bs, ov


def fetch_quote(ticker, key):            # 不缓存——市值要新（铁律 #1）
    gq = _av_get("GLOBAL_QUOTE", key, ticker).get("Global Quote", {})
    return _av_num(gq.get("05. price")), gq.get("07. latest trading day")


def compute_fcff_series_av(cf_annual, is_annual, add_back_interest=True):
    is_by_date = {x.get("fiscalDateEnding"): x for x in is_annual if x.get("fiscalDateEnding")}
    rows = []
    for c in cf_annual[:5]:
        inc = is_by_date.get(c.get("fiscalDateEnding"), {})
        cfo = _av_num(c.get("operatingCashflow"))
        capex = abs(_av_num(c.get("capitalExpenditures")))
        rev = _av_num(inc.get("totalRevenue"))
        ibt = _av_num(inc.get("incomeBeforeTax"))
        texp = _av_num(inc.get("incomeTaxExpense"))
        intx = abs(_av_num(inc.get("interestExpense")))
        eff_tax = (texp / ibt) if ibt > 0 else 0.21
        fcff = cfo - capex
        if add_back_interest:
            fcff += intx * (1 - eff_tax)
        rows.append(dict(year=(c.get("fiscalDateEnding", "") or "")[:4], revenue=rev,
                         cfo=cfo, capex=capex, intx=intx, eff_tax=eff_tax, fcff=fcff))
    return rows


def net_debt_av(bs_annual):
    b = bs_annual[0]
    debt = _av_num(b.get("shortLongTermDebtTotal"))   # AV 已给"有息负债合计"(短期+长期)，非总负债
    if debt == 0:
        debt = (_av_num(b.get("shortTermDebt")) or _av_num(b.get("currentDebt"))) + \
               (_av_num(b.get("longTermDebt")) or _av_num(b.get("longTermDebtNoncurrent")))
    cash = _av_num(b.get("cashAndShortTermInvestments")) or _av_num(b.get("cashAndCashEquivalentsAtCarryingValue"))
    return debt - cash, debt, cash, b.get("fiscalDateEnding"), b.get("reportedCurrency")


def market_cap_av(ov, quote):
    cur = ov.get("Currency") or "USD"
    name = ov.get("Name")
    shares = _av_num(ov.get("SharesOutstanding"))
    price, day = quote if quote else (0.0, None)
    if price > 0 and shares > 0:
        return price * shares, cur, name, f"截至 {day} 收盘（AV 价×股数）"
    return _av_num(ov.get("MarketCapitalization")), cur, name, "AV OVERVIEW 快照（AV 不提供时点）"


REGION_PRESETS = {"美国 (US)": (4.5, 5.5), "日本 (JP)": (1.6, 6.0),
                  "韩国 (KR)": (3.5, 6.0), "自定义": (None, None)}
MODE_FCFF = "从 FCFF 出发（已盈利 / 正现金流）"
MODE_REV = "从收入出发（当前 FCFF 为负 / 盈利前）"
DEFAULTS = {"in_ccy": "USD", "in_mktcap": 1000.0, "in_netdebt": 0.0,
            "in_fcff0": 60.0, "in_rev0": 10.0, "in_curfcff": -3.0}

# ============================================================
# 3. 页面
# ============================================================
st.set_page_config(page_title="反推 DCF · 预期翻译器", layout="centered")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.title("反推 DCF · 预期翻译器")
st.info(
    "**这个工具只做一件事**：把市场报价（市值）反推成它隐含的增长预期。\n\n"
    "它**不是估值、不是目标价、不是买卖建议**。它只告诉你——以今天的价格买入，"
    "市场默认了什么样的增长。\n\n"
    "下面所有默认值（含自动拉取的财报）都只是**一个起点**。结论完全取决于你喂进去的输入，"
    "请务必按最新财报和你自己的判断逐项调整。"
)
with st.expander("⚠ 先读：方法适用边界", expanded=False):
    st.warning(
        "1. **FCFF 为负的成长股** —— 不要用 FCFF 模式，切到收入模式；稳态利润率是你假设的、最任性的输入。\n\n"
        "2. **强周期股** —— 别用峰值 FCFF 当基年。自动拉取会给你 5 年序列，请考虑用周期均值。\n\n"
        "3. **exit multiple 当终值** —— 会掩盖 capex 跑步机，资产重的公司尤其失真。\n\n"
        "4. **自动拉取仅美股**（Alpha Vantage 免费档），且现金流表**不含 SBC**——需要时手动填。"
        "非美标的请手动填，并核对市值与财报币种。"
    )

st.divider()
mode = st.radio("反推方法", [MODE_FCFF, MODE_REV], index=0)

# ---------- 数据来源（Alpha Vantage 预填层） ----------
st.subheader("数据来源")
use_av = st.toggle("从 Alpha Vantage 自动拉取（免费 · 仅美股）", value=False)

if use_av:
    user_key = st.text_input("你的 Alpha Vantage key（仅本次会话使用，不保存）", type="password",
                             help="免费 key：alphavantage.co/support/#api-key，秒拿，25 次/天。")
    try:
        secret_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        secret_key = ""
    api_key = (user_key or "").strip() or secret_key

    if requests is None:
        st.error("缺少 requests 库；重新部署即可。")
    elif not api_key:
        st.info("填入一个免费 Alpha Vantage key 即可自动拉取。没有 key 时用手动模式。")
    else:
        c1, c2 = st.columns([3, 1])
        ticker = c1.text_input("美股 ticker", placeholder="例: NVDA / AMD / MU").strip().upper()
        if c2.button("拉取", use_container_width=True) and ticker:
            try:
                with st.spinner("拉取中（AV 免费档限速，每秒 1 次，约 5–6 秒）…"):
                    cf, isn, bs, ov = fetch_av(ticker, api_key)
                    time.sleep(1.1)
                    quote = fetch_quote(ticker, api_key)
                st.session_state["av_raw"] = dict(ticker=ticker, cf=cf, isn=isn, bs=bs, ov=ov, quote=quote)
            except Exception as e:
                st.session_state.pop("av_raw", None)
                st.error(f"拉取失败：{e}")

    if "av_raw" in st.session_state:
        raw = st.session_state["av_raw"]
        st.markdown(f"**{raw['ticker']}** 已拉取。下面口径可调，确认后点「填入」推入输入框。")
        add_int = st.checkbox("利息加回（CFO→FCFF 口径）", value=True)

        rows = compute_fcff_series_av(raw["cf"], raw["isn"], add_int)
        nd, nd_debt, nd_cash, nd_date, nd_cur = net_debt_av(raw["bs"])
        mc, mc_cur, name, mc_when = market_cap_av(raw["ov"], raw.get("quote"))

        st.caption(f"市值 {mc:,.0f} {mc_cur} · {mc_when} — 发布前请用你的源核对（铁律 #1）"
                   + ("　⚠ 市值/财报币种不一致" if mc_cur != nd_cur else ""))
        st.caption(f"净负债构成：有息负债 {nd_debt:,.0f} − 现金及短投 {nd_cash:,.0f} = "
                   f"{nd:,.0f}（财报 as-of {nd_date} {nd_cur}）")

        df = pd.DataFrame(rows)[["year", "revenue", "cfo", "capex", "intx", "eff_tax", "fcff"]]
        df.columns = ["年", "收入", "CFO", "Capex", "利息", "有效税率", "FCFF"]
        st.dataframe(df.style.format({"收入": "{:,.0f}", "CFO": "{:,.0f}", "Capex": "{:,.0f}",
                                      "利息": "{:,.0f}", "有效税率": "{:.1%}", "FCFF": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("注：AV 现金流表不含 SBC。如需按 SBC 负担口径，在下方手动填 SBC 金额从基年 FCFF 扣除。")

        # 周期告警：正负年并存直接警告；否则看偏离均值
        fcffs = [r["fcff"] for r in rows]
        avg5 = sum(fcffs) / len(fcffs)
        if any(f > 0 for f in fcffs) and any(f < 0 for f in fcffs):
            st.warning("FCFF 在正负之间波动（疑似周期 / 转折期）——基年口径影响极大，慎选。")
        elif abs(avg5) > 1e-9 and abs(fcffs[0] / avg5 - 1) > 0.25:
            st.warning(f"最新年 FCFF 偏离均值 {(fcffs[0]/avg5-1)*100:+.0f}%——疑似周期位置，考虑用均值。")

        opts = [f"{r['year']}（最新）" if i == 0 else r["year"] for i, r in enumerate(rows)]
        opts.append("近 5 年均值（周期）")
        pick = st.selectbox("基年 FCFF 口径", opts, index=0)
        base_fcff = avg5 if pick.startswith("近") else rows[opts.index(pick)]["fcff"]
        sbc_manual = st.number_input("手动 SBC（从基年 FCFF 扣除，可选）", value=0.0, step=1.0, format="%.1f")

        if st.button("↧ 填入下方输入框", type="primary"):
            st.session_state["in_mktcap"] = float(mc)
            st.session_state["in_netdebt"] = float(nd)
            st.session_state["in_ccy"] = mc_cur
            if mode == MODE_FCFF:
                st.session_state["in_fcff0"] = float(base_fcff - sbc_manual)
            else:
                st.session_state["in_rev0"] = float(rows[0]["revenue"])
                st.session_state["in_curfcff"] = float(rows[0]["fcff"] - sbc_manual)
            st.success("已填入。下面仍可手动调整。")

st.divider()

# ---------- 输入区 ----------
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("市场报价")
    ccy = st.text_input("币种", key="in_ccy")
    st.number_input(f"市值 ({ccy})", key="in_mktcap", step=10.0, format="%.1f")
    st.number_input(f"净负债 = 有息负债 − 现金 ({ccy})", key="in_netdebt", step=10.0, format="%.1f")
    ev_target = st.session_state["in_mktcap"] + st.session_state["in_netdebt"]
    st.metric("→ 隐含 EV", f"{ev_target:,.1f}")
    src = st.text_input("市值来源 + 日期（铁律 #1）", placeholder="例: 2026-06-27 收盘 · TradingView")

with col_r:
    st.subheader("基本面（可调）")
    if mode == MODE_FCFF:
        st.number_input(f"基年 FCFF ({ccy})", key="in_fcff0", step=1.0, format="%.1f",
                        help="周期股请用周期均值或显式标注，不要用峰值。")
        fcff0 = st.session_state["in_fcff0"]
    else:
        st.number_input(f"当前收入 ({ccy})", key="in_rev0", step=1.0, format="%.1f")
        st.number_input(f"当前 FCFF（可为负，{ccy}）", key="in_curfcff", step=0.5, format="%.1f")
        rev0 = st.session_state["in_rev0"]
        cur_fcff = st.session_state["in_curfcff"]
        m0 = cur_fcff / rev0 if rev0 > 0 else 0.0
        st.metric("当前 FCFF 利润率 m₀", f"{m0*100:.1f}%")
        st.markdown("**↓ 全模型最任性的假设 ↓**")
        mT = st.number_input("稳态 FCFF 利润率 mT (%)", value=25.0, step=1.0, format="%.1f") / 100
        k = st.number_input("达到稳态的年数 K（≤N）", value=8, min_value=1, max_value=30, step=1)

# ---------- WACC ----------
with st.expander("WACC（默认美国口径，可改）", expanded=False):
    region = st.selectbox("市场口径", list(REGION_PRESETS.keys()), index=0)
    rf_d, erp_d = REGION_PRESETS[region]
    if st.checkbox("直接输入 WACC（跳过 CAPM）", value=False):
        wacc = st.number_input("WACC (%)", value=9.0, step=0.1, format="%.2f") / 100
        ke = None
    else:
        c1, c2, c3 = st.columns(3)
        rf = c1.number_input("Rf (%)", value=float(rf_d or 4.5), step=0.1, format="%.2f")
        erp = c2.number_input("ERP (%)", value=float(erp_d or 5.5), step=0.1, format="%.2f")
        beta = c3.number_input("β（Damodaran 行业）", value=1.10, step=0.05, format="%.2f")
        _av = st.session_state.get("av_raw")
        if _av and _av.get("ov"):
            _avb = _av_num(_av["ov"].get("Beta"))
            if _avb:
                st.caption(f"参考：AV 回归 β = {_avb:.2f}（个股市场 β，仅作 sensitivity；"
                           f"你的口径请填 Damodaran 行业 β，别用回归 β）")
        c4, c5, c6 = st.columns(3)
        kd = c4.number_input("税前 Kd (%)", value=5.0, step=0.1, format="%.2f")
        tax = c5.number_input("税率 (%)", value=21.0, step=0.5, format="%.2f")
        wd = c6.number_input("债务权重 (%)", value=5.0, step=1.0, format="%.1f")
        wacc, ke = capm_wacc(rf/100, erp/100, beta, kd/100, tax/100, wd/100)
    w1, w2 = st.columns(2)
    w1.metric("WACC", f"{wacc*100:.2f}%")
    if ke is not None:
        w2.metric("Ke", f"{ke*100:.2f}%")

# ---------- 期限 & 终值 ----------
with st.expander("显性期 & 终值方法", expanded=False):
    n = st.number_input("显性期年数 N", value=10, min_value=1, max_value=30, step=1)
    ml = st.radio("终值方法", ["永续增长 Gordon（默认）", "Exit Multiple"], index=0)
    if mode == MODE_FCFF:
        if ml.startswith("永续"):
            terminal = "gordon"
            g_term = st.number_input("永续 g (%)", value=2.5, step=0.1, format="%.2f") / 100
            exit_multiple, base_metric = 15.0, None
        else:
            g_term = 0.025
            exit_multiple = st.number_input("Exit 倍数 (x)", value=15.0, step=0.5, format="%.1f")
            ap = st.radio("倍数应用于", ["终值年 FCFF", "终值年 EBITDA 等指标"], index=0)
            terminal = "exit_fcff" if ap == "终值年 FCFF" else "exit_metric"
            base_metric = st.number_input(f"基年 EBITDA / 指标 ({ccy})", value=120.0,
                                          step=1.0, format="%.1f") if terminal == "exit_metric" else None
    else:
        if ml.startswith("永续"):
            terminal = "gordon"
            g_term = st.number_input("永续 g (%)", value=2.5, step=0.1, format="%.2f") / 100
            exit_multiple = 6.0
        else:
            g_term = 0.025
            ap = st.radio("倍数应用于", ["终值年 FCFF", "终值年收入 (EV/Sales)"], index=1)
            if ap == "终值年 FCFF":
                terminal = "exit_fcff"
                exit_multiple = st.number_input("Exit 倍数 (x FCFF)", value=15.0, step=0.5, format="%.1f")
            else:
                terminal = "exit_rev"
                exit_multiple = st.number_input("EV/Sales 倍数 (x)", value=6.0, step=0.5, format="%.1f")

st.divider()

# ============================================================
# 4. 计算 & 输出
# ============================================================
if mode == MODE_FCFF:
    stop = False
    if fcff0 <= 0:
        st.error("基年 FCFF ≤ 0：请切换到「从收入出发」模式。"); stop = True
    if terminal == "gordon" and wacc <= g_term:
        st.error(f"WACC ({wacc*100:.2f}%) 必须大于永续 g ({g_term*100:.2f}%)。"); stop = True
    if ev_target <= 0:
        st.error("隐含 EV ≤ 0，无法反推。"); stop = True
    if not stop:
        kw = dict(terminal=terminal, g_term=g_term, exit_multiple=exit_multiple, base_metric=base_metric)
        cagr = solve_cagr(ev_target, fcff0, wacc, n, **kw)
        if cagr is None:
            st.error("目标 EV 超出可解范围。请检查输入。")
        else:
            _, pve, pvt, fcff_n, _ = dcf_ev(fcff0, cagr, wacc, n, **kw)
            ts = pvt / (pve + pvt)
            st.metric(f"隐含 {n} 年 FCFF CAGR", f"{cagr*100:.1f}%")
            st.markdown(f"> 市场报价隐含：未来 **{n} 年 FCFF 年均增长 {cagr*100:.1f}%**，"
                        f"即 {n} 年后达到今天的 **{(1+cagr)**n:.1f} 倍**。")
            d1, d2, d3 = st.columns(3)
            d1.metric("终值现值占比", f"{ts*100:.0f}%")
            d2.metric(f"FCFF_{n}", f"{fcff_n:,.1f}")
            d3.metric("WACC", f"{wacc*100:.2f}%")
            if ts > 0.75:
                st.warning("终值占比 >75%：结论高度依赖终值假设。")
            if cagr < 0:
                st.info("解出 CAGR 为负：当前价格隐含 FCFF 长期收缩。")
            if src.strip() == "":
                st.caption("提示：市值来源/日期未填（铁律 #1）。")
else:
    stop = False
    if rev0 <= 0:
        st.error("当前收入 ≤ 0，无法反推。"); stop = True
    if mT <= 0:
        st.error("稳态利润率 mT ≤ 0：收入模式要求终态盈利。"); stop = True
    if k > n:
        st.error(f"达稳态年数 K ({k}) 不能大于 N ({n})。"); stop = True
    if terminal == "gordon" and wacc <= g_term:
        st.error(f"WACC ({wacc*100:.2f}%) 必须大于永续 g ({g_term*100:.2f}%)。"); stop = True
    if ev_target <= 0:
        st.error("隐含 EV ≤ 0，无法反推。"); stop = True
    if not stop:
        kw = dict(terminal=terminal, g_term=g_term, exit_multiple=exit_multiple)
        cagr = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT, k, **kw)
        if cagr is None:
            st.error("目标 EV 超出可解范围。请检查 mT / K / 终值方法。")
        else:
            _, pve, pvt, fcff_n, rev_n, _ = dcf_ev_rev(rev0, cagr, wacc, n, m0, mT, k, **kw)
            ts = pvt / (pve + pvt) if (pve + pvt) != 0 else float("inf")
            cy = crossover_year(m0, mT, k)
            st.metric(f"隐含 {n} 年【收入】CAGR", f"{cagr*100:.1f}%")
            lo = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT+0.05, k, **kw)
            hi = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT-0.05, k, **kw)
            sens = (f" 若稳态利润率改为 {(mT+0.05)*100:.0f}% / {(mT-0.05)*100:.0f}%，"
                    f"隐含 CAGR 变为 **{lo*100:.1f}% / {hi*100:.1f}%**。") if (lo and hi) else ""
            st.warning(f"⚠ **收入端反推**：头条是【收入】CAGR，不是 FCFF CAGR。"
                       f"结论高度依赖你假设的稳态利润率 mT = {mT*100:.0f}%（最任性的输入）。{sens}")
            st.markdown(f"> 未来 **{n} 年收入年均增长 {cagr*100:.1f}%**（{n} 年后收入 = 今天的 "
                        f"**{(1+cagr)**n:.1f} 倍**）。利润率在 {k} 年内从 {m0*100:.1f}% 收敛到 {mT*100:.0f}%。")
            st.info(f"按此路径，FCFF 预计第 **{cy:.1f}** 年转正（之前各年为负，已计入折现）。")
            d1, d2, d3 = st.columns(3)
            d1.metric("终值现值占比", f"{ts*100:.0f}%" if ts != float("inf") else "n/a")
            d2.metric(f"FCFF_{n}", f"{fcff_n:,.1f}")
            d3.metric("m₀ → mT", f"{m0*100:.0f}% → {mT*100:.0f}%")
            if pve < 0:
                st.warning("显性期现值为负：整个估值由终值支撑——本质是纯叙事押注。")
            elif ts > 0.75:
                st.warning("终值占比 >75%：结论高度依赖终值假设。")
            if src.strip() == "":
                st.caption("提示：市值来源/日期未填（铁律 #1）。")

st.divider()
st.caption("方法论演示，输出为市场预期的反推翻译，非投资建议。自动拉取数据为 Alpha Vantage 快照，"
           "发布前请按最新财报与个人判断核对。")
