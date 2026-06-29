# -*- coding: utf-8 -*-
"""
反推 DCF · 预期翻译器 (Reverse DCF Expectations Calculator) — v0.3
------------------------------------------------------------------------
新增：美股 ticker 自动拉取（FMP 免费档）。非美 / 手动模式完整保留。
自动拉取只做【预填】，所有字段仍可手动覆盖。市值单独处理（铁律 #1）。

运行：
    pip install -r requirements.txt
    streamlit run reverse_dcf_app.py
FMP key：在 Streamlit → Settings → Secrets 加一行  FMP_API_KEY = "你的key"
"""

import datetime
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
# 2. FMP 取数（仅美股 / 免费档）
# ============================================================
FMP_BASE = "https://financialmodelingprep.com/api/v3"   # 若 FMP 弃用 v3，改这一行为 stable 路径


def _fmp_get(path, key, **params):
    params["apikey"] = key
    r = requests.get(f"{FMP_BASE}/{path}", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("Error Message"):
        raise RuntimeError(data["Error Message"])
    return data


@st.cache_data(ttl=86400, show_spinner=False)   # 慢变量缓存 24h，省额度
def fetch_fundamentals(ticker, key, years=5):
    cf = _fmp_get(f"cash-flow-statement/{ticker}", key, limit=years)
    isn = _fmp_get(f"income-statement/{ticker}", key, limit=years)
    bs = _fmp_get(f"balance-sheet-statement/{ticker}", key, limit=1)
    prof = _fmp_get(f"profile/{ticker}", key)
    if not cf or not isn or not bs:
        raise RuntimeError("无财报数据——可能是非美标的（免费档仅美股）或 ticker 拼写有误。")
    return cf, isn, bs, prof


def fetch_marketcap(ticker, key):            # 市值【不缓存】——铁律 #1
    return _fmp_get(f"quote/{ticker}", key)


def compute_fcff_series(cf_list, is_list, add_back_interest=True, charge_sbc=True):
    is_by_year = {str(x.get("calendarYear")): x for x in is_list}
    rows = []
    for c in cf_list:
        y = str(c.get("calendarYear"))
        cfo = float(c.get("operatingCashFlow") or 0)
        capex = abs(float(c.get("capitalExpenditure") or 0))
        sbc = float(c.get("stockBasedCompensation") or 0)
        inc = is_by_year.get(y, {})
        rev = float(inc.get("revenue") or 0)
        ibt = float(inc.get("incomeBeforeTax") or 0)
        texp = float(inc.get("incomeTaxExpense") or 0)
        intx = abs(float(inc.get("interestExpense") or 0))
        eff_tax = (texp / ibt) if ibt > 0 else 0.21
        fcff = cfo - capex
        if charge_sbc:
            fcff -= sbc
        if add_back_interest:
            fcff += intx * (1 - eff_tax)
        rows.append(dict(year=y, revenue=rev, cfo=cfo, capex=capex, sbc=sbc,
                         eff_tax=eff_tax, fcff=fcff))
    return rows


def net_debt_from_bs(bs_list):
    b = bs_list[0]
    td = float(b.get("totalDebt") or 0)
    cash = float(b.get("cashAndShortTermInvestments") or b.get("cashAndCashEquivalents") or 0)
    return td - cash, b.get("date"), b.get("reportedCurrency")


def market_cap_from_quote(quote_list, prof_list):
    q = quote_list[0]
    mc = float(q.get("marketCap") or 0)
    ts = q.get("timestamp")
    when = (datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M UTC")) if ts else "n/a"
    cur = (prof_list[0].get("currency") if prof_list else None) or "USD"
    name = prof_list[0].get("companyName") if prof_list else None
    return mc, when, cur, name


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
        "4. **自动拉取仅美股**（FMP 免费档）。非美标的请手动填，并核对市值与财报币种是否一致。"
    )

st.divider()
mode = st.radio("反推方法", [MODE_FCFF, MODE_REV], index=0)

# ---------- 数据来源（FMP 预填层） ----------
st.subheader("数据来源")
use_fmp = st.toggle("从 FMP 自动拉取（仅美股，免费档）", value=False)

if use_fmp:
    try:
        api_key = st.secrets.get("FMP_API_KEY", "")
    except Exception:
        api_key = ""
    if requests is None:
        st.error("缺少 requests 库。requirements.txt 已含，重新部署即可。")
    elif not api_key:
        st.warning("未配置 FMP_API_KEY：Streamlit → Settings → Secrets 加一行 "
                   "`FMP_API_KEY = \"你的key\"`。在此之前仅手动模式可用。")
    else:
        c1, c2 = st.columns([3, 1])
        ticker = c1.text_input("美股 ticker", placeholder="例: NVDA / AMD / MU").strip().upper()
        if c2.button("拉取", use_container_width=True) and ticker:
            try:
                cf, isn, bs, prof = fetch_fundamentals(ticker, api_key)
                quote = fetch_marketcap(ticker, api_key)
                st.session_state["fmp_raw"] = dict(ticker=ticker, cf=cf, isn=isn,
                                                   bs=bs, prof=prof, quote=quote)
            except Exception as e:
                st.session_state.pop("fmp_raw", None)
                st.error(f"拉取失败：{e}")

    if "fmp_raw" in st.session_state:
        raw = st.session_state["fmp_raw"]
        st.markdown(f"**{raw['ticker']}** 已拉取。下面的拆解可调，确认后点「填入」推入输入框。")
        a, b = st.columns(2)
        add_int = a.checkbox("利息加回（CFO→FCFF 口径）", value=True)
        chg_sbc = b.checkbox("SBC 当真实成本扣除", value=True)

        rows = compute_fcff_series(raw["cf"], raw["isn"], add_int, chg_sbc)
        nd, nd_date, nd_cur = net_debt_from_bs(raw["bs"])
        mc, mc_when, mc_cur, name = market_cap_from_quote(raw["quote"], raw["prof"])

        # 数据来源与时点
        st.caption(f"市值 {mc:,.0f} {mc_cur} · 时点 {mc_when}（FMP 快照，发布前请用你的源核对）"
                   f" ｜ 财报 as-of {nd_date} {nd_cur}"
                   + ("　⚠ 市值/财报币种不一致" if mc_cur != nd_cur else ""))

        # FCFF 序列表
        df = pd.DataFrame(rows)[["year", "revenue", "cfo", "capex", "sbc", "eff_tax", "fcff"]]
        df.columns = ["年", "收入", "CFO", "Capex", "SBC", "有效税率", "FCFF"]
        st.dataframe(df.style.format({"收入": "{:,.0f}", "CFO": "{:,.0f}", "Capex": "{:,.0f}",
                                      "SBC": "{:,.0f}", "有效税率": "{:.1%}", "FCFF": "{:,.0f}"}),
                     hide_index=True, use_container_width=True)

        latest = rows[0]["fcff"]
        avg5 = sum(r["fcff"] for r in rows) / len(rows)
        if avg5 != 0 and abs(latest / avg5 - 1) > 0.25:
            st.warning(f"最新年 FCFF 偏离 5 年均值 {(latest/avg5-1)*100:+.0f}%——疑似周期位置，"
                       f"考虑选「5 年均值」当基年。")

        # 基年选择
        opts = [f"{r['year']}（最新）" if i == 0 else r["year"] for i, r in enumerate(rows)]
        opts.append("5 年均值（周期）")
        pick = st.selectbox("基年 FCFF 口径", opts, index=0)
        chosen_fcff = avg5 if pick.startswith("5 年") else rows[opts.index(pick)]["fcff"]
        latest_rev = rows[0]["revenue"]

        if st.button("↧ 填入下方输入框", type="primary"):
            st.session_state["in_mktcap"] = float(mc)
            st.session_state["in_netdebt"] = float(nd)
            st.session_state["in_ccy"] = mc_cur
            if mode == MODE_FCFF:
                st.session_state["in_fcff0"] = float(chosen_fcff)
            else:
                st.session_state["in_rev0"] = float(latest_rev)
                st.session_state["in_curfcff"] = float(latest)   # 当前(最新年)FCFF
            st.success("已填入。下面仍可手动调整。")

st.divider()

# ---------- 输入区 ----------
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("市场报价")
    ccy = st.text_input("币种", key="in_ccy")
    st.number_input(f"市值 ({ccy})", key="in_mktcap", step=10.0, format="%.1f")
    st.number_input(f"净负债 = 总负债 − 现金 ({ccy})", key="in_netdebt", step=10.0, format="%.1f")
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
st.caption("方法论演示，输出为市场预期的反推翻译，非投资建议。自动拉取数据为 FMP 快照，"
           "发布前请按最新财报与个人判断核对。")
