# -*- coding: utf-8 -*-
"""
反推 DCF · 预期翻译器 (Reverse DCF Expectations Calculator) — 最小骨架 v0.2
------------------------------------------------------------------------
定位：把市场报价(市值)反推成隐含的增长预期。不是估值、不是目标价、不是投资建议。
默认值只是最粗糙的起点，所有输入用户可调。

两种反推模式：
  1) 从 FCFF 出发  —— 适合已盈利 / 正现金流公司
  2) 从收入出发    —— 适合当前 FCFF 为负 / 盈利前公司：让利润率从当前负值
                      沿路径收敛到一个【你假设的】稳态正利润率，早年 FCFF 可为负，
                      某年转正。头条改为隐含【收入】CAGR，并显眼提醒稳态利润率是要害假设。

运行：
    pip install -r requirements.txt
    streamlit run reverse_dcf_app.py

本版本：纯手动输入，不接任何数据 API。
"""

import streamlit as st
import pandas as pd
from scipy.optimize import brentq

# ============================================================
# 1. 引擎 (均已离线验证)
# ============================================================

# ---- 1a. FCFF 端 ----
def dcf_ev(fcff0, cagr, wacc, n, terminal="gordon", g_term=0.025,
           exit_multiple=15.0, base_metric=None):
    pv = 0.0
    for t in range(1, n + 1):
        pv += fcff0 * (1 + cagr) ** t / (1 + wacc) ** t
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
    lo, hi = -0.90, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=1e-7)


# ---- 1b. 收入端 (利润率收敛路径) ----
def margin_path(m0, mT, n, k):
    """线性从 m0 收敛到 mT，第 k 年(含)达到 mT 并保持。k<=n。"""
    return [m0 + (mT - m0) * t / k if t <= k else mT for t in range(1, n + 1)]


def dcf_ev_rev(rev0, cagr, wacc, n, m0, mT, k,
               terminal="gordon", g_term=0.025, exit_multiple=6.0, exit_base="fcff"):
    ms = margin_path(m0, mT, n, k)
    pv, fcffs = 0.0, []
    for i, t in enumerate(range(1, n + 1)):
        rev_t = rev0 * (1 + cagr) ** t
        fcff_t = rev_t * ms[i]
        fcffs.append(fcff_t)
        pv += fcff_t / (1 + wacc) ** t
    rev_n = rev0 * (1 + cagr) ** n
    fcff_n = rev_n * mT                       # k<=n 保证终值年利润率=mT
    if terminal == "gordon":
        tv = fcff_n * (1 + g_term) / (wacc - g_term)
    elif terminal == "exit_fcff":
        tv = fcff_n * exit_multiple
    elif terminal == "exit_rev":              # EV/Sales 退出倍数
        tv = rev_n * exit_multiple
    else:
        raise ValueError("unknown terminal")
    pv_tv = tv / (1 + wacc) ** n
    return pv + pv_tv, pv, pv_tv, fcff_n, rev_n, fcffs


def solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT, k, **kw):
    f = lambda g: dcf_ev_rev(rev0, g, wacc, n, m0, mT, k, **kw)[0] - ev_target
    lo, hi = -0.90, 5.0
    if f(lo) > 0 or f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=1e-7)


def crossover_year(m0, mT, k):
    """FCFF 利润率穿过 0 的年份(近似)。m0>=0 则当前已正。"""
    if m0 >= 0:
        return 0.0
    return k * (-m0) / (mT - m0)


def capm_wacc(rf, erp, beta, kd, tax, wd):
    ke = rf + beta * erp
    return (1 - wd) * ke + wd * kd * (1 - tax), ke


REGION_PRESETS = {
    "美国 (US)": (4.5, 5.5),
    "日本 (JP)": (1.6, 6.0),
    "韩国 (KR)": (3.5, 6.0),
    "自定义":     (None, None),
}

MODE_FCFF = "从 FCFF 出发（已盈利 / 正现金流）"
MODE_REV  = "从收入出发（当前 FCFF 为负 / 盈利前）"

# ============================================================
# 2. 页面
# ============================================================

st.set_page_config(page_title="反推 DCF · 预期翻译器", layout="centered")
st.title("反推 DCF · 预期翻译器")

st.info(
    "**这个工具只做一件事**：把市场报价（市值）反推成它隐含的增长预期。\n\n"
    "它**不是估值、不是目标价、不是买卖建议**。它只告诉你——以今天的价格买入，"
    "市场默认了什么样的增长。\n\n"
    "下面所有默认值都只是**一个最粗糙的起点**。结论完全取决于你喂进去的输入，"
    "请务必按最新财报和你自己的判断逐项调整。"
)

with st.expander("⚠ 先读：方法适用边界", expanded=True):
    st.warning(
        "1. **FCFF 为负的成长股** —— 不要用 FCFF 模式。切到**收入模式**：它让利润率从当前负值"
        "收敛到一个稳态正值再反推。但请注意——那个**稳态利润率是你假设的，也是全模型最任性的输入**。\n\n"
        "2. **强周期股** —— 用周期峰值 FCFF 当基年 = 默认永续停在周期顶。请用周期均值，或显式标注。\n\n"
        "3. **用 exit multiple 当终值** —— 会掩盖资产重置（capex 跑步机），资产重的公司尤其失真。\n\n"
        "另：**市值与 FCFF / 收入必须同币种**。TradingView 截图默认是你所在地区的币种，先核对表头。"
    )

st.divider()

mode = st.radio("反推方法", [MODE_FCFF, MODE_REV], index=0)

# ---- 输入区 ----
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("市场报价（固定输入）")
    ccy = st.text_input("币种", value="USD")
    mktcap = st.number_input(f"市值 (Market Cap, {ccy})", value=1000.0, step=10.0, format="%.1f")
    netdebt = st.number_input(f"净负债 = 总负债 − 现金 ({ccy})", value=0.0, step=10.0, format="%.1f")
    ev_target = mktcap + netdebt
    st.metric("→ 隐含 EV", f"{ev_target:,.1f}")
    src = st.text_input("市值来源 + 日期（铁律 #1）",
                        placeholder="例: 2026-06-27 收盘 · TradingView")

with col_r:
    st.subheader("基本面（用户可调）")
    if mode == MODE_FCFF:
        fcff0 = st.number_input(f"基年 FCFF ({ccy})", value=60.0, step=1.0, format="%.1f",
                                help="周期股请用周期均值或显式标注，不要用峰值。")
        st.caption("折旧 / capex / NWC 变动口径请自行确认，这里只接受最终 FCFF。")
    else:
        rev0 = st.number_input(f"当前收入 ({ccy})", value=10.0, step=1.0, format="%.1f")
        cur_fcff = st.number_input(f"当前 FCFF（可为负，{ccy}）", value=-3.0, step=0.5, format="%.1f")
        m0 = cur_fcff / rev0 if rev0 > 0 else 0.0
        st.metric("当前 FCFF 利润率 m₀", f"{m0*100:.1f}%")
        st.markdown("**↓ 全模型最任性的假设 ↓**")
        mT = st.number_input("稳态 FCFF 利润率 mT (%)", value=25.0, step=1.0, format="%.1f",
                             help="公司成熟后能稳定达到的 FCFF 利润率。结论对这个数极度敏感。") / 100
        k = st.number_input("达到稳态利润率的年数 K（≤N）", value=8, min_value=1, max_value=30, step=1)

# ---- WACC ----
with st.expander("WACC（默认美国口径，可改）", expanded=False):
    region = st.selectbox("市场口径", list(REGION_PRESETS.keys()), index=0)
    rf_d, erp_d = REGION_PRESETS[region]
    direct = st.checkbox("直接输入 WACC（跳过 CAPM）", value=False)
    if direct:
        wacc = st.number_input("WACC (%)", value=9.0, step=0.1, format="%.2f") / 100
        ke = None
    else:
        c1, c2, c3 = st.columns(3)
        rf  = c1.number_input("Rf (%)", value=float(rf_d or 4.5), step=0.1, format="%.2f")
        erp = c2.number_input("ERP (%)", value=float(erp_d or 5.5), step=0.1, format="%.2f")
        beta = c3.number_input("β（Damodaran 行业）", value=1.10, step=0.05, format="%.2f")
        c4, c5, c6 = st.columns(3)
        kd  = c4.number_input("税前 Kd (%)", value=5.0, step=0.1, format="%.2f")
        tax = c5.number_input("税率 (%)", value=21.0, step=0.5, format="%.2f")
        wd  = c6.number_input("债务权重 (%)", value=5.0, step=1.0, format="%.1f")
        wacc, ke = capm_wacc(rf/100, erp/100, beta, kd/100, tax/100, wd/100)
    cwacc, cke = st.columns(2)
    cwacc.metric("WACC", f"{wacc*100:.2f}%")
    if ke is not None:
        cke.metric("Ke", f"{ke*100:.2f}%")

# ---- 期限 & 终值 ----
with st.expander("显性期 & 终值方法", expanded=False):
    n = st.number_input("显性期年数 N", value=10, min_value=1, max_value=30, step=1)
    if mode == MODE_FCFF:
        ml = st.radio("终值方法", ["永续增长 Gordon（默认）", "Exit Multiple"], index=0)
        if ml.startswith("永续"):
            terminal = "gordon"
            g_term = st.number_input("永续 g (%)", value=2.5, step=0.1, format="%.2f") / 100
            exit_multiple, base_metric = 15.0, None
        else:
            g_term = 0.025
            exit_multiple = st.number_input("Exit 倍数 (x)", value=15.0, step=0.5, format="%.1f")
            ap = st.radio("倍数应用于", ["终值年 FCFF", "终值年 EBITDA 等指标"], index=0)
            terminal = "exit_fcff" if ap == "终值年 FCFF" else "exit_metric"
            base_metric = None
            if terminal == "exit_metric":
                base_metric = st.number_input(f"基年 EBITDA / 指标 ({ccy})", value=120.0, step=1.0, format="%.1f")
    else:
        ml = st.radio("终值方法", ["永续增长 Gordon（默认）", "Exit Multiple"], index=0)
        if ml.startswith("永续"):
            terminal = "gordon"
            g_term = st.number_input("永续 g (%)", value=2.5, step=0.1, format="%.2f") / 100
            exit_multiple, exit_base = 6.0, "fcff"
        else:
            g_term = 0.025
            ap = st.radio("倍数应用于", ["终值年 FCFF", "终值年收入 (EV/Sales)"], index=1)
            if ap == "终值年 FCFF":
                terminal, exit_base = "exit_fcff", "fcff"
                exit_multiple = st.number_input("Exit 倍数 (x FCFF)", value=15.0, step=0.5, format="%.1f")
            else:
                terminal, exit_base = "exit_rev", "rev"
                exit_multiple = st.number_input("EV/Sales 倍数 (x)", value=6.0, step=0.5, format="%.1f")

st.divider()

# ============================================================
# 3. 计算 & 输出（头条只给一个数）
# ============================================================

if mode == MODE_FCFF:
    # ---- FCFF 模式 ----
    stop = False
    if fcff0 <= 0:
        st.error("基年 FCFF ≤ 0：FCFF 模式不适用。请在上方切换到「从收入出发」模式。")
        stop = True
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
            term_share = pvt / (pve + pvt)
            st.metric(f"隐含 {n} 年 FCFF CAGR", f"{cagr*100:.1f}%")
            st.markdown(f"> 市场报价隐含：未来 **{n} 年 FCFF 年均增长 {cagr*100:.1f}%**，"
                        f"即 {n} 年后 FCFF 达到今天的 **{(1+cagr)**n:.1f} 倍**。")
            d1, d2, d3 = st.columns(3)
            d1.metric("终值现值占比", f"{term_share*100:.0f}%")
            d2.metric(f"FCFF_{n}", f"{fcff_n:,.1f}")
            d3.metric("WACC", f"{wacc*100:.2f}%")
            if term_share > 0.75:
                st.warning("终值占比 >75%：结论高度依赖终值假设，谨慎解读。")
            if cagr < 0:
                st.info("解出 CAGR 为负：当前价格隐含 FCFF 长期收缩。")
            if src.strip() == "":
                st.caption("提示：市值来源/日期未填。发布前请坐实（铁律 #1）。")
            with st.expander("敏感性（可选）", expanded=False):
                if terminal == "gordon":
                    wg = [wacc-0.01, wacc, wacc+0.01]; gg = [max(g_term-0.005,0), g_term, g_term+0.005]
                    rows = []
                    for w in wg:
                        r = {}
                        for g_ in gg:
                            rr = solve_cagr(ev_target, fcff0, w, n, terminal="gordon", g_term=g_) if w > g_ else None
                            r[f"g={g_*100:.1f}%"] = f"{rr*100:.1f}%" if rr is not None else "—"
                        rows.append(r)
                    st.caption("解出的隐含 FCFF CAGR：行=WACC，列=永续 g")
                    st.dataframe(pd.DataFrame(rows, index=[f"WACC={w*100:.1f}%" for w in wg]))

else:
    # ---- 收入模式 ----
    stop = False
    if rev0 <= 0:
        st.error("当前收入 ≤ 0，无法反推。"); stop = True
    if mT <= 0:
        st.error("稳态利润率 mT ≤ 0：收入模式要求终态盈利。若你认为它永不转正，本模型不适用。"); stop = True
    if k > n:
        st.error(f"达到稳态的年数 K ({k}) 不能大于显性期 N ({n})。"); stop = True
    if terminal == "gordon" and wacc <= g_term:
        st.error(f"WACC ({wacc*100:.2f}%) 必须大于永续 g ({g_term*100:.2f}%)。"); stop = True
    if ev_target <= 0:
        st.error("隐含 EV ≤ 0，无法反推。"); stop = True

    if not stop:
        kw = dict(terminal=terminal, g_term=g_term, exit_multiple=exit_multiple)
        cagr = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT, k, **kw)
        if cagr is None:
            st.error("目标 EV 超出可解范围。请检查输入（含 mT / K / 终值方法）。")
        else:
            _, pve, pvt, fcff_n, rev_n, fcffs = dcf_ev_rev(rev0, cagr, wacc, n, m0, mT, k, **kw)
            term_share = pvt / (pve + pvt) if (pve + pvt) != 0 else float("inf")
            cy = crossover_year(m0, mT, k)

            st.metric(f"隐含 {n} 年【收入】CAGR", f"{cagr*100:.1f}%")
            # 特殊提醒（要害）
            lo = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT+0.05, k, **kw)
            hi = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mT-0.05, k, **kw)
            sens = ""
            if lo is not None and hi is not None:
                sens = (f" 若稳态利润率改为 {(mT+0.05)*100:.0f}% / {(mT-0.05)*100:.0f}%，"
                        f"隐含 CAGR 变为 **{lo*100:.1f}% / {hi*100:.1f}%**。")
            st.warning(
                f"⚠ **收入端反推**：头条是【收入】CAGR，不是 FCFF CAGR。"
                f"结论高度依赖你假设的稳态 FCFF 利润率 mT = {mT*100:.0f}%（全模型最任性的输入）。{sens}"
            )
            st.markdown(
                f"> 市场报价隐含：未来 **{n} 年收入年均增长 {cagr*100:.1f}%**（{n} 年后收入 = 今天的 "
                f"**{(1+cagr)**n:.1f} 倍**）。模型假设 FCFF 利润率在 {k} 年内从 {m0*100:.1f}% 收敛到 {mT*100:.0f}%。"
            )
            st.info(f"按此路径，FCFF 预计第 **{cy:.1f}** 年转正（之前各年为负，已计入折现）。")

            d1, d2, d3 = st.columns(3)
            d1.metric("终值现值占比", f"{term_share*100:.0f}%" if term_share != float("inf") else "n/a")
            d2.metric(f"FCFF_{n}", f"{fcff_n:,.1f}")
            d3.metric("m₀ → mT", f"{m0*100:.0f}% → {mT*100:.0f}%")

            if pve < 0:
                st.warning("显性期现值为负：整个估值由终值支撑——本质是纯叙事押注，谨慎解读。")
            elif term_share > 0.75:
                st.warning("终值占比 >75%：结论高度依赖终值假设，谨慎解读。")
            if src.strip() == "":
                st.caption("提示：市值来源/日期未填。发布前请坐实（铁律 #1）。")

            with st.expander("稳态利润率 mT 敏感性（可选）", expanded=False):
                rows = []
                for mt in [mT-0.10, mT-0.05, mT, mT+0.05, mT+0.10]:
                    if mt <= 0:
                        rows.append({"隐含收入 CAGR": "—"}); continue
                    rr = solve_cagr_rev(ev_target, rev0, wacc, n, m0, mt, k, **kw)
                    rows.append({"隐含收入 CAGR": f"{rr*100:.1f}%" if rr is not None else "—"})
                st.caption("固定市值，改变稳态利润率 mT 时解出的隐含收入 CAGR")
                st.dataframe(pd.DataFrame(rows, index=[f"mT={mt*100:.0f}%" for mt in
                                                       [mT-0.10, mT-0.05, mT, mT+0.05, mT+0.10]]))

st.divider()
st.caption("本工具为方法论演示，输出为市场预期的反推翻译，非投资建议。默认值仅为起点，请按最新财报与个人判断调整。")
