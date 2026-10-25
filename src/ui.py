"""Visual theme and small HTML components, modelled on the APIS 2022 data-story dashboard
(light blue canvas, white rounded cards, numbered section headings, KPI cards, amber notice).
Requires Streamlit >= 1.40 (st.container(key=...) is used to style cards)."""
import re
import streamlit as st

CSS = """
<style>
:root{--blue:#1F5AA6;--navy:#0F1F3D;--ink:#1B2437;--muted:#6B7488;--canvas:#E9EFF8;--tint:#DCE7F7;--amber:#FDF1DC;--line:#E3E8F1;}
[data-testid="stAppViewContainer"]{background:var(--canvas);}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1340px;padding-top:2.2rem;padding-bottom:4rem;}
html,body,[class*="css"]{color:var(--ink);}
h1,h2,h3{color:var(--ink);letter-spacing:-0.01em;}

/* sidebar */
[data-testid="stSidebar"]{background:#fff;border-radius:0 28px 28px 0;box-shadow:0 2px 14px rgba(20,40,80,.06);}
/* sidebar navigation (text list; current page highlighted) */
[class*="st-key-nav_"] button,[class*="st-key-navactive_"] button{background:transparent;border:none;box-shadow:none;width:100%;justify-content:flex-start;text-align:left;padding:.5rem .75rem;border-radius:10px;color:var(--ink);}
[class*="st-key-nav_"] button,[class*="st-key-navactive_"] button{justify-content:flex-start!important;gap:.65rem;}
[class*="st-key-nav_"] button>div,[class*="st-key-navactive_"] button>div{justify-content:flex-start!important;gap:.65rem;width:100%;}
[class*="st-key-nav_"] [data-testid="stMarkdownContainer"],[class*="st-key-navactive_"] [data-testid="stMarkdownContainer"]{width:auto;text-align:left;}
[class*="st-key-nav_"] button p,[class*="st-key-navactive_"] button p{text-align:left;font-size:.95rem;}
[class*="st-key-nav_"] button:hover{background:#EEF2F8;color:var(--blue);}
[class*="st-key-navactive_"] button,[class*="st-key-navactive_"] button:hover{background:var(--tint);color:var(--blue);font-weight:700;}
[data-baseweb="select"]>div{background:#EEF2F8!important;border:none!important;border-radius:12px!important;}
.side-label{font-size:.72rem;letter-spacing:.12em;font-weight:700;color:var(--muted);margin:1.4rem 0 .4rem;}
.brand{display:flex;gap:12px;align-items:center;margin:.6rem 0 .2rem;}
.logo{width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#1F5AA6,#3F8FE0);position:relative;flex:none;}
.logo:before,.logo:after{content:"";position:absolute;width:15px;height:15px;border-radius:3px;}
.logo:before{left:9px;top:9px;background:#2BB673;}.logo:after{left:17px;top:16px;background:#F2B134;opacity:.95;}
.brand b{display:block;font-size:1rem;}.brand span{font-size:.75rem;color:var(--muted);}

/* header card + pills */
.hdr{background:#fff;border-radius:20px;padding:1.3rem 1.6rem;display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;}
.hdr h1{font-size:1.65rem;margin:0;padding:0;line-height:1.2;font-weight:800;}
.hdr p{margin:.25rem 0 0;color:var(--muted);font-size:.98rem;}
.pill{display:inline-block;padding:.45rem .95rem;border-radius:999px;background:#EEF2F8;color:var(--ink);font-size:.82rem;font-weight:600;margin-left:.5rem;}
.pill.dark{background:var(--navy);color:#fff;}

/* callouts */
.ccard{border-radius:20px;padding:1.2rem 1.5rem;height:100%;background:#fff;}
.ccard.tint{background:var(--tint);}
.lbl{font-size:.72rem;letter-spacing:.12em;font-weight:700;color:#5B6578;margin-bottom:.5rem;text-transform:uppercase;}
.ccard p{margin:0;font-size:1.02rem;line-height:1.6;}
.notice{background:var(--amber);border-radius:14px;padding:.95rem 1.3rem;font-size:.95rem;line-height:1.55;margin:1rem 0 .4rem;}
.notice.blue{background:var(--tint);}

/* numbered section title */
.sec{display:flex;align-items:center;gap:.7rem;margin:2.2rem 0 .9rem;}
.sec .n{width:30px;height:30px;border-radius:50%;background:var(--blue);color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center;font-size:.9rem;flex:none;}
.sec h2{margin:0;padding:0;font-size:1.35rem;font-weight:800;}

/* KPI cards */
.kpi{background:#fff;border-radius:20px;padding:1.2rem 1.4rem;height:100%;}
.kpi .v{font-size:2.15rem;font-weight:800;line-height:1.1;display:flex;align-items:center;gap:.6rem;}
.kpi .v small{font-size:.7rem;font-weight:700;color:var(--blue);background:var(--tint);padding:.2rem .6rem;border-radius:999px;}
.kpi .t{color:#4A5468;margin-top:.55rem;font-size:.98rem;}
.kpi .c{color:var(--muted);margin-top:.35rem;font-size:.78rem;}

/* white chart / content cards */
div[class*="st-key-card"]{background:#fff;border-radius:20px;padding:1.2rem 1.4rem;box-shadow:0 1px 2px rgba(20,40,80,.04);}
.ctitle{font-size:1.08rem;font-weight:700;line-height:1.35;margin-bottom:.4rem;}
.csub{color:var(--muted);font-size:.85rem;margin-bottom:.4rem;}
[data-testid="stForm"]{background:#fff;border:none;border-radius:20px;padding:1.3rem 1.5rem;}
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid="stBaseButton-primaryFormSubmit"],[data-testid="stBaseButton-primary"]{background:var(--blue)!important;border-color:var(--blue)!important;border-radius:12px!important;font-weight:700;}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def md_inline(text: str) -> str:
    """Convert **bold** markdown to <b> for use inside raw HTML blocks."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def brand():
    st.markdown('<div class="brand"><div class="logo"></div><div><b>APIS 2022 · ML</b>'
                '<span>SDG 4 · Quality Education</span></div></div>', unsafe_allow_html=True)


def side_label(text: str):
    st.markdown(f'<div class="side-label">{text}</div>', unsafe_allow_html=True)


def header(title: str, subtitle: str, pills=("APIS 2022", "SDG 4 · Target 4.1", "For DepEd & policymakers")):
    chips = "".join(f'<span class="pill{" dark" if i == 0 else ""}">{p}</span>' for i, p in enumerate(pills))
    st.markdown(f'<div class="hdr"><div><h1>{title}</h1><p>{subtitle}</p></div><div>{chips}</div></div>',
                unsafe_allow_html=True)


def callout_card(label: str, body_html: str, tint: bool = False):
    st.markdown(f'<div class="ccard{" tint" if tint else ""}"><div class="lbl">{label}</div>'
                f'<p>{body_html}</p></div>', unsafe_allow_html=True)


def notice(body_html: str, lead: str = "", blue: bool = False):
    lead_html = f"<b>{lead}</b> " if lead else ""
    st.markdown(f'<div class="notice{" blue" if blue else ""}">{lead_html}{body_html}</div>',
                unsafe_allow_html=True)


def section(n: int, title: str):
    st.markdown(f'<div class="sec"><div class="n">{n}</div><h2>{title}</h2></div>', unsafe_allow_html=True)


def kpi(value: str, label: str, caption: str = "", badge: str = ""):
    b = f"<small>{badge}</small>" if badge else ""
    c = f'<div class="c">{caption}</div>' if caption else ""
    st.markdown(f'<div class="kpi"><div class="v">{value}{b}</div><div class="t">{label}</div>{c}</div>',
                unsafe_allow_html=True)


def card(key: str):
    """White rounded content card. `key` must be unique on the page."""
    return st.container(key=f"card_{key}")


def card_title(text: str, sub: str = ""):
    st.markdown(f'<div class="ctitle">{text}</div>' + (f'<div class="csub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)
