"""Presentation helpers: journal-style typography, numbered figure/table captions, key-figure strips."""
import html

import streamlit as st

SERIF = 'Georgia, "Times New Roman", Times, serif'

CSS = f"""
<style>
.block-container {{padding-top: 4.2rem; max-width: 1180px;}}
#MainMenu, footer {{visibility: hidden;}}
h1, h2, h3, h4, h5 {{font-family: {SERIF} !important; color: #111; font-weight: 600; letter-spacing: 0;}}
.titlebar {{border-bottom: 2px solid #111; padding-bottom: .9rem; margin-bottom: 1.1rem;}}
.titlebar .title {{font-family: {SERIF} !important; font-size: 2.1rem !important; font-weight: 600 !important; line-height: 1.22 !important; color: #111; margin: 0 !important;}}
.titlebar .byline {{font-size: 1rem !important; color: #555; margin: .4rem 0 0 0;}}
.abstract {{font-family: {SERIF}; font-size: 1.02rem; line-height: 1.62; color: #1a1a1a; margin: 0 0 .4rem 0;}}
.abstract .h {{font-weight: 700; font-variant: small-caps; letter-spacing: .04em; margin-right: .4rem;}}
.sec {{font-family: {SERIF}; font-size: 1.28rem; font-weight: 600; color: #111;
       border-bottom: 1px solid #BDBDBD; padding-bottom: .22rem; margin: 1.9rem 0 .7rem 0;}}
.lead {{font-size: .95rem; color: #333; line-height: 1.55; margin: 0 0 .8rem 0;}}
.cap {{font-family: {SERIF}; font-size: .92rem; color: #222; line-height: 1.5; margin: .1rem 0 1.6rem 0;}}
.cap b {{font-weight: 700;}}
.tcap {{font-family: {SERIF}; font-size: .92rem; color: #222; line-height: 1.5; margin: .9rem 0 .35rem 0;}}
.tcap b {{font-weight: 700;}}
.figs {{display: flex; border-top: 2px solid #111; border-bottom: 1px solid #BDBDBD; margin: .5rem 0 1.1rem 0;}}
.figs .f {{flex: 1; padding: .55rem .95rem .6rem .95rem; border-left: 1px solid #E3E3E3; min-width: 0;}}
.figs .f:first-child {{border-left: none; padding-left: .1rem;}}
.figs .l {{font-size: .7rem; text-transform: uppercase; letter-spacing: .09em; color: #555;}}
.figs .v {{font-family: {SERIF}; font-size: 1.75rem; color: #111; line-height: 1.25;
           font-variant-numeric: tabular-nums;}}
.figs .s {{font-size: .78rem; color: #666; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}}
.note {{border-left: 3px solid var(--c, #555); background: #F7F7F7; padding: .55rem .95rem; font-size: .93rem;
        line-height: 1.5; margin: .5rem 0 .9rem 0; color: #1a1a1a;}}
.note .t {{font-weight: 700; margin-right: .35rem;}}
.meta {{font-size: .85rem; color: #555; line-height: 1.55;}}
.meta b {{color: #111;}}
.refs {{font-size: .9rem; line-height: 1.55; padding-left: 1.6rem; text-indent: -1.6rem;}}
[data-baseweb="tab"] {{font-size: .95rem;}}
</style>
"""

LEVEL_COLOR = {"Low": "#7A7A7A", "Medium": "#B7791F", "High": "#C2410C", "Critical": "#7F1D1D", "note": "#555555"}


def esc(x) -> str:
    return html.escape(str(x))


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def reset_numbers():
    """Called once per full script run so captions are numbered top to bottom (fragments keep their numbers)."""
    st.session_state["_nums"] = {"Figure": {}, "Table": {}}


def _number(kind: str, key: str) -> int:
    nums = st.session_state.setdefault("_nums", {"Figure": {}, "Table": {}})[kind]
    return nums.setdefault(key, len(nums) + 1)


def figure_caption(key: str, text: str):
    st.markdown(f"<p class='cap'><b>Figure {_number('Figure', key)}.</b> {text}</p>", unsafe_allow_html=True)


def table_caption(key: str, text: str):
    st.markdown(f"<p class='tcap'><b>Table {_number('Table', key)}.</b> {text}</p>", unsafe_allow_html=True)


def section(title: str, lead: str = None):
    st.markdown(f"<div class='sec'>{esc(title)}</div>", unsafe_allow_html=True)
    if lead:
        st.markdown(f"<p class='lead'>{lead}</p>", unsafe_allow_html=True)


def key_figures(items):
    """items: list of (label, value, note). Flat strip of headline numbers."""
    cells = "".join(f"<div class='f'><div class='l'>{esc(l)}</div><div class='v'>{esc(v)}</div>"
                    f"<div class='s'>{esc(s)}&nbsp;</div></div>" for l, v, s in items)
    st.markdown(f"<div class='figs'>{cells}</div>", unsafe_allow_html=True)


def callout(text_html: str, level: str = "note", title: str = None):
    """text_html must already be escaped where it contains data."""
    head = f"<span class='t'>{esc(title)}</span>" if title else ""
    st.markdown(f"<div class='note' style='--c:{LEVEL_COLOR.get(level, '#555')}'>{head}{text_html}</div>",
                unsafe_allow_html=True)
