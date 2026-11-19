import base64
import html
import json
import os
from datetime import datetime

import numpy as np
import streamlit as st
from plotly.offline import get_plotlyjs_version

SERIF = 'Georgia, "Times New Roman", Times, serif'
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"

CSS = f"""
<style>
.block-container {{padding-top: 4.2rem; max-width: 1180px;}}
#MainMenu, footer {{visibility: hidden;}}
h1, h2, h3, h4, h5 {{font-family: {SERIF} !important; color: #111; font-weight: 600; letter-spacing: 0;}}
.titlebar {{padding-bottom: .5rem;}}
.titlebar > div {{flex: 1 1 360px; min-width: 0;}}
.titlebar > svg {{flex: 0 0 330px;}}
.titlebar .title {{font-family: {SERIF} !important; font-size: 2.1rem !important; font-weight: 600 !important; line-height: 1.22 !important; color: #111; margin: 0 !important;}}
.titlebar .byline {{font-size: 1rem !important; color: #555; margin: .4rem 0 0 0;}}
.abstract {{font-family: {SERIF}; font-size: 1.02rem; line-height: 1.62; color: #1a1a1a; margin: 0 0 .4rem 0;}}
.abstract .h {{font-weight: 700; font-variant: small-caps; letter-spacing: .04em; margin-right: .4rem;}}
.sec {{font-family: {SERIF}; font-size: 1.28rem; font-weight: 600; color: #111; margin: 1.9rem 0 .7rem 0;}}
.sec::after {{content: ""; display: block; height: 1px; background: #BDBDBD; margin-top: .25rem;
              transform-origin: left; animation: grow 1.1s .15s cubic-bezier(.2,.7,.2,1) both;}}
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

/* ---------------------------------------------------------------- motion */
@keyframes rise {{from {{opacity: 0; transform: translateY(14px);}} to {{opacity: 1; transform: none;}}}}
@keyframes grow {{from {{transform: scaleX(0);}} to {{transform: scaleX(1);}}}}
@keyframes draw {{from {{stroke-dashoffset: 1;}} to {{stroke-dashoffset: 0;}}}}
@keyframes fade {{from {{opacity: 0;}} to {{opacity: 1;}}}}
@keyframes ping {{0% {{r: 3; opacity: .75;}} 100% {{r: 13; opacity: 0;}}}}
@keyframes blink {{0%, 100% {{opacity: 1;}} 50% {{opacity: .25;}}}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {{animation: rise .7s cubic-bezier(.2,.7,.2,1) both;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(2) {{animation-delay: .05s;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) {{animation-delay: .10s;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(4) {{animation-delay: .15s;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(5) {{animation-delay: .20s;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(6) {{animation-delay: .25s;}}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(n+7) {{animation-delay: .30s;}}
[data-baseweb="tab-panel"] {{animation: fade .5s ease both;}}
.titlebar {{display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem; flex-wrap: wrap;}}
.rule {{height: 2px; background: #111; transform-origin: left; animation: grow 1.3s cubic-bezier(.2,.7,.2,1) both;
        margin-bottom: 1.1rem;}}
.trace {{display: block;}}
.trace .axis {{stroke: #888; stroke-width: .8;}}
.trace .tick {{font: 8.5px {SERIF}; fill: #666;}}
.trace .cap {{font: italic 9px {SERIF}; fill: #555;}}
.trace path {{fill: none; stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round;
              stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 2.6s cubic-bezier(.4,0,.2,1) .4s forwards;}}
.trace path.ok {{stroke: #0072B2;}}
.trace path.bad {{stroke: #D55E00; animation-delay: 1.1s;}}
.trace circle.pk {{fill: #0072B2;}}
.trace circle.ring {{fill: none; stroke: #0072B2; stroke-width: 1; animation: ping 2.4s ease-out 2.6s infinite;}}
.dot {{display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: .5rem; background: #7A7A7A;}}
.dot.ok {{background: #009E73;}}
.dot.live {{background: #009E73; animation: blink 1.8s ease-in-out infinite;}}
@media (prefers-reduced-motion: reduce) {{*, *::before, *::after {{animation: none !important; transition: none !important;}}}}
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


def _embed(doc: str, height: int):
    """Raw HTML in an iframe: st.iframe on current Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(doc, height=height, scrolling=False)


def animating() -> bool:
    return bool(st.session_state.get("animate", True))


def _plain(o):
    """Plotly figure dict -> JSON-safe python (numpy arrays become plain lists, datetimes become strings)."""
    if isinstance(o, dict):
        if "bdata" in o and "dtype" in o:                      # plotly >= 6 stores numpy arrays as base64
            raw = o["bdata"]
            raw = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
            arr = np.frombuffer(raw, dtype=np.dtype(o["dtype"]).newbyteorder("<"))
            if "shape" in o:
                arr = arr.reshape([int(x) for x in str(o["shape"]).split(",")])
            return _plain(arr)
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        if np.issubdtype(o.dtype, np.datetime64):
            return o.astype("datetime64[ms]").astype(str).tolist()
        return _plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and o != o:
        return None
    if hasattr(o, "isoformat"):
        return o.isoformat(sep=" ")
    return o


def _js(obj) -> str:
    return json.dumps(_plain(obj), default=str).replace("</", "<\\/")


def key_figures(items):
    """items: list of (label, value, note). Flat strip of headline numbers (numbers count up when scrolled into view)."""
    if not animating():
        cells = "".join(f"<div class='f'><div class='l'>{esc(l)}</div><div class='v'>{esc(v)}</div>"
                        f"<div class='s'>{esc(s)}&nbsp;</div></div>" for l, v, s in items)
        st.markdown(f"<div class='figs'>{cells}</div>", unsafe_allow_html=True)
        return
    html_doc = _KF_HTML.replace("__ITEMS__", _js([[str(l), str(v), str(s)] for l, v, s in items]))
    _embed(html_doc, 112)


def plot(fig):
    """Render a Plotly figure; when animations are on, traces build up (bars grow, curves draw) on first view."""
    if not animating():
        st.plotly_chart(fig, width="stretch", theme=None,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "svg", "scale": 3}})
        return
    d = fig.to_dict()
    height = int(d.get("layout", {}).get("height") or 360)
    cfg = {"displaylogo": False, "toImageButtonOptions": {"format": "svg", "scale": 3}}
    doc = (_PLOT_HTML.replace("__CDN__", PLOTLY_CDN).replace("__FIG__", _js({"data": d["data"], "layout": d["layout"]}))
           .replace("__CFG__", _js(cfg)).replace("__H__", str(height)))
    _embed(doc, height + 4)


def hero(title: str, byline: str, ok, bad):
    """Title block with a hand-drawn-style signal trace of hourly logon volume (drawn in once on load)."""
    ok, bad = list(ok), list(bad)
    top = max(max(o + b for o, b in zip(ok, bad)), 1)
    w, h, x0, x1, base, amp = 330, 96, 10, 322, 70, 58

    def path(vals):
        pts = [(x0 + i * (x1 - x0) / (len(vals) - 1), base - v / top * amp) for i, v in enumerate(vals)]
        return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts), pts

    p_ok, pts = path([o + b for o, b in zip(ok, bad)])
    p_bad, _ = path(bad)
    pk = max(range(len(pts)), key=lambda i: -pts[i][1])
    ticks = "".join(f"<text class='tick' x='{x0 + i * (x1 - x0) / 23:.1f}' y='84' text-anchor='middle'>{i:02d}</text>"
                    for i in (0, 6, 12, 18, 23))
    svg = (f"<svg class='trace' width='{w}' height='{h}' viewBox='0 0 {w} {h}' role='img' "
           f"aria-label='Hourly logon volume'>"
           f"<line class='axis' x1='{x0}' y1='{base}' x2='{x1}' y2='{base}'/>{ticks}"
           f"<path class='ok' pathLength='1' d='{p_ok}'/><path class='bad' pathLength='1' d='{p_bad}'/>"
           f"<circle class='ring' cx='{pts[pk][0]:.1f}' cy='{pts[pk][1]:.1f}' r='3'/>"
           f"<circle class='pk' cx='{pts[pk][0]:.1f}' cy='{pts[pk][1]:.1f}' r='3'/>"
           f"<text class='cap' x='{x0}' y='95'>Logons per hour of day: all (blue), failed (red)</text></svg>")
    st.markdown(f"<div class='titlebar'><div><p class='title'>{esc(title)}</p><p class='byline'>{esc(byline)}</p></div>"
                f"{svg}</div><div class='rule'></div>", unsafe_allow_html=True)


def status_block(models_dir, alert_file):
    """Small instrument-style status panel for the sidebar."""
    ready = all((models_dir / f).exists() for f in ("scaler.pkl", "isolation_forest.pkl", "oneclass_svm.pkl",
                                                    "lof.pkl"))
    rows = [("ok" if ready else "", "Models loaded" if ready else "Models not found")]
    if os.path.exists(alert_file):
        age = datetime.now().timestamp() - os.path.getmtime(alert_file)
        live = age < 120
        txt = "Monitor active" if live else f"Alert log updated {_ago(age)} ago"
        rows.append(("live" if live else "ok", txt))
    else:
        rows.append(("", "No alert log yet"))
    st.markdown("<div class='meta'><b>System status</b><br>" + "<br>".join(
        f"<span class='dot {c}'></span>{esc(t)}" for c, t in rows) + "</div>", unsafe_allow_html=True)


def _ago(sec: float) -> str:
    for n, u in ((86400, "d"), (3600, "h"), (60, "min")):
        if sec >= n:
            return f"{int(sec // n)} {u}"
    return f"{int(sec)} s"


_KF_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;font-family:"Helvetica Neue",Arial,sans-serif;color:#111}
.figs{display:flex;position:relative;border-bottom:1px solid #BDBDBD;margin-top:6px}
.figs::before{content:"";position:absolute;left:0;top:0;height:2px;width:100%;background:#111;transform-origin:left;transform:scaleX(0)}
.f{flex:1;min-width:0;padding:10px 14px 10px 14px;border-left:1px solid #E3E3E3;opacity:0;transform:translateY(8px)}
.f:first-child{border-left:none;padding-left:2px}
.l{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#555}
.v{font:28px Georgia,"Times New Roman",serif;line-height:1.25;font-variant-numeric:tabular-nums}
.s{font-size:12.5px;color:#666;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.go .figs::before{animation:grow 1.1s cubic-bezier(.2,.7,.2,1) forwards}
.go .f{animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:calc(var(--i)*110ms + 150ms)}
@keyframes grow{to{transform:scaleX(1)}}
@keyframes rise{to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.figs::before{transform:none}.f{opacity:1;transform:none}.go .f,.go .figs::before{animation:none}}
</style></head><body><div class="figs" id="s"></div><script>
const ITEMS=__ITEMS__;
const root=document.getElementById('s');
ITEMS.forEach(function(it,i){
  var d=document.createElement('div');d.className='f';d.style.setProperty('--i',i);
  d.innerHTML='<div class="l"></div><div class="v"></div><div class="s"></div>';
  d.children[0].textContent=it[0];d.children[1].textContent=it[1];d.children[2].textContent=it[2]||'\\u00a0';
  root.appendChild(d);
});
function parse(v){var m=/^(\\d[\\d,]*)(\\.(\\d+))?(%?)$/.exec(v);if(!m)return null;
  return {n:parseFloat(m[1].replace(/,/g,'')+(m[2]||'')),dec:m[3]?m[3].length:0,comma:m[1].indexOf(',')>-1,pct:m[4]};}
function fmt(x,p){var s=p.comma?x.toLocaleString('en-US',{minimumFractionDigits:p.dec,maximumFractionDigits:p.dec}):x.toFixed(p.dec);return s+p.pct;}
function run(){
  document.body.classList.add('go');
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  Array.from(root.children).forEach(function(c,i){
    var el=c.children[1],p=parse(ITEMS[i][1]);if(!p)return;
    var t0=performance.now()+150+i*110,dur=1300;el.textContent=fmt(0,p);
    function step(now){var k=Math.min(Math.max((now-t0)/dur,0),1),e=1-Math.pow(1-k,3);el.textContent=fmt(p.n*e,p);if(k<1)requestAnimationFrame(step);else el.textContent=ITEMS[i][1];}
    requestAnimationFrame(step);
  });
}
new IntersectionObserver(function(es,io){if(es.some(function(e){return e.isIntersecting})){io.disconnect();run();}},{threshold:0.3}).observe(root);
</script></body></html>"""

_PLOT_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent}#g{width:100%;height:__H__px}
</style><script src="__CDN__"></script></head><body><div id="g"></div><script>
(function(){
var FIG=__FIG__,CFG=__CFG__;
var gd=document.getElementById('g'),data=FIG.data,layout=FIG.layout||{};
delete layout.width;layout.autosize=true;layout.height=__H__;
var config=Object.assign({responsive:true},CFG);
var ANIM=!matchMedia('(prefers-reduced-motion: reduce)').matches;
function L(a){return Array.isArray(a)?a:[];}
function frame(t){
  return data.map(function(tr){
    var o={},ty=tr.type||'scatter';
    if(ty==='bar'){
      if(tr.orientation==='h')o.x=L(tr.x).map(function(v){return v*t;});else o.y=L(tr.y).map(function(v){return v*t;});
    }else if(ty==='scatter'||ty==='scattergl'){
      var n=L(tr.x).length,k=t>=1?n:Math.floor(t*n);
      o.x=tr.x.slice(0,k);o.y=tr.y.slice(0,k);if(Array.isArray(tr.text))o.text=tr.text.slice(0,k);
    }else if(ty==='heatmap'){
      o.z=tr.z.map(function(r){return r.map(function(v){return v*t;});});
    }
    return o;
  });
}
function merged(t){var f=frame(t);return data.map(function(tr,i){return Object.assign({},tr,f[i]);});}
data.forEach(function(tr){
  if(tr.type==='heatmap'&&tr.zmin==null){var fl=[].concat.apply([],tr.z);tr.zmin=Math.min.apply(null,fl);tr.zmax=Math.max.apply(null,fl);}
});
function done(){Plotly.react(gd,data,layout,config);}
function play(){
  var N=36,fr=[],idx=data.map(function(_,i){return i;});
  for(var i=1;i<=N;i++){var x=i/N;fr.push({data:frame(1-Math.pow(1-x,3)),traces:idx});}
  Plotly.animate(gd,fr,{frame:{duration:28,redraw:true},transition:{duration:0},mode:'immediate'}).then(done).catch(done);
}
var started=false;
function start(){
  if(started)return;started=true;gd.style.visibility='hidden';
  Plotly.newPlot(gd,data,layout,config).then(function(){
    if(!ANIM){gd.style.visibility='visible';return;}
    Object.keys(gd._fullLayout).forEach(function(k){
      if(/^[xy]axis\\d*$/.test(k)){var ax=gd._fullLayout[k],u=layout[k]||{};
        if(u.range==null&&ax.range){layout[k]=Object.assign({},u,{range:ax.range.slice(),autorange:false});}}
    });
    return Plotly.react(gd,merged(0),layout,config).then(function(){gd.style.visibility='visible';play();});
  }).catch(function(e){console.error(e);gd.style.visibility='visible';});
}
new IntersectionObserver(function(es,io){if(es.some(function(e){return e.isIntersecting})){io.disconnect();start();}},{threshold:0.2}).observe(gd);
})();
</script></body></html>"""


def callout(text_html: str, level: str = "note", title: str = None):
    """text_html must already be escaped where it contains data."""
    head = f"<span class='t'>{esc(title)}</span>" if title else ""
    st.markdown(f"<div class='note' style='--c:{LEVEL_COLOR.get(level, '#555')}'>{head}{text_html}</div>",
                unsafe_allow_html=True)