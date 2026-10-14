"""Past-Me AI - a chatbot decision coach (pure Python / Streamlit).   Run:  streamlit run app.py"""
import json
import time
import uuid
from datetime import date, datetime
from html import escape as e

import altair as alt
import pandas as pd
import streamlit as st

from decision_sim import config, explain, llm, stats, store
from decision_sim.engine import Engine
from decision_sim.schema import (CHOICE_LABELS, CHOICES, DECISION_TYPES, DEFAULTS, EXPERIENCE, RESULTS,
                                 RISK, TIME, TYPE_ICONS)
from decision_sim.understand import understand

st.set_page_config(page_title="Past-Me AI · Decision Coach", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")
CHATS_PATH = config.ROOT / "data" / "chats.json"

# =================================================================== styles
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@property --n{syntax:'<integer>';initial-value:0;inherits:false}
:root{--bg:#08080f;--glass:rgba(255,255,255,.035);--glass2:rgba(255,255,255,.065);--line:rgba(255,255,255,.08);--line2:rgba(255,255,255,.14);
--txt:#ededf5;--txt2:#c8c8da;--mute:#8b8ba3;--v:#8b5cf6;--c:#06b6d4;--p:#ec4899;--g:#10b981;--a:#f59e0b;--r:#f43f5e;
--grad:linear-gradient(135deg,#8b5cf6 0%,#6366f1 45%,#06b6d4 100%);--ease:cubic-bezier(.2,.8,.2,1);--spring:cubic-bezier(.34,1.56,.64,1)}
html,body,.stApp,button,input,textarea,select{font-family:'Inter',system-ui,sans-serif}
.stApp{background:var(--bg)}
[data-testid="stAppViewContainer"]{background:
 radial-gradient(700px 520px at 15% -8%,rgba(124,58,237,.30),transparent 60%),
 radial-gradient(620px 520px at 105% 40%,rgba(8,145,178,.20),transparent 60%),
 radial-gradient(600px 480px at 45% 115%,rgba(219,39,119,.16),transparent 60%),var(--bg);
 background-size:140% 140%;animation:aurora 28s ease-in-out infinite alternate}
@keyframes aurora{0%{background-position:0% 0%}100%{background-position:100% 100%}}
[data-testid="stHeader"]{background:transparent}
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent!important}
.block-container{max-width:860px;padding-top:1.4rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:rgba(11,11,20,.82);backdrop-filter:blur(22px);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebarContent"]{padding-top:.6rem}
@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pop{0%{opacity:0;transform:scale(.85)}100%{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes flow{to{background-position:300% center}}
@keyframes grow{from{width:0}to{width:var(--w)}}
@keyframes ring{to{stroke-dashoffset:var(--off)}}
@keyframes count{to{--n:var(--t)}}
@keyframes sheen{0%{transform:translateX(-100%)}60%,100%{transform:translateX(100%)}}
@keyframes morph{0%,100%{border-radius:50%}33%{border-radius:44% 56% 52% 48%/52% 44% 56% 48%}66%{border-radius:56% 44% 48% 52%/44% 56% 44% 56%}}
@keyframes ping{0%{transform:scale(.9);opacity:1}100%{transform:scale(1.4);opacity:0}}
@keyframes beat{0%{box-shadow:0 0 0 0 rgba(16,185,129,.6)}70%{box-shadow:0 0 0 8px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}
@keyframes breathe{50%{transform:scale(1.15);opacity:.55}}
@keyframes cardIn{from{opacity:0;transform:translateY(22px) scale(.97);filter:blur(4px)}to{opacity:1;transform:none;filter:none}}
@keyframes show{to{opacity:1}}
@keyframes hide{to{opacity:0}}
.count{animation:count 1.8s var(--ease) .3s forwards;counter-reset:n var(--n)}.count::after{content:counter(n)}
.grad-text{background:linear-gradient(90deg,#a78bfa,#22d3ee,#f472b6,#a78bfa);background-size:300% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:flow 7s linear infinite}

/* ---------- sidebar ---------- */
.brand{display:flex;align-items:center;gap:12px;margin:0 0 14px}
.logo{position:relative;width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--grad);font-size:1.2rem;box-shadow:0 8px 24px -8px rgba(139,92,246,.9)}
.logo::after{content:"";position:absolute;inset:-3px;border-radius:14px;border:1px solid rgba(139,92,246,.6);animation:ping 2.8s var(--ease) infinite}
.brand b{font-family:'Space Grotesk';font-size:1.15rem;display:block;line-height:1.1}.brand small{color:var(--mute);font-size:.75rem}
.side-label{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--mute);margin:14px 0 4px 4px}
.st-key-newchat button{background:var(--grad)!important;border:none!important;color:#fff!important;font-weight:600;border-radius:14px!important;
 box-shadow:0 10px 28px -12px rgba(139,92,246,1);transition:transform .25s var(--spring),box-shadow .25s!important}
.st-key-newchat button:hover{transform:translateY(-2px);box-shadow:0 16px 34px -12px rgba(139,92,246,1)}
[class*="st-key-hist_"] button{justify-content:flex-start!important;text-align:left;background:transparent!important;border:1px solid transparent!important;
 border-radius:10px!important;padding:6px 10px!important;min-height:0;color:var(--txt2)!important;transition:background .2s,transform .2s!important}
[class*="st-key-hist_"] button p{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.87rem}
[class*="st-key-hist_"] button:hover{background:var(--glass2)!important;transform:translateX(3px)}
[class*="st-key-hist_"] button[kind="primary"]{background:var(--glass2)!important;border-color:var(--line2)!important;color:var(--txt)!important}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:8px 12px;border-radius:11px;margin:1px 0;width:100%;transition:background .2s}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--glass2)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,rgba(139,92,246,.28),rgba(6,182,212,.12));box-shadow:inset 0 0 0 1px rgba(139,92,246,.35)}
.status{display:flex;align-items:center;gap:8px;font-size:.8rem;color:var(--mute);margin-top:6px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--a)}.dot.on{background:var(--g);animation:beat 2s infinite}

/* ---------- topbar ---------- */
.topbar{display:flex;align-items:center;gap:10px;animation:up .4s var(--ease) both}
.tb-title{font-weight:600;font-size:1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill-model{display:inline-flex;align-items:center;gap:7px;font-size:.74rem;font-weight:600;padding:5px 11px;border-radius:99px;border:1px solid var(--line);background:var(--glass);color:var(--txt2)}

/* ---------- welcome ---------- */
.welcome{display:flex;flex-direction:column;align-items:center;text-align:center;padding:4vh 0 8px}
.orb{position:relative;width:100px;height:100px;margin-bottom:24px;animation:pop .8s var(--spring) both}
.orb .core{position:absolute;inset:20px;background:var(--grad);display:grid;place-items:center;font-size:1.8rem;
 box-shadow:0 0 50px rgba(139,92,246,.75),inset 0 0 18px rgba(255,255,255,.3);animation:morph 8s ease-in-out infinite}
.orb .ring{position:absolute;inset:0;border-radius:50%;border:2px solid transparent;border-top-color:#22d3ee;border-right-color:#8b5cf6;animation:spin 2.6s linear infinite}
.orb .ring.r2{inset:-12px;border-top-color:#f472b6;border-right-color:transparent;border-left-color:#8b5cf6;animation-duration:4.5s;animation-direction:reverse;opacity:.7}
.orb .sat{position:absolute;width:9px;height:9px;border-radius:50%;background:#f472b6;box-shadow:0 0 14px #f472b6;top:-20px;left:45px;transform-origin:5px 70px;animation:spin 6s linear infinite}
.welcome h1{font-family:'Space Grotesk';font-size:clamp(1.9rem,4.4vw,2.8rem);line-height:1.12;letter-spacing:-.02em;font-weight:700;margin:0;padding:0;animation:up .7s .1s var(--ease) both}
.welcome p{color:var(--mute);max-width:560px;margin:12px auto 26px;font-size:1rem;animation:up .7s .2s var(--ease) both}
.caps{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin:18px 0 6px}
.cap{display:inline-flex;align-items:center;gap:6px;font-size:.78rem;color:var(--mute);padding:6px 12px;border-radius:99px;border:1px solid var(--line);background:var(--glass);animation:pop .5s var(--spring) both}
[class*="st-key-sg"] button{text-align:left;justify-content:flex-start!important;align-items:flex-start!important;min-height:96px;padding:14px 16px!important;
 border-radius:18px!important;background:var(--glass)!important;border:1px solid var(--line)!important;backdrop-filter:blur(10px);
 transition:transform .35s var(--spring),border-color .3s,box-shadow .3s!important;animation:up .6s var(--ease) both}
[class*="st-key-sg"] button p{font-size:.9rem;color:var(--txt2);line-height:1.5}
[class*="st-key-sg"] button:hover{transform:translateY(-4px);border-color:rgba(139,92,246,.55)!important;box-shadow:0 0 0 1px rgba(139,92,246,.3),0 18px 46px -18px rgba(139,92,246,.7)}
.st-key-sg1 button{animation-delay:.08s}.st-key-sg2 button{animation-delay:.16s}.st-key-sg3 button{animation-delay:.24s}

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"]{background:transparent;padding:6px 0;gap:14px;animation:up .5s var(--ease) both}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
 flex:0 1 auto;max-width:82%;background:rgba(139,92,246,.16);border:1px solid rgba(139,92,246,.28);border-radius:20px 20px 6px 20px;padding:10px 16px}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p{margin:0}
[data-testid="stChatMessageAvatarAssistant"]{background:var(--grad)!important;border-radius:11px;box-shadow:0 6px 18px -6px rgba(139,92,246,.9);color:#fff}
[data-testid="stChatMessageAvatarUser"]{background:var(--glass2)!important;border:1px solid var(--line2);border-radius:11px}
[data-testid="stChatMessageContent"] h3{font-family:'Space Grotesk';font-size:1.06rem;padding:.7rem 0 .2rem;display:flex;align-items:center;gap:8px}
[data-testid="stChatMessageContent"] h3::before{content:"";width:4px;height:1em;border-radius:3px;background:var(--grad)}
[data-testid="stChatMessageContent"] li::marker{color:var(--v)}
[data-testid="stChatMessageContent"] strong{color:#fff}
[data-testid="stChatMessageContent"] blockquote{border-left:3px solid var(--a);background:rgba(245,158,11,.08);border-radius:0 10px 10px 0;padding:6px 12px}
[data-testid="stChatInput"]{border-radius:22px!important;border:1px solid var(--line2)!important;background:rgba(16,16,28,.92)!important;
 box-shadow:0 20px 60px -24px rgba(0,0,0,.8);transition:border-color .3s,box-shadow .3s}
[data-testid="stChatInput"]:focus-within{border-color:rgba(139,92,246,.65)!important;box-shadow:0 0 0 4px rgba(139,92,246,.14),0 20px 60px -24px rgba(139,92,246,.6)}
[data-testid="stChatInputSubmitButton"]{background:var(--grad)!important;border-radius:12px!important;color:#fff!important}
[class*="st-key-fu_"] button{border-radius:99px!important;border:1px solid var(--line2)!important;background:var(--glass)!important;padding:4px 14px!important;
 min-height:0;transition:all .3s var(--spring)!important;animation:pop .45s var(--spring) both}
[class*="st-key-fu_"] button p{font-size:.84rem}
[class*="st-key-fu_"] button:hover{border-color:var(--v)!important;transform:translateY(-2px);box-shadow:0 8px 22px -10px rgba(139,92,246,.9);background:rgba(139,92,246,.14)!important}
[class*="st-key-act_"] button{background:transparent!important;border:1px solid var(--line)!important;border-radius:10px!important;padding:2px 10px!important;min-height:0;color:var(--mute)!important}
[class*="st-key-act_"] button:hover{color:var(--txt)!important;border-color:var(--v)!important}
.notice{font-size:.8rem;color:var(--a);margin-top:6px}

/* ---------- thinking ---------- */
.thinking{display:inline-flex;flex-direction:column;gap:8px;padding:14px 18px;border-radius:16px;background:var(--glass);border:1px solid var(--line);min-width:290px;animation:up .4s var(--ease) both}
.tk{display:flex;align-items:center;gap:11px;font-size:.88rem;color:var(--txt2);opacity:.35;animation:show .4s var(--d) forwards}
.tk .ic{position:relative;width:18px;height:18px;flex-shrink:0}
.tk .ic::before{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid var(--line2);border-top-color:#22d3ee;border-right-color:#8b5cf6;opacity:0;
 animation:spin .7s linear infinite,show .01s var(--d) forwards,hide .01s var(--e) forwards}
.tk .ic::after{content:"✓";position:absolute;inset:0;border-radius:50%;background:var(--g);color:#fff;font-size:.66rem;font-weight:800;display:grid;place-items:center;
 opacity:0;transform:scale(.4);animation:pop .35s var(--spring) var(--e) forwards}
.tk .lbl{background:linear-gradient(90deg,var(--mute),#fff,var(--mute));background-size:300% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:flow 2.4s linear infinite}

/* ---------- decision card ---------- */
.dcard{position:relative;margin:16px 0 6px;border-radius:22px;background:var(--glass);backdrop-filter:blur(14px);padding:20px 22px;
 box-shadow:0 22px 60px -26px rgba(0,0,0,.8);animation:cardIn .75s var(--ease) both}
.dcard::before{content:"";position:absolute;inset:0;border-radius:22px;padding:1px;pointer-events:none;
 background:linear-gradient(135deg,rgba(139,92,246,.75),var(--line) 35%,var(--line) 65%,rgba(6,182,212,.75));
 -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}
.dc-head{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.dc-title{font-weight:600;color:var(--txt);margin-right:auto}
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:99px;font-size:.76rem;font-weight:700;border:1px solid;animation:pop .5s .25s var(--spring) both}
.badge.go{color:#34d399;background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.4)}
.badge.mid{color:#fbbf24;background:rgba(245,158,11,.1);border-color:rgba(245,158,11,.4)}
.badge.stop{color:#fb7185;background:rgba(244,63,94,.1);border-color:rgba(244,63,94,.4)}
.badge.ghost{color:var(--mute);border-color:var(--line2);background:var(--glass2)}
.dc-hero{display:grid;grid-template-columns:150px 1fr;gap:22px;align-items:center;padding-bottom:16px;border-bottom:1px solid var(--line)}
@media(max-width:640px){.dc-hero{grid-template-columns:1fr;justify-items:center;text-align:center}}
.gauge{position:relative;width:150px;height:150px}
.gauge::before{content:"";position:absolute;inset:18px;border-radius:50%;background:radial-gradient(circle,var(--gc),transparent 70%);animation:breathe 3s ease-in-out infinite}
.gauge svg{position:relative;width:150px;height:150px;transform:rotate(-90deg)}
.gauge .trk{fill:none;stroke:var(--line2);stroke-width:10}
.gauge .arc{fill:none;stroke-width:10;stroke-linecap:round;stroke-dasharray:327;stroke-dashoffset:327;animation:ring 1.9s var(--ease) .25s forwards}
.gauge .gv{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge .gv b{font-family:'Space Grotesk';font-size:2.4rem;line-height:1;color:var(--txt)}.gauge .gv b i{font-style:normal;font-size:1rem;color:var(--mute)}
.gauge .gv span{font-size:.6rem;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);margin-top:5px}
.eyebrow{font-size:.68rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#22d3ee}
.rec{font-family:'Space Grotesk';font-size:1.42rem;font-weight:700;line-height:1.2;color:var(--txt);margin:5px 0 12px;animation:up .6s .3s var(--ease) both}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px}
.stat{padding:9px 12px;border-radius:12px;background:var(--glass2);border:1px solid var(--line);animation:up .5s var(--ease) both;text-align:left}
.stat small{display:block;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mute)}
.stat b{font-size:.93rem;color:var(--txt);display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meter{height:5px;border-radius:9px;background:var(--line2);margin-top:6px;overflow:hidden}
.meter>div{height:100%;width:0;border-radius:9px;background:var(--grad);animation:grow 1.5s var(--ease) .4s forwards}
.sec-h{font-size:.8rem;font-weight:600;color:var(--mute);margin:16px 0 4px}
.opt{margin:12px 0;animation:up .5s var(--ease) both}
.opt-h{display:flex;justify-content:space-between;align-items:center;gap:10px;font-size:.92rem;margin-bottom:7px;color:var(--txt)}
.opt-h .n{display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-weight:550}.opt-h .s{font-family:'Space Grotesk';font-weight:700}
.track{height:10px;border-radius:99px;background:var(--line);overflow:hidden}
.fill{position:relative;overflow:hidden;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,#8b5cf6,#06b6d4);animation:grow 1.4s var(--ease) forwards}
.fill::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.45),transparent);animation:sheen 2.8s 1.6s ease-in-out infinite;transform:translateX(-100%)}
.fill.best{background:linear-gradient(90deg,#10b981,#06b6d4)}
.pill{font-size:.63rem;font-weight:700;padding:2px 8px;border-radius:99px;border:1px solid var(--line2);color:var(--mute);white-space:nowrap}
.pill.low{color:#34d399;border-color:rgba(16,185,129,.4)}.pill.medium{color:#fbbf24;border-color:rgba(245,158,11,.4)}
.pill.high{color:#fb7185;border-color:rgba(244,63,94,.4)}.pill.star{color:#fff;background:var(--grad);border-color:transparent}
.pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}@media(max-width:640px){.pc-grid{grid-template-columns:1fr}}
.pc-name{grid-column:1/-1;font-weight:600;color:var(--txt);display:flex;gap:8px;align-items:center}
.pc{padding:10px 14px;border-radius:14px;background:var(--glass2);border:1px solid var(--line)}
.pc.pro{border-top:2px solid var(--g)}.pc.con{border-top:2px solid var(--r)}
.li{display:flex;gap:9px;padding:5px 0;font-size:.88rem;color:var(--txt2);line-height:1.5;animation:up .45s var(--ease) both}
.li .b{flex-shrink:0;width:18px;height:18px;border-radius:6px;display:grid;place-items:center;font-size:.7rem;font-weight:800;margin-top:2px}
.b.p{background:rgba(16,185,129,.16);color:#34d399}.b.c{background:rgba(244,63,94,.16);color:#fb7185}.b.q{background:rgba(6,182,212,.16);color:#22d3ee}
.fac{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;padding:10px 0;border-bottom:1px dashed var(--line);animation:up .45s var(--ease) both}
.fac:last-child{border-bottom:none}.fac b{color:var(--txt);font-size:.9rem}.fac .nt{font-size:.82rem;color:var(--mute)}
.wd{display:flex;gap:3px}.wd i{width:14px;height:6px;border-radius:3px;background:var(--line2)}
.wd i.on.positive{background:#10b981}.wd i.on.negative{background:#f43f5e}.wd i.on.neutral{background:#f59e0b}
.risk{padding:11px 14px;border-radius:12px;background:var(--glass2);border:1px solid var(--line);margin-bottom:8px;animation:up .45s var(--ease) both}
.risk b{color:var(--txt);font-size:.9rem;display:block}.risk span{font-size:.84rem;color:var(--mute)}
.st{display:flex;gap:12px;align-items:flex-start;padding:8px 0;animation:up .45s var(--ease) both}
.num{width:26px;height:26px;border-radius:50%;flex-shrink:0;display:grid;place-items:center;font-size:.76rem;font-weight:700;color:#fff;background:var(--grad);box-shadow:0 0 16px rgba(139,92,246,.5)}
.st .tx{padding-top:2px;font-size:.9rem;color:var(--txt2)}
.quote{padding:13px 16px;border-radius:14px;background:var(--glass2);border-left:3px solid var(--v);color:var(--txt2);font-style:italic;margin:6px 0 14px}
.quote.f{border-left-color:var(--c)}
.ref-row{display:flex;justify-content:space-between;font-size:.86rem;padding:6px 0;border-bottom:1px dashed var(--line);color:var(--txt2)}
.muted{color:var(--mute);font-size:.84rem}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--glass2);padding:4px;border-radius:13px;border:1px solid var(--line);overflow-x:auto}
.stTabs [data-baseweb="tab"]{border-radius:9px;padding:6px 12px;height:auto;transition:background .3s,color .3s}
.stTabs [data-baseweb="tab"] p{font-size:.82rem}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(139,92,246,.4),rgba(6,182,212,.22))}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-panel"]{animation:up .4s var(--ease) both}

/* ---------- pages ---------- */
.page-head{margin:6px 0 18px;animation:up .5s var(--ease) both}
.page-head h1{font-family:'Space Grotesk';font-size:clamp(1.6rem,3.4vw,2.2rem);letter-spacing:-.02em;margin:2px 0 6px;padding:0}
.page-head p{color:var(--mute);max-width:640px;margin:0}
.card{background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:18px 20px;backdrop-filter:blur(12px);margin:8px 0;
 animation:up .55s var(--ease) both;transition:transform .3s var(--spring),border-color .3s,box-shadow .3s}
.card:hover{transform:translateY(-3px);border-color:rgba(139,92,246,.45);box-shadow:0 0 0 1px rgba(139,92,246,.25),0 18px 50px -20px rgba(139,92,246,.6)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}@media(max-width:760px){.tiles{grid-template-columns:1fr 1fr}}
.tile{position:relative;margin:0}.tile small{color:var(--mute);font-size:.78rem}
.tile b{display:block;font-family:'Space Grotesk';font-size:2rem;line-height:1.2;margin-top:4px}
.tile .ti{position:absolute;right:14px;top:14px;width:34px;height:34px;border-radius:10px;display:grid;place-items:center;background:var(--glass2)}
.tl{display:grid;grid-template-columns:52px 1fr;gap:14px;align-items:center}
.tl .n{width:52px;height:52px;border-radius:16px;display:grid;place-items:center;font-size:1.4rem;background:var(--glass2);border:1px solid var(--line);position:relative}
.tl .n::after{content:attr(data-n);position:absolute;top:-6px;right:-6px;width:20px;height:20px;border-radius:50%;background:var(--grad);color:#fff;font-size:.68rem;font-weight:700;display:grid;place-items:center}
.tl h4{margin:0 0 2px;padding:0;font-size:1rem}.tl p{color:var(--mute);font-size:.9rem;margin:0}
div[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:14px 18px;animation:up .55s var(--ease) both}
[data-testid="stMetricValue"]{font-family:'Space Grotesk';font-weight:700}
[data-testid="stForm"]{background:var(--glass);border:1px solid var(--line);border-radius:22px;padding:24px;animation:up .55s var(--ease) both}
[data-testid="stFormSubmitButton"] button,.stDownloadButton button{border-radius:14px!important}
[data-testid="stFormSubmitButton"] button{background:var(--grad)!important;border:none!important;color:#fff!important;font-weight:600;transition:transform .25s var(--spring),box-shadow .25s!important}
[data-testid="stFormSubmitButton"] button:hover{transform:translateY(-2px);box-shadow:0 12px 30px -10px rgba(139,92,246,1)}
[data-testid="stExpander"] details{border-radius:14px;border:1px solid var(--line);background:var(--glass)}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:var(--line2);border-radius:8px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = [
    ("💼", "Career", "Should I accept a startup job offer? It pays 20% more but my current job is stable, and I have 2 weeks to decide."),
    ("💰", "Money", "I'm thinking of putting half my savings into crypto. Last time I lost money. Should I?"),
    ("🎓", "Education", "Should I do a master's abroad or keep working and study part-time online?"),
    ("🏠", "Life", "Should I move to a new city for a better job, away from my family and friends?"),
]
VCLASS = {"Go for it": "go", "Proceed with caution": "mid", "Hold off": "stop"}
VICON = {"go": "✅", "mid": "⚠️", "stop": "⛔"}
VCOLOR = {"go": ("#10b981", "#06b6d4", "rgba(16,185,129,.28)"), "mid": ("#f59e0b", "#f97316", "rgba(245,158,11,.25)"),
          "stop": ("#f43f5e", "#ec4899", "rgba(244,63,94,.25)")}
THINKING = ('<div class="thinking">' + "".join(
    f'<div class="tk" style="--d:{d}s;--e:{n}s"><span class="ic"></span><span class="lbl">{t}</span></div>'
    for t, d, n in [("Understanding your situation", 0, 1.6), ("Identifying your real options", 1.6, 3.4),
                    ("Scoring outcomes &amp; risks", 3.4, 5.4), ("Checking what past-you learned", 5.4, 7.2),
                    ("Writing my advice", 7.2, 999)]) + "</div>")


def H(html: str, where=st):
    where.markdown(html, unsafe_allow_html=True)


# =================================================================== storage
def load_chats() -> dict:
    try:
        return json.loads(CHATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_chats():
    try:
        CHATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CHATS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.session_state.chats, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CHATS_PATH)
    except Exception:
        pass  # history is a convenience; never crash the chat over it


def ss_init():
    ss = st.session_state
    if "chats" not in ss:
        ss.chats = load_chats()
    ss.setdefault("cur", None)
    ss.setdefault("page", "💬 Chat")


def current() -> dict | None:
    return st.session_state.chats.get(st.session_state.cur)


@st.cache_resource(show_spinner=False)
def get_engine(version: float) -> Engine:
    return Engine(store.rows())


def engine() -> Engine:
    store.ensure_data()
    return get_engine(config.DATA_PATH.stat().st_mtime)


@st.cache_data(show_spinner=False)
def backtest(version: float) -> dict:
    return engine().backtest()


# =================================================================== decision logic
def past_me() -> str:
    """The user's own logged decisions are the voice of 'past-me' for the AI."""
    mine = [r for r in engine().rows if str(r["decision_id"]).startswith("U")][-12:]
    return "\n".join(f"- {str(r['date'])[:10]} | {r['decision_type']} | {r['context']} | did: "
                     f"{CHOICE_LABELS.get(r['choice'], r['choice'])} | expected {r['expected_result']}, got "
                     f"{r['actual_result']} | satisfaction {r['satisfaction']}/10" for r in mine)


def features(src: dict) -> dict:
    lean = src.get("leaning_category")
    f = {"assumptions": [], "decision_type": src.get("decision_type") if src.get("decision_type") in DECISION_TYPES else "other",
         "choice": lean if lean in CHOICES else (src.get("choice") if src.get("choice") in CHOICES else None)}
    for field, opts in {"previous_experience": EXPERIENCE, "risk_level": RISK, "time_available": TIME}.items():
        v = src.get(field)
        if v not in opts:
            v = DEFAULTS[field]
            f["assumptions"].append(f"{field.replace('_', ' ')} = {v}")
        f[field] = v
    return f


def data_check(f: dict, k: int):
    eng = engine()
    args = (f["decision_type"], f["previous_experience"], f["risk_level"], f["time_available"])
    cmp = eng.compare(*args, k=k)
    best = eng.best_choice(cmp)
    fc = eng.forecast(f["decision_type"], f.get("choice") or best, *args[1:], k=k)
    return cmp, best, fc


def reference(adv: dict, k: int) -> dict | None:
    """What the (simulated) historical dataset says: a cross-check only."""
    try:
        cmp, best, fc = data_check(features(adv), k)
        return {"best": CHOICE_LABELS[best], "p": round(fc.p * 100), "confidence": fc.confidence, "n": fc.n_used,
                "options": {CHOICE_LABELS[c]: round(v.p * 100) for c, v in sorted(cmp.items(), key=lambda x: -x[1].p)}}
    except Exception:
        return None


def offline_analysis(f: dict, k: int) -> tuple[dict, str]:
    """No AI: turn the dataset forecast into the same card shape the AI produces."""
    cmp, best, fc = data_check(f, k)
    verdict = {"Favourable": "Go for it", "Uncertain": "Proceed with caution"}.get(fc.verdict, "Hold off")
    tips = [t.replace("*", "") for t in explain._tips(f["previous_experience"], f["risk_level"],
                                                       f["time_available"], f.get("choice"))]
    card = {"mode": "data", "is_decision": True, "stage": "analyze", "title": f"{f['decision_type'].title()} decision",
            "decision_type": f["decision_type"], "verdict": verdict, "recommendation": CHOICE_LABELS[best],
            "rec_score": round(cmp[best].p * 100), "confidence": round(fc.conf_score * 100),
            "leaning": CHOICE_LABELS[f["choice"]] if f.get("choice") else None, "leaning_category": f.get("choice"),
            "previous_experience": f["previous_experience"], "risk_level": f["risk_level"], "time_available": f["time_available"],
            "options": [{"name": CHOICE_LABELS[c], "score": round(v.p * 100), "risk": f["risk_level"],
                         "reversible": c != "act_now", "pros": [], "cons": []}
                        for c, v in sorted(cmp.items(), key=lambda x: -x[1].p)],
            "factors": [], "risks": [], "next_steps": tips, "questions": [], "assumptions": f["assumptions"],
            "past_me": "", "future_me": "", "follow_ups": ["What if I wait a few weeks?", "Why is that the best option?"],
            "ref": None, "summary": explain.rule_summary(fc), "reasoning": ""}
    text = ("> ⚡ **Offline mode**: the AI is unavailable, so this forecast comes from the historical dataset.\n\n"
            + explain.build(f, fc, cmp, best, f["assumptions"]))
    return card, text


def brief(a: dict) -> str:
    keys = ("title", "verdict", "recommendation", "confidence", "summary", "next_steps", "assumptions")
    return json.dumps({k: a.get(k) for k in keys} | {"option_scores": {o["name"]: o["score"] for o in a.get("options", [])}})


def typed(text: str):
    for i in range(0, len(text), 5):
        yield text[i:i + 5]
        time.sleep(0.006)


def ai_stream(conversation: list[dict], adv: dict, last_txt: str):
    got = False
    for chunk in llm.stream_reply(conversation, adv, last_txt):
        got = True
        yield chunk
    if not got:  # streaming failed after a good analysis: answer from the analysis itself
        yield from typed(f"**Bottom line:** {adv.get('summary') or 'See the analysis below.'}\n\n{adv.get('reasoning', '')}"
                         if adv.get("is_decision") else "Sorry, I couldn't reach the AI just now. Please try again.")


def run_turn(text: str, use_ai: bool, k: int):
    ss = st.session_state
    if not current():
        cid = uuid.uuid4().hex[:10]
        ss.chats[cid] = {"title": text.strip()[:48], "created": time.time(), "updated": time.time(), "messages": []}
        ss.cur = cid
    conv = current()
    history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
    last = next((m["analysis"] for m in reversed(conv["messages"]) if m.get("analysis")), None)
    last_txt = brief(last) if last else ""
    conv["messages"].append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        ph = st.empty()
        H(THINKING, ph)
        card, follow, notice = None, [], ""
        try:
            adv = llm.advise(text, history, past_me(), last_txt) if use_ai and llm.available() else None
            if adv is not None:
                if adv["is_decision"] and adv["stage"] == "analyze":
                    card = {"mode": "ai", **adv, "ref": reference(adv, k)}
                follow = adv.get("follow_ups", [])
                gen = ai_stream(history + [{"role": "user", "content": text}], adv, last_txt)
            else:
                if use_ai and llm.LAST_ERROR:
                    notice = f"AI unavailable ({llm.LAST_ERROR[:80]}), used the offline engine."
                f = understand(text, history, use_llm=False)
                if f["is_decision"]:
                    card, txt = offline_analysis(f, k)
                    follow = card["follow_ups"]
                    gen = typed(txt)
                else:
                    gen = typed("I'm in **offline mode** right now, so I can only analyse decisions. Tell me what "
                                "you're deciding, your options, how risky it feels, how much time you have, and any "
                                "similar past experience.")
            ph.empty()
            full = st.write_stream(gen)
        except Exception as ex:  # never show a crash to the user
            ph.empty()
            full = f"Sorry, something went wrong ({type(ex).__name__}). Please try again or rephrase."
            st.markdown(full)
    conv["messages"].append({"role": "assistant", "content": full if isinstance(full, str) else "".join(map(str, full)),
                             "analysis": card, "follow_ups": follow, "notice": notice})
    if card and not conv.get("titled"):
        conv["title"], conv["titled"] = card["title"], True
    conv["updated"] = time.time()
    save_chats()
    st.rerun()


# =================================================================== rendering
def gauge(value: float, label: str, key: str, vc: str) -> str:
    v = int(round(max(0, min(100, value))))
    c1, c2, glow = VCOLOR[vc]
    return (f'<div class="gauge" style="--gc:{glow}"><svg viewBox="0 0 120 120"><defs><linearGradient id="g{key}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient></defs>'
            f'<circle class="trk" cx="60" cy="60" r="52"/><circle class="arc" cx="60" cy="60" r="52" stroke="url(#g{key})" '
            f'style="--off:{327 - 327 * v / 100:.1f}"/></svg><div class="gv"><b><span class="count" style="--t:{v}"></span><i>%</i></b>'
            f'<span>{label}</span></div></div>')


def li(text: str, kind: str, sym: str, delay: float = 0) -> str:
    return f'<div class="li" style="animation-delay:{delay:.2f}s"><span class="b {kind}">{sym}</span><span>{e(text)}</span></div>'


def render_card(a: dict, key: str):
    vc = VCLASS.get(a["verdict"], "mid")
    rec = next((o for o in a["options"] if o["name"] == a["recommendation"]), a["options"][0])
    mode = ('<span class="badge ghost">✨ AI analysis</span>' if a.get("mode") == "ai"
            else '<span class="badge ghost">📚 Offline data engine</span>')
    lean = (f'<div class="stat" style="animation-delay:.55s"><small>Your leaning</small><b>{e(str(a["leaning"]))}</b></div>'
            if a.get("leaning") else "")
    bars = "".join(
        f'<div class="opt" style="animation-delay:{.15 + i * .1:.2f}s"><div class="opt-h"><span class="n">{e(o["name"])}'
        + (' <span class="pill star">★ best</span>' if o["name"] == a["recommendation"] else "")
        + f' <span class="pill {o["risk"]}">{o["risk"]} risk</span></span><span class="s">{o["score"]}%</span></div>'
        f'<div class="track"><div class="fill{" best" if o["name"] == a["recommendation"] else ""}" '
        f'style="--w:{o["score"]}%;animation-delay:{.25 + i * .12:.2f}s"></div></div></div>'
        for i, o in enumerate(a["options"]))
    H(f'<div class="dcard"><div class="dc-head"><span class="dc-title">{TYPE_ICONS.get(a.get("decision_type"), "🧩")} '
      f'{e(a["title"])}</span>{mode}<span class="badge {vc}">{VICON[vc]} {e(a["verdict"])}</span></div>'
      f'<div class="dc-hero">{gauge(a["rec_score"], "success odds", key, vc)}<div>'
      f'<div class="eyebrow">Recommended move</div><div class="rec">{e(a["recommendation"])}</div><div class="stats">'
      f'<div class="stat" style="animation-delay:.4s"><small>Confidence</small><b>{a["confidence"]}%</b>'
      f'<div class="meter"><div style="--w:{a["confidence"]}%"></div></div></div>'
      f'<div class="stat" style="animation-delay:.45s"><small>Risk</small><b>{rec["risk"].title()}</b></div>'
      f'<div class="stat" style="animation-delay:.5s"><small>Reversible</small><b>{"Yes" if rec["reversible"] else "Hard to undo"}</b></div>'
      f'{lean}</div></div></div><div class="sec-h">⚖️ Options compared · chance of a good outcome</div>{bars}</div>')

    names = ["➕➖ Pros & cons", "🎯 Key factors", "🛡️ Risks", "🚀 Next steps", "🔮 Past & future you", "📚 Data check"]
    has = [any(o["pros"] or o["cons"] for o in a["options"]), bool(a["factors"]), bool(a["risks"]), bool(a["next_steps"]),
           bool(a.get("past_me") or a.get("future_me")), bool(a.get("ref"))]
    shown = [n for n, h in zip(names, has) if h]
    if not shown:
        return
    tabs = dict(zip(shown, st.tabs(shown)))
    if names[0] in tabs:
        with tabs[names[0]]:
            H("".join(f'<div class="pc-grid"><div class="pc-name">{e(o["name"])} <span class="pill">{o["score"]}%</span></div>'
                      f'<div class="pc pro">{"".join(li(p, "p", "+", j * .05) for j, p in enumerate(o["pros"])) or "<span class=muted>—</span>"}</div>'
                      f'<div class="pc con">{"".join(li(c, "c", "−", j * .05) for j, c in enumerate(o["cons"])) or "<span class=muted>—</span>"}</div></div>'
                      for o in a["options"]))
    if names[1] in tabs:
        with tabs[names[1]]:
            H("".join(f'<div class="fac" style="animation-delay:{i * .06:.2f}s"><div><b>{e(f["name"])}</b>'
                      f'<div class="nt">{e(f["note"])}</div></div><div class="wd">'
                      + "".join(f'<i class="{"on " if j < f["weight"] else ""}{f["impact"]}"></i>' for j in range(5))
                      + "</div></div>" for i, f in enumerate(a["factors"])))
    if names[2] in tabs:
        with tabs[names[2]]:
            H("".join(f'<div class="risk" style="animation-delay:{i * .07:.2f}s"><b>⚠️ {e(r["risk"])}</b>'
                      f'<span>🛡️ {e(r["mitigation"])}</span></div>' for i, r in enumerate(a["risks"])))
    if names[3] in tabs:
        with tabs[names[3]]:
            H("".join(f'<div class="st" style="animation-delay:{i * .08:.2f}s"><div class="num">{i + 1}</div>'
                      f'<div class="tx">{e(s)}</div></div>' for i, s in enumerate(a["next_steps"])))
    if names[4] in tabs:
        with tabs[names[4]]:
            H(f'<div class="eyebrow">🕰️ Past-me says</div><div class="quote">{e(a.get("past_me") or "—")}</div>'
              f'<div class="eyebrow">🔮 Future-me, 5 years on</div><div class="quote f">{e(a.get("future_me") or "—")}</div>')
    if names[5] in tabs:
        with tabs[names[5]]:
            r = a["ref"]
            H(f'<div class="muted" style="margin-bottom:8px">Cross-check against {r["n"]} similar decisions in the bundled '
              f'<b>simulated</b> dataset (reference only). Best generic approach there: <b>{e(r["best"])}</b>, '
              f'confidence {e(r["confidence"])}.</div>'
              + "".join(f'<div class="ref-row"><span>{e(n)}</span><b>{p}%</b></div>' for n, p in r["options"].items()))
    if a.get("assumptions"):
        st.caption("🧩 Assumed: " + "; ".join(a["assumptions"]))


def export_md(conv: dict) -> str:
    out = [f"# {conv['title']}", f"_Exported from Past-Me AI · {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in conv["messages"]:
        out += [f"**{'You' if m['role'] == 'user' else 'Past-Me'}:**", "", m["content"], ""]
        a = m.get("analysis")
        if a:
            out += [f"> **{a['verdict']}** → {a['recommendation']} ({a['rec_score']}% success, {a['confidence']}% confidence)", ""]
            out += [f"- {o['name']}: {o['score']}%" for o in a["options"]] + [""]
    return "\n".join(out)


def to_log(a: dict):
    st.session_state.prefill = {"context": a["title"][:120], "decision_type": features(a)["decision_type"],
                                "choice": a.get("leaning_category") if a.get("leaning_category") in CHOICES else "act_now",
                                **{k: v for k, v in features(a).items() if k in ("previous_experience", "risk_level", "time_available")}}
    st.session_state.page = "📝 Log outcome"


def regenerate():
    msgs = current()["messages"]
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
        msgs.pop()
        st.session_state.pending = msgs.pop()["content"]


def render_message(m: dict, i: int, last: bool):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            return
        if m.get("analysis"):
            render_card(m["analysis"], f"m{st.session_state.cur}{i}")
        if m.get("notice"):
            H(f'<div class="notice">⚠️ {e(m["notice"])}</div>')
        if last:
            if m.get("follow_ups"):
                cols = st.columns(len(m["follow_ups"]))
                for j, (c, q) in enumerate(zip(cols, m["follow_ups"])):
                    if c.button(q, key=f"fu_{i}_{j}", width="stretch"):
                        st.session_state.pending = q
                        st.rerun()
            a1, a2, a3, _ = st.columns([1.1, 1.3, 1.1, 3])
            if a1.button("🔁 Regenerate", key=f"act_regen_{i}"):
                regenerate()
                st.rerun()
            if m.get("analysis") and a2.button("📝 Log this decision", key=f"act_log_{i}"):
                to_log(m["analysis"])
                st.rerun()
            a3.download_button("⤓ Export", export_md(current()), f"past-me-{st.session_state.cur}.md",
                               "text/markdown", key=f"act_exp_{i}")


# =================================================================== pages
def welcome():
    H('<div class="welcome"><div class="orb"><div class="ring"></div><div class="ring r2"></div><div class="sat"></div>'
      '<div class="core">🧭</div></div><h1>What decision are you <span class="grad-text">facing today?</span></h1>'
      '<p>Tell me your situation like you would a trusted mentor. I\'ll weigh every option, score the odds, '
      'expose the hidden risks and tell you exactly what I\'d do.</p></div>')
    cols = st.columns(2)
    for i, (ic, cat, q) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{ic} **{cat}**  \n{q}", key=f"sg{i}", width="stretch"):
            st.session_state.pending = q
            st.rerun()
    H('<div class="caps">' + "".join(f'<span class="cap" style="animation-delay:{.3 + i * .08:.2f}s">{c}</span>' for i, c in enumerate(
        ["⚖️ Scores every option", "❓ Asks smart questions", "🛡️ Spots hidden risks", "🕰️ Learns from your past",
         "⚡ Streams answers live"])) + "</div>")


def page_chat(use_ai: bool, k: int):
    conv = current()
    msgs = conv["messages"] if conv else []
    if msgs:
        l, r = st.columns([5, 1.6])
        H(f'<div class="topbar"><span class="tb-title">{e(conv["title"])}</span></div>', l)
        pill = (f'<span class="pill-model"><span class="dot on"></span>{e(config.GROQ_MODEL.split("/")[-1])}</span>' if use_ai
                else '<span class="pill-model"><span class="dot"></span>offline engine</span>')
        H(f'<div style="text-align:right">{pill}</div>', r)
    else:
        welcome()
    for i, m in enumerate(msgs):
        render_message(m, i, i == len(msgs) - 1)

    text = st.chat_input("Describe a decision you're facing…")
    pending = st.session_state.pop("pending", None)
    text = text or pending
    if text and text.strip():
        run_turn(text.strip(), use_ai, k)


def page_log():
    H('<div class="page-head"><div class="eyebrow">Teach past-me</div><h1>Log a real outcome</h1><p>Record what actually '
      'happened after a decision. The AI reads your logged outcomes as <b>past-me</b> memory, and they count double '
      'in the data engine.</p></div>')
    pf = st.session_state.pop("prefill", None) or {}
    if pf:
        st.info("✨ Pre-filled from your last analysis. Adjust anything that's off.")

    def idx(opts, key, default=0):
        return opts.index(pf[key]) if pf.get(key) in opts else default

    with st.form("log", clear_on_submit=False):
        ctx = st.text_input("What did you decide?", value=pf.get("context", ""), max_chars=120,
                            placeholder="e.g. Accepted the offer at Acme")
        c1, c2 = st.columns(2)
        dtype = c1.selectbox("Decision type", DECISION_TYPES, idx(DECISION_TYPES, "decision_type"),
                             format_func=lambda x: f"{TYPE_ICONS[x]} {x.title()}")
        choice = c2.selectbox("What you did", CHOICES, idx(CHOICES, "choice"), format_func=CHOICE_LABELS.get)
        c3, c4, c5 = st.columns(3)
        exp = c3.selectbox("Past experience", EXPERIENCE, idx(EXPERIENCE, "previous_experience"))
        risk = c4.selectbox("Risk level", RISK, idx(RISK, "risk_level", 1))
        tm = c5.selectbox("Time you had", TIME, idx(TIME, "time_available", 2))
        c6, c7 = st.columns(2)
        exp_res = c6.selectbox("What you expected", RESULTS, index=2)
        act_res = c7.selectbox("What actually happened", RESULTS, index=2)
        c8, c9 = st.columns(2)
        sat = c8.slider("Satisfaction (1–10)", 1, 10, 7)
        score = c9.slider("Overall outcome score (0–100)", 0, 100, 70)
        if st.form_submit_button("💾  Save outcome", width="stretch"):
            if not ctx.strip():
                st.error("Please describe the decision.")
            else:
                did = store.append_decision({
                    "decision_type": dtype, "context": ctx.strip(), "choice": choice, "previous_experience": exp,
                    "risk_level": risk, "time_available": tm, "expected_result": exp_res, "actual_result": act_res,
                    "satisfaction": sat, "outcome_score": score, "date": date.today().isoformat()})
                st.success(f"Saved as {did}. Past-me will remember this. 🎉")
                st.balloons()


def chart(data: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(data).encode(
        y=alt.Y("group_name:N", sort="-x", title=None, axis=alt.Axis(labelColor="#c8c8da", labelFontSize=12, domain=False, ticks=False)),
        x=alt.X("success_pct:Q", title="went well %", scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(labelColor="#8b8ba3", titleColor="#8b8ba3", gridColor="#ffffff12", domain=False)))
    bars = base.mark_bar(cornerRadiusEnd=8, height=16).encode(
        color=alt.Color("success_pct:Q", legend=None, scale=alt.Scale(domain=[30, 50, 75], range=["#f43f5e", "#f59e0b", "#10b981"])),
        tooltip=["group_name", "decisions", "avg_score", "success_pct"])
    text = base.mark_text(align="left", dx=6, color="#ededf5", fontWeight=600).encode(text=alt.Text("success_pct:Q", format=".0f"))
    return (bars + text).properties(height=max(150, 34 * len(data))).configure_view(strokeWidth=0).configure(background="transparent")


def page_insights():
    H('<div class="page-head"><div class="eyebrow">Your patterns</div><h1>Insights</h1><p>What the recorded decisions '
      'say about which choices tend to work out. The bundled rows are a simulated sample; your logged ones are real.</p></div>')
    eng = engine()
    rows = eng.rows
    mine = [r for r in rows if str(r["decision_id"]).startswith("U")]
    good = round(100 * sum(r["outcome_score"] >= 60 for r in rows) / len(rows))
    tiles = [("📚", "Decisions recorded", len(rows)), ("🙋", "Logged by you", len(mine)), ("✅", "Went well", f"{good}%"),
             ("💬", "Your conversations", len(st.session_state.chats))]
    H('<div class="tiles">' + "".join(f'<div class="card tile" style="animation-delay:{i * .07:.2f}s"><span class="ti">{ic}</span>'
                                      f'<small>{t}</small><b>{v}</b></div>' for i, (ic, t, v) in enumerate(tiles)) + "</div>")
    with st.expander("🧪 How accurate is the offline data engine? (live leave-one-out test)"):
        bt = backtest(config.DATA_PATH.stat().st_mtime)
        x, y, z = st.columns(3)
        x.metric("Accuracy", f"{bt['accuracy'] * 100:.0f}%", f"{(bt['accuracy'] - bt['baseline']) * 100:+.0f} pts vs majority guess")
        y.metric("Majority-guess baseline", f"{bt['baseline'] * 100:.0f}%")
        z.metric("Brier score", f"{bt['brier']:.3f}", "lower is better", delta_color="off")
    groups = [("decision_type", "🧩 Decision type"), ("choice", "🎯 Choice made"), ("risk_level", "🔥 Risk level"),
              ("previous_experience", "🕰️ Past experience")]
    for tab, (col, _) in zip(st.tabs([t for _, t in groups]), groups):
        with tab:
            data = pd.DataFrame(stats.summary_by(rows, col))
            st.altair_chart(chart(data), width="stretch")
    if mine:
        H('<div class="sec-h">🙋 Your logged outcomes</div>')
        st.dataframe(pd.DataFrame(mine[::-1])[["decision_id", "date", "decision_type", "context", "actual_result",
                                               "satisfaction", "outcome_score"]], hide_index=True, width="stretch")
    st.download_button("⬇️  Download decisions.csv", config.DATA_PATH.read_bytes(), "decisions.csv", "text/csv")


def page_about():
    H('<div class="page-head"><div class="eyebrow">Under the hood</div><h1>How Past-Me thinks</h1><p>Two passes on every '
      'message: a deep structured analysis first, then a conversational answer grounded in it, so the words and the '
      'scores always agree.</p></div>')
    steps = [("🧠", "Understand", "Reads the whole conversation and identifies your real options, in your own words."),
             ("❓", "Clarify", "If the decision is too vague to judge responsibly, it asks 2–4 targeted questions first."),
             ("⚖️", "Score", "Each option gets a calibrated 0–100 chance of a good outcome, weighing money, downside, "
                            "reversibility, time pressure and second-order effects."),
             ("🕰️", "Ask past-me", "Outcomes you save on <b>Log outcome</b> are fed back, so the advice learns from your own track record."),
             ("💬", "Explain", "The answer streams live: bottom line, why, how to play it, and what to watch out for."),
             ("📚", "Cross-check", "A similarity search + Bayesian model over the dataset is shown as a reference and "
                                  "takes over automatically if the AI is offline.")]
    H("".join(f'<div class="card tl" style="animation-delay:{i * .07:.2f}s"><div class="n" data-n="{i + 1}">{ic}</div>'
              f'<div><h4>{t}</h4><p>{d}</p></div></div>' for i, (ic, t, d) in enumerate(steps)))
    H('<div class="card"><h4 style="margin:0 0 6px;padding:0">💡 Tips for sharper advice</h4><div class="muted">'
      'Mention your options, what\'s at stake (money, time, people), your deadline and what happened last time. '
      'Ask "what if…" to test scenarios. This is a thinking aid, not professional advice; for medical, legal or major '
      'financial decisions, talk to a qualified expert.</div></div>')


# =================================================================== main
ss_init()
ss = st.session_state
with st.sidebar:
    H('<div class="brand"><div class="logo">🧭</div><div><b>Past-Me</b><small>AI Decision Coach</small></div></div>')
    if st.button("✚  New decision", key="newchat", width="stretch"):
        ss.cur, ss.page = None, "💬 Chat"
        st.rerun()
    pages = ["💬 Chat", "📊 Insights", "📝 Log outcome", "ℹ️ How it works"]
    ss.page = st.radio("Navigate", pages, index=pages.index(ss.page), label_visibility="collapsed")
    chats = sorted(ss.chats.items(), key=lambda kv: -kv[1]["updated"])[:30]
    if chats:
        H('<div class="side-label">Recent</div>')
        for cid, c in chats:
            first = next((m["analysis"] for m in c["messages"] if m.get("analysis")), None)
            icon = TYPE_ICONS.get(first.get("decision_type"), "💬") if first else "💬"
            if st.button(f"{icon} {c['title']}", key=f"hist_{cid}", width="stretch",
                         type="primary" if cid == ss.cur else "secondary"):
                ss.cur, ss.page = cid, "💬 Chat"
                st.rerun()
        if ss.cur in ss.chats and st.button("🗑️ Delete this chat", key="act_delete", width="stretch"):
            ss.chats.pop(ss.cur)
            ss.cur = None
            save_chats()
            st.rerun()
    st.divider()
    use_ai = st.toggle("✨ AI reasoning", value=llm.available(), disabled=not llm.available(),
                       help="Off = offline data engine only.")
    H(f'<div class="status"><span class="dot{" on" if use_ai else ""}"></span>'
      + (f"Groq · {e(config.GROQ_MODEL)}" if use_ai else ("Offline engine" if llm.available() else "No API key · offline engine"))
      + "</div>")
    with st.expander("⚙️ Advanced"):
        k = st.slider("Similar decisions (data engine)", 20, 120, 60, 10)

{"💬 Chat": lambda: page_chat(use_ai, k), "📊 Insights": page_insights, "📝 Log outcome": page_log,
 "ℹ️ How it works": page_about}[ss.page]()
