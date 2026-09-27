"""Bundle a ride trace (runs/ride.py --replay / --oracle / --open-loop) into one self-contained HTML replay.

usage: python -m export.export_ride results/ride_trace.json results/ride_view.html
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>🪰 Fly rides a Tarmac SL9</title>
<style>
:root{--bg:#0f1115;--fg:#e8e8e8;--dim:#8a8f98;--acc:#ff4d4d;--acc2:#4da3ff;--ok:#5bd66b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #222}
header b{font-size:16px} header .mode{color:var(--dim)}
button,select,input[type=range]{background:#1b1f27;color:var(--fg);border:1px solid #333;border-radius:6px;padding:4px 10px}
main{display:grid;grid-template-columns:420px 1fr;gap:12px;padding:12px 16px}
canvas{background:#141821;border-radius:10px;width:100%;display:block}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px}
.stat{background:#141821;border-radius:8px;padding:8px 10px}.stat small{color:var(--dim);display:block}
.stat span{font-size:18px;font-variant-numeric:tabular-nums}
#charts canvas{margin-bottom:8px}
</style></head><body>
<header><b>🪰🚴 Fly brain × S-Works Tarmac SL9</b><span class="mode" id="mode"></span>
<button id="play">▶ 播放</button>
<label>速度 <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select>×</label>
<label>骑手 <select id="rider"></select></label>
<input id="time" type="range" min="0" value="0" style="width:260px"><span id="clock">0.00 s</span></header>
<main>
<div><canvas id="rear" width="420" height="420"></canvas>
<div class="stats">
<div class="stat"><small>倾角 lean</small><span id="s_phi"></span></div>
<div class="stat"><small>把角 steer</small><span id="s_delta"></span></div>
<div class="stat"><small>速度</small><span id="s_v"></span></div>
<div class="stat"><small>转向扭矩</small><span id="s_T"></span></div>
<div class="stat"><small>踩踏功率</small><span id="s_P"></span></div>
<div class="stat"><small>全脑 spikes/s</small><span id="s_pop"></span></div>
<div class="stat"><small>状态</small><span id="s_done"></span></div>
<div class="stat"><small>直立骑手</small><span id="s_alive"></span></div>
</div></div>
<div><canvas id="top" width="900" height="260"></canvas><div id="charts"></div></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);const tr=D.trace;const B=tr[0].state.length;
const dn=D.dn_names||[];const hasDN=!!tr[0].dn_hz;
document.getElementById('mode').textContent=D.mode+' · '+B+' riders · '+tr[tr.length-1].t+' s';
const rsel=document.getElementById('rider');for(let i=0;i<B;i++){const o=document.createElement('option');o.value=i;o.textContent='#'+i;rsel.appendChild(o)}
const slider=document.getElementById('time');slider.max=tr.length-1;
let k=0,playing=false,rider=0,last=0;
const deg=r=>r*180/Math.PI;
// ---- rear view
const rc=document.getElementById('rear').getContext('2d');
function drawRear(row){const W=420,H=420;rc.clearRect(0,0,W,H);const s=row.state[rider];const phi=s[4],delta=s[5],done=row.done[rider];
 rc.strokeStyle='#2a3040';rc.lineWidth=1;rc.beginPath();rc.moveTo(0,330);rc.lineTo(W,330);rc.stroke();
 rc.save();rc.translate(W/2,330);rc.rotate(phi);const sc=230;
 // wheel (seen from behind: thin ellipse), frame, saddle, bars
 rc.strokeStyle=done?'#666':'#e8e8e8';rc.lineWidth=6;rc.beginPath();rc.ellipse(0,-0.336*sc,0.03*sc,0.336*sc,0,0,Math.PI*2);rc.stroke();
 rc.lineWidth=5;rc.strokeStyle=done?'#666':'#ff4d4d';rc.beginPath();rc.moveTo(0,-0.336*sc);rc.lineTo(0,-1.0*sc);rc.stroke();
 rc.beginPath();rc.moveTo(-0.07*sc,-1.0*sc);rc.lineTo(0.07*sc,-1.0*sc);rc.stroke();
 // handlebars rotate with steer: seen from behind, width foreshortens and one end comes forward
 const bw=0.21*sc*Math.cos(delta),dy=0.21*sc*Math.sin(delta)*0.35;rc.strokeStyle=done?'#666':'#4da3ff';rc.lineWidth=5;
 rc.beginPath();rc.moveTo(0,-1.05*sc);rc.lineTo(0,-1.12*sc);rc.stroke();
 rc.beginPath();rc.moveTo(-bw,-1.12*sc-dy);rc.lineTo(bw,-1.12*sc+dy);rc.stroke();
 // rider: legs, torso, head, and the fly on top
 rc.strokeStyle=done?'#666':'#f5c26b';rc.lineWidth=7;rc.beginPath();rc.moveTo(-0.08*sc,-0.9*sc);rc.lineTo(0,-1.0*sc);rc.lineTo(0.08*sc,-0.9*sc);rc.stroke();
 rc.beginPath();rc.moveTo(0,-1.0*sc);rc.lineTo(0,-1.45*sc);rc.stroke();rc.beginPath();rc.moveTo(-bw,-1.12*sc-dy);rc.lineTo(0,-1.38*sc);rc.lineTo(bw,-1.12*sc+dy);rc.stroke();
 rc.fillStyle=done?'#666':'#f5c26b';rc.beginPath();rc.arc(0,-1.55*sc,0.09*sc,0,Math.PI*2);rc.fill();
 rc.font='34px serif';rc.textAlign='center';rc.fillText('🪰',0,-1.7*sc);
 rc.restore();
 if(done){rc.fillStyle='#ff4d4d';rc.font='bold 22px system-ui';rc.textAlign='center';rc.fillText('倒了 / 跳车',W/2,60)}
 rc.fillStyle='#8a8f98';rc.font='12px system-ui';rc.textAlign='left';rc.fillText('rear view · 倾角 '+deg(phi).toFixed(1)+'° 把角 '+deg(delta).toFixed(1)+'°',10,18);}
// ---- top view
const tc=document.getElementById('top').getContext('2d');
function drawTop(kk){const W=900,H=260;tc.clearRect(0,0,W,H);const xmax=Math.max(20,...tr.map(r=>Math.max(...r.state.map(s=>s[0]))));const sx=(W-40)/xmax,sy=(H-20)/8;
 tc.strokeStyle='#2a3040';tc.setLineDash([6,6]);for(const y of[-1.5,1.5]){tc.beginPath();tc.moveTo(20,H/2+y*sy);tc.lineTo(W-20,H/2+y*sy);tc.stroke()}tc.setLineDash([]);
 for(let b=0;b<B;b++){tc.strokeStyle=b===rider?'#ff4d4d':'rgba(120,140,170,0.35)';tc.lineWidth=b===rider?2.5:1;tc.beginPath();
  for(let i=0;i<=kk;i++){const s=tr[i].state[b];const px=20+s[0]*sx,py=H/2+s[1]*sy;i?tc.lineTo(px,py):tc.moveTo(px,py)}tc.stroke();
  const s=tr[kk].state[b];tc.fillStyle=tr[kk].done[b]?'#666':(b===rider?'#ff4d4d':'#4da3ff');tc.beginPath();tc.arc(20+s[0]*sx,H/2+s[1]*sy,b===rider?5:3,0,Math.PI*2);tc.fill()}
 tc.fillStyle='#8a8f98';tc.font='12px system-ui';tc.fillText('top view · x 0…'+xmax.toFixed(0)+' m, lane ±1.5 m',10,14);}
// ---- strip charts
const series=[{name:'倾角 lean (°)',f:(r,b)=>deg(r.state[b][4]),c:'#ff4d4d',lim:[-30,30]},
 {name:'转向扭矩 steer torque (Nm)',f:(r,b)=>r.steer[b],c:'#4da3ff',lim:[-6.5,6.5]},
 {name:'速度 v (m/s)',f:(r,b)=>r.state[b][3],c:'#5bd66b',lim:[0,10]}];
if(hasDN){const pairs=[['DNp20_L','DNp20_R'],['DNg46_L','DNg46_R'],['DNp22_L','DNp22_R'],['b1 MN_L','b1 MN_R']];
 for(const [l,r] of pairs){const il=dn.indexOf(l),ir=dn.indexOf(r);if(il<0||ir<0)continue;
  series.push({name:l.replace('_L','')+'  L (blue) / R (red)  Hz',f:(row,b)=>row.dn_hz[b][il],f2:(row,b)=>row.dn_hz[b][ir],c:'#4da3ff',c2:'#ff4d4d',lim:[0,150]})}}
const charts=series.map(s=>{const c=document.createElement('canvas');c.width=900;c.height=90;document.getElementById('charts').appendChild(c);return c});
function drawCharts(kk){series.forEach((s,i)=>{const c=charts[i],g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);const n=tr.length;const X=j=>20+j*(W-40)/(n-1);const Y=v=>H-8-(v-s.lim[0])/(s.lim[1]-s.lim[0])*(H-24);
 g.strokeStyle='#2a3040';g.beginPath();g.moveTo(20,Y(0));g.lineTo(W-20,Y(0));g.stroke();
 const line=(f,col)=>{g.strokeStyle=col;g.lineWidth=1.5;g.beginPath();for(let j=0;j<n;j++){const v=f(tr[j],rider);j?g.lineTo(X(j),Y(v)):g.moveTo(X(j),Y(v))}g.stroke()};
 line(s.f,s.c);if(s.f2)line(s.f2,s.c2);
 g.strokeStyle='#e8e8e8';g.beginPath();g.moveTo(X(kk),8);g.lineTo(X(kk),H-8);g.stroke();
 g.fillStyle='#8a8f98';g.font='11px system-ui';g.fillText(s.name+'  '+s.f(tr[kk],rider).toFixed(1)+(s.f2?' / '+s.f2(tr[kk],rider).toFixed(1):''),24,12)})}
function render(){const row=tr[k];drawRear(row);drawTop(k);drawCharts(k);const s=row.state[rider];
 document.getElementById('clock').textContent=row.t.toFixed(2)+' s';slider.value=k;
 document.getElementById('s_phi').textContent=deg(s[4]).toFixed(1)+'°';document.getElementById('s_delta').textContent=deg(s[5]).toFixed(1)+'°';
 document.getElementById('s_v').textContent=(s[3]*3.6).toFixed(1)+' km/h';document.getElementById('s_T').textContent=row.steer[rider].toFixed(2)+' Nm';
 document.getElementById('s_P').textContent=row.power[rider].toFixed(0)+' W';document.getElementById('s_pop').textContent=row.pop_hz?Math.round(row.pop_hz[rider]).toLocaleString():'—';
 document.getElementById('s_done').textContent=row.done[rider]?'倒了':'骑着';document.getElementById('s_alive').textContent=row.done.filter(d=>!d).length+' / '+B;}
function tick(ts){if(playing){const sp=parseFloat(document.getElementById('speed').value);const dt=(tr[1].t-tr[0].t)*1000/sp;if(ts-last>dt){last=ts;k=Math.min(k+1,tr.length-1);if(k===tr.length-1)playing=false;render()}}requestAnimationFrame(tick)}
document.getElementById('play').onclick=()=>{playing=!playing;if(k>=tr.length-1){k=0}document.getElementById('play').textContent=playing?'⏸ 暂停':'▶ 播放'};
slider.oninput=e=>{k=+e.target.value;render()};rsel.onchange=e=>{rider=+e.target.value;render()};
render();requestAnimationFrame(tick);
</script></body></html>
"""


def main():
    src = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/ride_trace.json")
    dst = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/ride_view.html")
    data = json.loads(src.read_text())
    # keep the file small: drop the per-step sensory rates, keep DN rates
    for row in data["trace"]:
        row.pop("sense_hz", None)
    dst.write_text(HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    print(f"{dst}  ({dst.stat().st_size / 1e6:.1f} MB, {len(data['trace'])} steps, {len(data['trace'][0]['state'])} riders)")


if __name__ == "__main__":
    main()
