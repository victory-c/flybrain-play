"""A ride trace as a 3D road scene: the Tarmac and its rider lean, steer and pedal exactly as the
simulation says, with the fly on the helmet. Chase camera, side camera, or the fly's own eyes.

usage: python -m export.export_ride3d results/ride_trace.json results/ride_3d.html
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DN_SHOWN = ["DNp20_L", "DNp20_R", "DNg46_L", "DNg46_R", "DNp22_L", "DNp22_R", "b1 MN_L", "b1 MN_R"]

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>🪰 Fly rides a Tarmac SL9</title>
<style>
html,body{margin:0;height:100%;background:#0b0d12;color:#eee;font:14px/1.4 system-ui,sans-serif;overflow:hidden}
#c{position:fixed;inset:0;display:block}
.panel{position:fixed;background:rgba(10,12,18,.72);backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:10px 14px}
#hud{top:14px;right:14px;min-width:220px}
#hud .big{font-size:40px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}#hud .unit{font-size:14px;color:#9aa;margin-left:4px}
#hud .row{display:flex;justify-content:space-between;gap:16px;color:#cbd;margin-top:6px;font-variant-numeric:tabular-nums}#hud .row b{color:#fff}
#brain{top:14px;left:14px;width:230px}#brain h4{margin:0 0 6px;font-weight:600;font-size:13px;color:#9aa}
.bar{display:grid;grid-template-columns:62px 1fr 1fr;gap:6px;align-items:center;font-size:12px;margin:3px 0}
.bar i{display:block;height:9px;border-radius:5px;background:linear-gradient(90deg,#4da3ff,#8ec5ff);transform-origin:left}.bar i.r{background:linear-gradient(90deg,#ff4d4d,#ff9a9a)}
#ctl{bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:10px;align-items:center;white-space:nowrap}
button,select{background:#1b1f27;color:#eee;border:1px solid #333;border-radius:8px;padding:6px 12px;font-size:14px;cursor:pointer}button.on{background:#ff4d4d;border-color:#ff4d4d}
input[type=range]{width:300px}
#title{top:14px;left:50%;transform:translateX(-50%);color:#9aa;font-size:12px;white-space:nowrap}
#fall{position:fixed;top:40%;left:50%;transform:translateX(-50%);font-size:44px;font-weight:800;color:#ff4d4d;text-shadow:0 2px 12px #000;display:none}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud" class="panel"><div><span class="big" id="v">0.0</span><span class="unit">km/h</span></div>
<div class="row"><span>倾角 lean</span><b id="phi">0°</b></div><div class="row"><span>把角 steer</span><b id="delta">0°</b></div>
<div class="row"><span>转向扭矩</span><b id="T">0 Nm</b></div><div class="row"><span>踩踏功率</span><b id="P">0 W</b></div>
<div class="row"><span>里程</span><b id="x">0 m</b></div><div class="row"><span>直立骑手</span><b id="alive"></b></div></div>
<div id="brain" class="panel"><h4>🪰 苍蝇全脑 <span id="pop"></span> spikes/s</h4><div id="bars"></div><div style="font-size:11px;color:#778;margin-top:6px">下行神经元 左(蓝) / 右(红)，Hz</div></div>
<div id="ctl" class="panel"><button id="play">▶ 播放</button><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<input id="time" type="range" min="0" value="0"><span id="clock" style="font-variant-numeric:tabular-nums;min-width:56px">0.00 s</span>
<button id="cam0" class="on">跟拍</button><button id="cam1">侧拍</button><button id="cam2">苍蝇视角</button><button id="ghosts" class="on">其他骑手</button>
<select id="rider"></select></div>
<div id="title" class="panel"></div><div id="fall">倒了！</div>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),tr=D.trace,B=tr[0].state.length,dt=tr[1].t-tr[0].t;
const Q=new URLSearchParams(location.search);const dnNames=D.dn_names||[];const R=0.336,LAM=16.5*Math.PI/180;
document.getElementById('title').textContent=D.mode+' · '+B+' 名骑手 · '+tr[tr.length-1].t.toFixed(1)+' s · 男性果蝇全中枢神经系统 166,700 神经元';
// ---------- scene
const canvas=document.getElementById('c');const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.outputEncoding=THREE.sRGBEncoding;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x9fd0f5);scene.fog=new THREE.Fog(0x9fd0f5,60,220);
const camera=new THREE.PerspectiveCamera(60,1,0.05,400);
scene.add(new THREE.HemisphereLight(0xbfe3ff,0x4a6a2a,0.75));
const sun=new THREE.DirectionalLight(0xfff2d8,1.4);sun.position.set(-20,40,25);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);
Object.assign(sun.shadow.camera,{left:-25,right:25,top:25,bottom:-25,near:1,far:120});scene.add(sun);scene.add(sun.target);
const xmax=Math.max(60,...tr.map(r=>Math.max(...r.state.map(s=>s[0]))))+80;
const mat=(c,o={})=>new THREE.MeshStandardMaterial(Object.assign({color:c,roughness:.8,metalness:.05},o));
const ground=new THREE.Mesh(new THREE.PlaneGeometry(xmax+200,400),mat(0x5d8a3a));ground.rotation.x=-Math.PI/2;ground.position.x=xmax/2;ground.receiveShadow=true;scene.add(ground);
const road=new THREE.Mesh(new THREE.PlaneGeometry(xmax+100,7),mat(0x3a3d44,{roughness:.95}));road.rotation.x=-Math.PI/2;road.position.set(xmax/2,0.005,0);road.receiveShadow=true;scene.add(road);
const dash=new THREE.InstancedMesh(new THREE.PlaneGeometry(2,0.12),mat(0xf2f2f2),Math.ceil(xmax/6)+20);let m4=new THREE.Matrix4();
for(let i=0;i<dash.count;i++){m4.makeRotationX(-Math.PI/2);m4.setPosition(i*6-20,0.01,0);dash.setMatrixAt(i,m4)}scene.add(dash);
for(const z of[-3.2,3.2]){const e=new THREE.Mesh(new THREE.PlaneGeometry(xmax+100,0.12),mat(0xf2f2f2));e.rotation.x=-Math.PI/2;e.position.set(xmax/2,0.01,z);scene.add(e)}
// trees, posts, a few hills
const trunkG=new THREE.CylinderGeometry(0.12,0.18,1.6,6),crownG=new THREE.ConeGeometry(1.6,4.2,7),trunkM=mat(0x6b4a2b),crownM=mat(0x2f6b2a);
let seed=7;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647;
for(let x=-30;x<xmax+60;x+=7+rnd()*6){for(const side of[-1,1]){if(rnd()<0.35)continue;const z=side*(6+rnd()*14),s=0.7+rnd()*0.8;const t=new THREE.Mesh(trunkG,trunkM);t.position.set(x,0.8*s,z);t.scale.setScalar(s);t.castShadow=true;scene.add(t);const c=new THREE.Mesh(crownG,crownM);c.position.set(x,(1.6+2.1)*s,z);c.scale.setScalar(s);c.castShadow=true;scene.add(c)}}
for(let x=0;x<xmax;x+=10){const p=new THREE.Mesh(new THREE.BoxGeometry(0.08,0.9,0.08),mat(0xffffff));p.position.set(x,0.45,-3.8);scene.add(p)}
const hillM=mat(0x7fa25a);for(let i=0;i<14;i++){const h=new THREE.Mesh(new THREE.SphereGeometry(30+rnd()*40,16,10),hillM);h.position.set(rnd()*xmax,-25-rnd()*10,(rnd()<.5?-1:1)*(70+rnd()*60));scene.add(h)}
// ---------- bike + rider builder
function tube(g,a,b,r,m){const d=new THREE.Vector3().subVectors(b,a),l=d.length();const c=new THREE.Mesh(new THREE.CylinderGeometry(r,r,l,10),m);c.position.copy(a).addScaledVector(d,0.5);c.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.clone().normalize());c.castShadow=true;g.add(c);return c}
const V=(x,y,z=0)=>new THREE.Vector3(x,y,z);
const paint=mat(0xd21f2b,{roughness:.35,metalness:.3}),black=mat(0x151515,{roughness:.6}),alu=mat(0xbbbbbb,{metalness:.7,roughness:.3}),skin=mat(0xe0b08a),jersey=mat(0xf6f6f6,{roughness:.7}),shorts=mat(0x1a1a22),helmetM=mat(0xf4f4f4,{roughness:.3});
function wheel(){const g=new THREE.Group();const tire=new THREE.Mesh(new THREE.TorusGeometry(R-0.014,0.014,10,40),black);tire.castShadow=true;g.add(tire);const rim=new THREE.Mesh(new THREE.TorusGeometry(R-0.05,0.022,6,40),mat(0x222222,{metalness:.5,roughness:.4}));g.add(rim);
 for(let i=0;i<18;i++){const a=i/18*Math.PI*2;tube(g,V(0,0,0),V(Math.cos(a)*(R-0.06),Math.sin(a)*(R-0.06),0),0.0025,alu)}const hub=new THREE.Mesh(new THREE.CylinderGeometry(0.03,0.03,0.1,8),alu);hub.rotation.x=Math.PI/2;g.add(hub);return g}
function buildBike(ghost){const root=new THREE.Group();root.userData.ghost=ghost;const roll=new THREE.Group();root.add(roll);
 const rw=wheel();rw.position.set(0,R,0);roll.add(rw);
 const BB=V(0.41,0.264),ST=V(0.45,0.80),HT=V(0.80,0.80),HB=V(0.83,0.70),HUB=V(0,R),SAD=V(0.37,1.02);
 tube(roll,ST,HT,0.018,paint);tube(roll,BB,HB,0.022,paint);tube(roll,BB,ST,0.017,paint);tube(roll,ST,SAD,0.014,black);
 tube(roll,BB,HUB,0.011,paint);tube(roll,HUB,ST,0.009,paint);tube(roll,HT,HB,0.022,paint);
 const saddle=new THREE.Mesh(new THREE.BoxGeometry(0.26,0.035,0.13),black);saddle.position.copy(SAD).add(V(-0.02,0.02));roll.add(saddle);
 const bbShell=new THREE.Mesh(new THREE.CylinderGeometry(0.035,0.035,0.09,10),paint);bbShell.rotation.x=Math.PI/2;bbShell.position.copy(BB);roll.add(bbShell);
 const ring=new THREE.Mesh(new THREE.TorusGeometry(0.10,0.006,4,30),alu);ring.position.copy(BB).add(V(0,0,0.06));roll.add(ring);
 // steered front assembly: pivot at head-tube bottom, axis tilted back by LAM
 const tilt=new THREE.Group();tilt.position.copy(HB);tilt.rotation.z=LAM;roll.add(tilt);const steer=new THREE.Group();tilt.add(steer);const un=new THREE.Group();un.rotation.z=-LAM;steer.add(un);
 const FW=V(0.981-0.83,R-0.70);tube(un,V(0,0),V(FW.x*0.5,FW.y*0.5,0.045),0.012,paint);tube(un,V(0,0),V(FW.x*0.5,FW.y*0.5,-0.045),0.012,paint);
 tube(un,V(FW.x*0.5,FW.y*0.5,0.045),V(FW.x,FW.y,0.045),0.010,paint);tube(un,V(FW.x*0.5,FW.y*0.5,-0.045),V(FW.x,FW.y,-0.045),0.010,paint);
 const fw=wheel();fw.position.copy(FW);un.add(fw);
 tube(un,V(0,0),V(-0.03,0.10),0.02,paint);tube(un,V(-0.03,0.10),V(0.07,0.13),0.015,black);// steerer + stem
 const bar=V(0.07,0.13);tube(un,V(bar.x,bar.y,-0.21),V(bar.x,bar.y,0.21),0.014,black);for(const z of[-0.2,0.2]){tube(un,V(bar.x,bar.y,z),V(bar.x+0.09,bar.y,z),0.013,black);tube(un,V(bar.x+0.09,bar.y,z),V(bar.x+0.10,bar.y-0.11,z),0.013,black)}
 // rider
 const hip=V(0.40,0.99),sh=V(0.72,1.22),head=V(0.84,1.34);tube(roll,hip,sh,0.085,jersey);tube(roll,V(hip.x-0.03,hip.y),V(hip.x+0.05,hip.y-0.02),0.095,shorts);
 const helmet=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,12),helmetM);helmet.position.copy(head).add(V(0,0.02));helmet.scale.set(1.15,0.95,1);helmet.castShadow=true;roll.add(helmet);
 const face=new THREE.Mesh(new THREE.SphereGeometry(0.085,12,10),skin);face.position.copy(head).add(V(0.02,-0.03));roll.add(face);
 const legs=[],arms=[];for(const s of[-1,1]){const z=0.14*s;const U=V(0,1,0);const thigh=tube(roll,hip,hip.clone().add(U),0.055,shorts),shin=tube(roll,hip,hip.clone().add(U),0.045,skin),shoe=new THREE.Mesh(new THREE.BoxGeometry(0.24,0.06,0.09),black);roll.add(shoe);const crank=tube(roll,BB,BB.clone().add(V(0,0.17)),0.012,black);legs.push({z,thigh,shin,shoe,crank});
  const hand=V(1.00,0.81,z*1.45);const upper=tube(roll,sh,sh.clone().add(U),0.04,jersey),fore=tube(roll,sh,sh.clone().add(U),0.035,skin);arms.push({z:z*1.45,upper,fore,hand})}
 // the fly on the helmet
 const fly=new THREE.Group();fly.position.copy(head).add(V(-0.02,0.14));roll.add(fly);const body=new THREE.Mesh(new THREE.SphereGeometry(0.035,10,8),mat(0x3a2a1a,{roughness:.5}));body.scale.set(1.5,0.8,0.9);fly.add(body);
 const eyeM=mat(0xc0201a,{emissive:0x600000});for(const s of[-1,1]){const e=new THREE.Mesh(new THREE.SphereGeometry(0.012,8,6),eyeM);e.position.set(0.045,0.012,0.018*s);fly.add(e)}
 const wingM=new THREE.MeshStandardMaterial({color:0xdfe9ff,transparent:true,opacity:.45,side:THREE.DoubleSide});for(const s of[-1,1]){const w=new THREE.Mesh(new THREE.PlaneGeometry(0.075,0.03),wingM);w.position.set(-0.03,0.02,0.035*s);w.rotation.x=Math.PI/2;w.rotation.z=-0.5;fly.add(w)}
 const glow=new THREE.PointLight(0xff8040,0,0.6);glow.position.set(0,0.05,0);fly.add(glow);const halo=new THREE.Mesh(new THREE.SphereGeometry(0.06,10,8),new THREE.MeshBasicMaterial({color:0xffa060,transparent:true,opacity:0}));fly.add(halo);
 if(ghost){root.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.transparent=true;o.material.opacity=0.15;o.castShadow=false}});glow.intensity=0}
 return {root,roll,steer,rw,fw,legs,arms,fly,glow,halo,BB,hip,sh,helmet,crank:0,ghost}}
function setTube(c,a,b){const d=new THREE.Vector3().subVectors(b,a),l=d.length();c.position.copy(a).addScaledVector(d,0.5);c.scale.set(1,l/c.geometry.parameters.height,1);c.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.clone().normalize())}
function pose(bk,s,done,steerT,powerW,pop){const [x,y,psi,v,phi,delta]=s;bk.root.position.set(x,0,y);bk.root.rotation.y=-psi;
 bk.roll.rotation.x=done?(phi>=0?1:-1)*1.35:phi;bk.steer.rotation.y=-delta;
 const wa=x/R;bk.rw.rotation.z=-wa;bk.fw.rotation.z=-wa;const ca=wa/2.6;bk.crank=ca;
 for(const L of bk.legs){const a=ca+(L.z>0?0:Math.PI);const ped=V(bk.BB.x+0.17*Math.cos(a),bk.BB.y+0.17*Math.sin(a),L.z);setTube(L.crank,V(bk.BB.x,bk.BB.y,L.z*0.5),ped);
  const hip=V(bk.hip.x,bk.hip.y,L.z);const mid=hip.clone().lerp(ped,0.5);const d=new THREE.Vector3().subVectors(ped,hip);const n=V(-d.y,d.x,0).normalize();const half=Math.min(hip.distanceTo(ped)/2,0.455);const kneeOff=Math.sqrt(Math.max(0.46*0.46-half*half,0.0025));mid.addScaledVector(n,kneeOff);
  setTube(L.thigh,hip,mid);setTube(L.shin,mid,ped);L.shoe.position.copy(ped).add(V(0.05,-0.02));}
 for(const A of bk.arms){const sh=V(bk.sh.x,bk.sh.y,A.z*0.7);const hand=A.hand.clone();hand.z=A.z;hand.x+=(-delta*0.1)*(A.z>0?1:-1);const mid=sh.clone().lerp(hand,0.5).add(V(0.02,-0.08));setTube(A.upper,sh,mid);setTube(A.fore,mid,hand)}
 const g=Math.min(1,(pop||0)/150000);if(!bk.ghost)bk.glow.intensity=0.4+1.2*g;bk.halo.material.opacity=0.08+0.3*g;bk.halo.scale.setScalar(1+0.6*g)}
// ---------- riders
const bikes=[];const shown=Math.min(B,24);let hero=null;for(let i=0;i<shown;i++){const bk=buildBike(true);scene.add(bk.root);bikes.push(bk)}hero=buildBike(false);scene.add(hero.root);
const rsel=document.getElementById('rider');for(let i=0;i<B;i++){const o=document.createElement('option');o.value=i;o.textContent='骑手 #'+i;rsel.appendChild(o)}
let rider=Math.min(B-1,+(Q.get('rider')??(D.best_rider||0)));
// ---------- brain bars
const barsEl=document.getElementById('bars');const pairs=[];for(let i=0;i<dnNames.length;i+=2){const n=dnNames[i].replace(/_L$/,'');const row=document.createElement('div');row.className='bar';row.innerHTML='<span>'+n+'</span><i></i><i class="r"></i>';barsEl.appendChild(row);pairs.push({l:i,r:i+1,el:row.querySelectorAll('i')})}
// ---------- playback + camera
let k=Math.min(tr.length-1,Math.round((+Q.get('t')||0)/dt)),playing=false,camMode=+(Q.get('cam')||0),last=0;const slider=document.getElementById('time');slider.max=tr.length-1;
const camPos=new THREE.Vector3(-6,2,2),camTgt=new THREE.Vector3();const tmp=new THREE.Vector3();let snap=true,showGhosts=Q.get('ghosts')!=='0';
function frame(){const row=tr[k];for(let i=0;i<shown;i++){bikes[i].root.visible=showGhosts&&(i!==rider);pose(bikes[i],row.state[i],row.done[i],row.steer[i],row.power[i],0)}pose(hero,row.state[rider],row.done[rider],row.steer[rider],row.power[rider],row.pop_hz?row.pop_hz[rider]:0);
 const s=row.state[rider];const yaw=-s[2];const fwd=V(Math.cos(yaw),0,-Math.sin(yaw)),right=V(Math.sin(yaw),0,Math.cos(yaw));const base=V(s[0],0,s[1]);
 if(camMode===0){tmp.copy(base).addScaledVector(fwd,-5.5).addScaledVector(right,1.6).add(V(0,1.9,0));camPos.lerp(tmp,snap?1:0.08);camTgt.lerp(base.clone().add(V(0,0.9,0)).addScaledVector(fwd,1.5),snap?1:0.15);camera.position.copy(camPos);camera.lookAt(camTgt)}
 else if(camMode===1){tmp.copy(base).addScaledVector(right,7).addScaledVector(fwd,1.5).add(V(0,1.3,0));camPos.lerp(tmp,snap?1:0.1);camTgt.lerp(base.clone().add(V(0,0.8,0)),snap?1:0.2);camera.position.copy(camPos);camera.lookAt(camTgt)}
 else{const bk=hero;{bk.fly.updateWorldMatrix(true,false);const p=new THREE.Vector3(0.05,0.06,0).applyMatrix4(bk.fly.matrixWorld);const f=new THREE.Vector3(3,0.3,0).applyMatrix4(bk.fly.matrixWorld);camera.position.copy(p);camera.up.set(0,1,0).applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(bk.roll.matrixWorld));camera.lookAt(f);camera.up.set(0,1,0)}}
 sun.position.set(s[0]-20,40,25);sun.target.position.set(s[0],0,0);
 document.getElementById('v').textContent=(s[3]*3.6).toFixed(1);document.getElementById('phi').textContent=(s[4]*57.3).toFixed(1)+'°';document.getElementById('delta').textContent=(s[5]*57.3).toFixed(1)+'°';
 document.getElementById('T').textContent=row.steer[rider].toFixed(2)+' Nm';document.getElementById('P').textContent=row.power[rider].toFixed(0)+' W';document.getElementById('x').textContent=s[0].toFixed(1)+' m';
 document.getElementById('alive').textContent=row.done.filter(d=>!d).length+' / '+B;document.getElementById('pop').textContent=row.pop_hz?Math.round(row.pop_hz[rider]).toLocaleString():'—';
 document.getElementById('fall').style.display=row.done[rider]?'block':'none';document.getElementById('clock').textContent=row.t.toFixed(2)+' s';slider.value=k;
 if(row.dn_hz){const d=row.dn_hz[rider];for(const p of pairs){p.el[0].style.transform='scaleX('+Math.min(1,d[p.l]/120)+')';p.el[1].style.transform='scaleX('+Math.min(1,d[p.r]/120)+')'}}
 snap=false;renderer.render(scene,camera)}
function resize(){const w=innerWidth,h=innerHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}addEventListener('resize',resize);resize();
function loop(ts){if(playing){const sp=parseFloat(document.getElementById('speed').value);if(ts-last>dt*1000/sp){last=ts;k=Math.min(k+1,tr.length-1);if(k===tr.length-1){playing=false;document.getElementById('play').textContent='↺ 重播'}}}frame();requestAnimationFrame(loop)}
document.getElementById('play').onclick=()=>{if(k>=tr.length-1)k=0;playing=!playing;document.getElementById('play').textContent=playing?'⏸ 暂停':'▶ 播放'};
slider.oninput=e=>{k=+e.target.value;snap=true};rsel.onchange=e=>{rider=+e.target.value;snap=true};
document.getElementById('ghosts').onclick=e=>{showGhosts=!showGhosts;e.target.classList.toggle('on',showGhosts)};document.getElementById('ghosts').classList.toggle('on',showGhosts);
for(const j of[0,1,2])document.getElementById('cam'+j).classList.toggle('on',j===camMode);rsel.value=rider;
for(const i of[0,1,2]){document.getElementById('cam'+i).onclick=()=>{camMode=i;snap=true;for(const j of[0,1,2])document.getElementById('cam'+j).classList.toggle('on',j===i)}}
addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();document.getElementById('play').click()}if(e.key==='1'||e.key==='2'||e.key==='3')document.getElementById('cam'+(+e.key-1)).click()});
requestAnimationFrame(loop);
</script></body></html>
"""


def main():
    src = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/ride_trace.json")
    dst = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/ride_3d.html")
    data = json.loads(src.read_text())
    names = data.get("dn_names", [])
    keep = [i for i, n in enumerate(names) if n in DN_SHOWN]
    for row in data["trace"]:
        row.pop("sense_hz", None)
        row.pop("brake", None)
        if "dn_hz" in row:
            row["dn_hz"] = [[r[i] for i in keep] for r in row["dn_hz"]]
        row["state"] = [[round(v, 4) for v in s] for s in row["state"]]
    data["dn_names"] = [names[i] for i in keep]
    res = data.pop("result", None)
    if res and "upright" in res:
        data["best_rider"] = int(max(range(len(res["upright"])), key=lambda i: (res["upright"][i], -abs(res["lateral"][i]))))
    dst.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    print(f"{dst}  ({dst.stat().st_size / 1e6:.1f} MB, {len(data['trace'])} steps, {len(data['trace'][0]['state'])} riders)")


if __name__ == "__main__":
    main()
