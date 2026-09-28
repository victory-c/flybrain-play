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
<button id="cam0" class="on">跟拍</button><button id="cam1">侧拍</button><button id="cam2">苍蝇视角</button><button id="cam3">特写</button><button id="ghosts" class="on">其他骑手</button>
<select id="rider"></select></div>
<div id="title" class="panel"></div><div id="fall">倒了！</div>
<script id="data" type="application/json">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
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
const tan=mat(0xc9a36b,{roughness:.85}),rimM=mat(0x101010,{roughness:.35,metalness:.4}),steel=mat(0xd8d8d8,{metalness:.9,roughness:.25}),redAcc=mat(0xc8102e,{roughness:.4,metalness:.3});
function lathe(pts,m,seg=48){const g=new THREE.LatheGeometry(pts.map(p=>new THREE.Vector2(p[0],p[1])),seg);const c=new THREE.Mesh(g,m);c.rotation.x=Math.PI/2;c.castShadow=true;return c}
function wheel(depth){const g=new THREE.Group();const rOut=R-0.025,rIn=rOut-depth;
 const tread=new THREE.Mesh(new THREE.TorusGeometry(R-0.012,0.012,12,56),black);tread.castShadow=true;g.add(tread);
 g.add(new THREE.Mesh(new THREE.TorusGeometry(R-0.021,0.012,10,56),tan));
 g.add(lathe([[rIn,-0.010],[rOut,-0.012],[rOut+0.004,0],[rOut,0.012],[rIn,0.010],[rIn,-0.010]],rimM));
 for(let i=0;i<21;i++){const a=i/21*Math.PI*2,z=(i%2?1:-1)*0.018;const sp=new THREE.Mesh(new THREE.BoxGeometry(0.003,rIn-0.03,0.0015),steel);sp.position.set(Math.cos(a)*(rIn+0.03)/2*1.0,Math.sin(a)*(rIn+0.03)/2,z);sp.rotation.z=a-Math.PI/2;g.add(sp)}
 const hub=new THREE.Mesh(new THREE.CylinderGeometry(0.02,0.02,0.1,12),black);hub.rotation.x=Math.PI/2;g.add(hub);
 const rotor=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.08,0.002,40),steel);rotor.rotation.x=Math.PI/2;rotor.position.z=-0.058;g.add(rotor);
 const rc2=new THREE.Mesh(new THREE.CylinderGeometry(0.045,0.045,0.003,24),black);rc2.rotation.x=Math.PI/2;rc2.position.z=-0.059;g.add(rc2);
 return g}
function decal(text,w,h){const c=document.createElement('canvas');c.width=1024;c.height=Math.round(1024*h/w);const x=c.getContext('2d');x.fillStyle='rgba(0,0,0,0)';x.fillRect(0,0,c.width,c.height);x.fillStyle='#fff';x.font='italic bold '+Math.round(c.height*0.78)+'px Helvetica,Arial,sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(text,c.width/2,c.height/2);
 const t=new THREE.CanvasTexture(c);t.encoding=THREE.sRGBEncoding;return new THREE.MeshBasicMaterial({map:t,transparent:true,depthWrite:false})}
function buildBike(ghost){const root=new THREE.Group();root.userData.ghost=ghost;const roll=new THREE.Group();root.add(roll);
 const BB=V(0.41,0.264),STt=V(0.44,0.79),HT=V(0.815,0.86),HB=V(0.848,0.71),HUB=V(0,R),SAD=V(0.37,1.00),SSJ=V(0.455,0.62),CROWN=V(0.862,0.65);
 const rw=wheel(0.060);rw.position.copy(HUB);roll.add(rw);
 // aero centre-plane tubes: 2:1 profiles by flattening the group sideways
 const aero=new THREE.Group();aero.scale.z=0.5;roll.add(aero);
 tube(aero,HB.clone().add(V(0.01,-0.02)),BB.clone().add(V(0.02,0.01)),0.038,paint);            // down tube
 tube(aero,BB,STt.clone().add(V(-0.02,0.08)),0.028,paint);                                       // seat tube
 tube(aero,STt.clone().add(V(-0.01,0.03)),SAD.clone().add(V(0.0,-0.02)),0.017,black);            // seat post
 tube(aero,STt,HT,0.024,paint);                                                                  // top tube
 tube(aero,HT.clone().add(V(-0.01,0.03)),HB.clone().add(V(0.0,-0.02)),0.034,paint);              // head tube
 for(const z of[-0.045,0.045]){tube(roll,BB.clone().add(V(0.02,0,z*0.6)),HUB.clone().add(V(0.02,-0.0,z)),0.011,paint);   // chainstays
  tube(roll,HUB.clone().add(V(0.0,0.01,z)),SSJ.clone().add(V(0,0,z*0.4)),0.008,paint)}                                     // dropped seat stays
 const saddle=new THREE.Mesh(new THREE.BoxGeometry(0.26,0.03,0.135),black);saddle.position.copy(SAD).add(V(-0.03,0.015));roll.add(saddle);
 // bottle cages
 for(const [a,b] of[[V(0.60,0.50),V(0.70,0.62)],[V(0.43,0.47),V(0.44,0.58)]]){const bt=new THREE.Mesh(new THREE.CylinderGeometry(0.036,0.036,0.21,14),black);bt.position.copy(a).lerp(b,0.5);bt.quaternion.setFromUnitVectors(V(0,1,0),b.clone().sub(a).normalize());roll.add(bt)}
 // drivetrain: crank, chainrings, chain, cassette, derailleur
 const bbShell=new THREE.Mesh(new THREE.CylinderGeometry(0.036,0.036,0.09,12),paint);bbShell.rotation.x=Math.PI/2;bbShell.position.copy(BB);roll.add(bbShell);
 const ring=lathe([[0.08,-0.0015],[0.105,-0.0015],[0.105,0.0015],[0.08,0.0015],[0.08,-0.0015]],steel,56);ring.position.copy(BB).add(V(0,0,0.062));roll.add(ring);
 const ring2=lathe([[0.03,-0.0015],[0.08,-0.0015],[0.08,0.0015],[0.03,0.0015],[0.03,-0.0015]],black,40);ring2.position.copy(BB).add(V(0,0,0.056));roll.add(ring2);
 const cass=new THREE.Mesh(new THREE.CylinderGeometry(0.048,0.03,0.04,24),steel);cass.rotation.x=Math.PI/2;cass.position.copy(HUB).add(V(0,0,0.06));roll.add(cass);
 const der=new THREE.Mesh(new THREE.BoxGeometry(0.06,0.09,0.02),black);der.position.copy(HUB).add(V(0.03,-0.07,0.06));roll.add(der);
 const chainPts=[];for(let i=0;i<=40;i++){const a=i/40*Math.PI*2;const cx=i<20?BB.x:HUB.x,cy=i<20?BB.y:HUB.y,cr=i<20?0.105:0.045;const ang=i<20?(Math.PI/2+i/19*Math.PI):(3*Math.PI/2+(i-20)/20*Math.PI);chainPts.push(V(cx+cr*Math.cos(ang),cy+cr*Math.sin(ang),0.06))}
 roll.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(chainPts,true),80,0.004,6,true),mat(0x777777,{metalness:.8,roughness:.4})));
 // S-WORKS on the down tube, both sides
 const dtDir=HB.clone().sub(BB).normalize(),dtMid=HB.clone().lerp(BB,0.55);const dm=decal('S-WORKS',0.34,0.06);
 for(const side of[1,-1]){const pl=new THREE.Mesh(new THREE.PlaneGeometry(0.34,0.06),dm);pl.position.copy(dtMid).add(V(0,0,side*0.0205));pl.rotation.z=Math.atan2(dtDir.y,dtDir.x);if(side<0)pl.rotation.y=Math.PI;roll.add(pl)}
 // steered front assembly: pivot at head-tube bottom, axis tilted back by LAM
 const tilt=new THREE.Group();tilt.position.copy(HB);tilt.rotation.z=LAM;roll.add(tilt);const steer=new THREE.Group();tilt.add(steer);const un=new THREE.Group();un.rotation.z=-LAM;steer.add(un);
 const rel=v=>v.clone().sub(HB);const FWc=rel(V(0.981,R)),CR=rel(CROWN);
 tube(un,V(0,0),CR,0.03,paint);                                                                  // crown
 for(const z of[-0.045,0.045]){tube(un,CR.clone().add(V(0,0,z)),FWc.clone().add(V(0,0,z)),0.012,paint)}   // fork blades
 const fw=wheel(0.051);fw.position.copy(FWc);un.add(fw);
 const cal=new THREE.Mesh(new THREE.BoxGeometry(0.05,0.07,0.025),black);cal.position.copy(FWc).add(V(-0.02,0.07,-0.058));un.add(cal);
 // integrated Roval Rapide cockpit: stem, flat aero tops, hoods, drops
 const HTt=rel(HT.clone().add(V(-0.005,0.035)));tube(un,rel(HB.clone().add(V(0,0.0))),HTt,0.02,black);
 const stemEnd=HTt.clone().add(V(0.10,0.02));tube(un,HTt,stemEnd,0.016,black);
 const tops=new THREE.Mesh(new THREE.BoxGeometry(0.05,0.02,0.40),black);tops.position.copy(stemEnd);un.add(tops);
 for(const z of[-0.20,0.20]){const c0=stemEnd.clone().add(V(0,0,z));const pts=[c0,c0.clone().add(V(0.09,0.005)),c0.clone().add(V(0.13,-0.03)),c0.clone().add(V(0.125,-0.09)),c0.clone().add(V(0.07,-0.13))];
  un.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),24,0.012,8,false),black));
  const hood=new THREE.Mesh(new THREE.BoxGeometry(0.08,0.035,0.03),black);hood.position.copy(c0).add(V(0.12,0.015));un.add(hood)}
 const bar=V(0.085,0.19);  // hoods position relative to HB, used by the rider's hands
 // rider
 const hip=V(0.40,0.99),sh=V(0.72,1.22),head=V(0.84,1.34);tube(roll,hip,sh,0.07,jersey);tube(roll,V(hip.x-0.03,hip.y),V(hip.x+0.05,hip.y-0.02),0.08,shorts);
 const helmet=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,12),helmetM);helmet.position.copy(head).add(V(0,0.02));helmet.scale.set(1.15,0.95,1);helmet.castShadow=true;roll.add(helmet);
 const face=new THREE.Mesh(new THREE.SphereGeometry(0.085,12,10),skin);face.position.copy(head).add(V(0.02,-0.03));roll.add(face);
 const legs=[],arms=[];for(const s of[-1,1]){const z=0.14*s;const U=V(0,1,0);const thigh=tube(roll,hip,hip.clone().add(U),0.048,shorts),shin=tube(roll,hip,hip.clone().add(U),0.038,skin),shoe=new THREE.Mesh(new THREE.BoxGeometry(0.24,0.06,0.09),black);roll.add(shoe);const crank=tube(roll,BB,BB.clone().add(V(0,0.17)),0.012,black);legs.push({z,thigh,shin,shoe,crank});
  const hand=V(0.97,0.88,z*1.45);const upper=tube(roll,sh,sh.clone().add(U),0.036,jersey),fore=tube(roll,sh,sh.clone().add(U),0.03,skin);arms.push({z:z*1.45,upper,fore,hand})}
 // the fly on the helmet
 const fly=new THREE.Group();fly.position.copy(head).add(V(-0.02,0.14));roll.add(fly);const body=new THREE.Mesh(new THREE.SphereGeometry(0.035,10,8),mat(0x3a2a1a,{roughness:.5}));body.scale.set(1.5,0.8,0.9);fly.add(body);
 const eyeM=mat(0xc0201a,{emissive:0x600000});for(const s of[-1,1]){const e=new THREE.Mesh(new THREE.SphereGeometry(0.012,8,6),eyeM);e.position.set(0.045,0.012,0.018*s);fly.add(e)}
 const wingM=new THREE.MeshStandardMaterial({color:0xdfe9ff,transparent:true,opacity:.45,side:THREE.DoubleSide});for(const s of[-1,1]){const w=new THREE.Mesh(new THREE.PlaneGeometry(0.075,0.03),wingM);w.position.set(-0.03,0.02,0.035*s);w.rotation.x=Math.PI/2;w.rotation.z=-0.5;fly.add(w)}
 const glow=new THREE.PointLight(0xff8040,0,0.6);glow.position.set(0,0.05,0);fly.add(glow);const halo=new THREE.Mesh(new THREE.SphereGeometry(0.06,10,8),new THREE.MeshBasicMaterial({color:0xffa060,transparent:true,opacity:0}));fly.add(halo);
 if(ghost){root.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.transparent=true;o.material.opacity=0.15;o.castShadow=false}});glow.intensity=0}
 root.userData.bikeParts=[rw,aero,tilt,saddle,bbShell,ring,ring2,cass,der];
 return {root,roll,steer,rw,fw,legs,arms,fly,glow,halo,BB,hip,sh,helmet,crank:0,ghost}}
function loadModel(bk,url,yawDeg){new THREE.GLTFLoader().load(url,g=>{const m=g.scene;m.rotation.y=(yawDeg||0)*Math.PI/180;m.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(m);const size=box.getSize(new THREE.Vector3());const L=Math.max(size.x,size.z);const k=(0.981+2*R)/L;m.scale.setScalar(k);m.updateMatrixWorld(true);const b2=new THREE.Box3().setFromObject(m);m.position.set(-R-b2.min.x,-b2.min.y,-(b2.min.z+b2.max.z)/2);m.traverse(o=>{if(o.isMesh){o.castShadow=true}});bk.roll.add(m);bk.roll.traverse(o=>{});for(const part of bk.root.userData.bikeParts)part.visible=false;bk.roll.children.forEach(c=>{if(c.isMesh&&c.material&&c.material.map&&c.material.transparent)c.visible=false})},undefined,e=>console.warn('model load failed',e))}
function setTube(c,a,b){const d=new THREE.Vector3().subVectors(b,a),l=d.length();c.position.copy(a).addScaledVector(d,0.5);c.scale.set(1,l/c.geometry.parameters.height,1);c.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.clone().normalize())}
function pose(bk,s,done,steerT,powerW,pop){const [x,y,psi,v,phi,delta]=s;bk.root.position.set(x,0,y);bk.root.rotation.y=-psi;
 bk.roll.rotation.x=done?(phi>=0?1:-1)*1.35:phi;bk.steer.rotation.y=-delta;
 const wa=x/R;bk.rw.rotation.z=-wa;bk.fw.rotation.z=-wa;const ca=-wa/1.5;bk.crank=ca;
 for(const L of bk.legs){const a=ca+(L.z>0?0:Math.PI);const ped=V(bk.BB.x+0.17*Math.cos(a),bk.BB.y+0.17*Math.sin(a),L.z);setTube(L.crank,V(bk.BB.x,bk.BB.y,L.z*0.5),ped);
  const hip=V(bk.hip.x,bk.hip.y,L.z);const mid=hip.clone().lerp(ped,0.5);const d=new THREE.Vector3().subVectors(ped,hip);const n=V(-d.y,d.x,0).normalize();const half=Math.min(hip.distanceTo(ped)/2,0.455);const kneeOff=Math.sqrt(Math.max(0.46*0.46-half*half,0.0025));mid.addScaledVector(n,kneeOff);
  setTube(L.thigh,hip,mid);setTube(L.shin,mid,ped);L.shoe.position.copy(ped).add(V(0.05,-0.02));}
 for(const A of bk.arms){const sh=V(bk.sh.x,bk.sh.y,A.z*0.7);const hand=A.hand.clone();hand.z=A.z;hand.x+=(-delta*0.1)*(A.z>0?1:-1);const mid=sh.clone().lerp(hand,0.5).add(V(0.02,-0.08));setTube(A.upper,sh,mid);setTube(A.fore,mid,hand)}
 const g=Math.min(1,(pop||0)/150000);if(!bk.ghost)bk.glow.intensity=0.4+1.2*g;bk.halo.material.opacity=0.08+0.3*g;bk.halo.scale.setScalar(1+0.6*g)}
// ---------- riders
const bikes=[];const shown=Math.min(B,24);let hero=null;for(let i=0;i<shown;i++){const bk=buildBike(true);scene.add(bk.root);bikes.push(bk)}hero=buildBike(false);scene.add(hero.root);if(Q.get('model'))loadModel(hero,Q.get('model'),+(Q.get('model_yaw')||0));
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
 else if(camMode===3){tmp.copy(base).addScaledVector(right,2.6).addScaledVector(fwd,0.55).add(V(0,0.75,0));camPos.lerp(tmp,snap?1:0.1);camTgt.lerp(base.clone().add(V(0,0.62,0)).addScaledVector(fwd,0.5),snap?1:0.2);camera.position.copy(camPos);camera.lookAt(camTgt)}
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
for(const j of[0,1,2,3])document.getElementById('cam'+j).classList.toggle('on',j===camMode);rsel.value=rider;
for(const i of[0,1,2,3]){document.getElementById('cam'+i).onclick=()=>{camMode=i;snap=true;for(const j of[0,1,2,3])document.getElementById('cam'+j).classList.toggle('on',j===i)}}
addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();document.getElementById('play').click()}if('1234'.includes(e.key))document.getElementById('cam'+(+e.key-1)).click()});
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
