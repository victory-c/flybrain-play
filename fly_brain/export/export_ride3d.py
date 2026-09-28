"""A ride trace as a 3D road scene: the bike and its rider lean, steer and pedal exactly as the
simulation says, with the fly on the helmet. Chase, side, close-up camera, or the fly's own eyes.

The followed rider rides a real 3D model when one is available: by default assets/colnago_v4rs.glb
(Colnago V4Rs from Colnago's own web configurator, Draco-compressed; see bike/README.md for where it
comes from and why it must not be redistributed). Its wheels, crank, fork and bars are re-parented
onto pivots so they spin and steer. Without a model, a procedural S-Works Tarmac SL9 is drawn.

usage: python -m export.export_ride3d results/ride_trace.json results/ride_3d.html [--bike PATH|none]
"""
import argparse
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIKE = ROOT / "assets" / "colnago_v4rs.glb"
DN_SHOWN = ["DNp20_L", "DNp20_R", "DNg46_L", "DNg46_R", "DNp22_L", "DNp22_R", "b1 MN_L", "b1 MN_R"]

HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>🪰 Fly rides a road bike</title>
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
input[type=range]{width:260px}
#title{top:14px;left:50%;transform:translateX(-50%);color:#9aa;font-size:12px;white-space:nowrap}
#fall{position:fixed;top:40%;left:50%;transform:translateX(-50%);font-size:44px;font-weight:800;color:#ff4d4d;text-shadow:0 2px 12px #000;display:none}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud" class="panel"><div><span class="big" id="v">0.0</span><span class="unit">km/h</span></div>
<div class="row"><span>倾角 lean</span><b id="phi">0°</b></div><div class="row"><span>把角 steer</span><b id="delta">0°</b></div>
<div class="row"><span>转向扭矩</span><b id="T">0 Nm</b></div><div class="row"><span>踩踏功率</span><b id="P">0 W</b></div>
<div class="row"><span>踏频</span><b id="cad">0 rpm</b></div><div class="row"><span>里程</span><b id="x">0 m</b></div><div class="row"><span>直立骑手</span><b id="alive"></b></div></div>
<div id="brain" class="panel"><h4>🪰 苍蝇全脑 <span id="pop"></span> spikes/s</h4><div id="bars"></div><div style="font-size:11px;color:#778;margin-top:6px">下行神经元 左(蓝) / 右(红)，Hz</div></div>
<div id="ctl" class="panel"><button id="play">▶ 播放</button><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select>
<input id="time" type="range" min="0" value="0"><span id="clock" style="font-variant-numeric:tabular-nums;min-width:56px">0.00 s</span>
<button id="cam0" class="on">跟拍</button><button id="cam1">侧拍</button><button id="cam2">苍蝇视角</button><button id="cam3">特写</button><button id="ghosts" class="on">其他骑手</button>
<select id="paint" title="车架涂装"></select><select id="rider"></select></div>
<div id="title" class="panel">加载车模…</div><div id="fall">倒了！</div>
<script id="data" type="application/json">__DATA__</script>
<script id="bikeglb" type="application/octet-stream">__BIKE__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/DRACOLoader.js"></script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),tr=D.trace,B=tr[0].state.length,dt=tr[1].t-tr[0].t;
const Q=new URLSearchParams(location.search);const dnNames=D.dn_names||[];
const V=(x,y,z=0)=>new THREE.Vector3(x,y,z);
// ---------- scene
const canvas=document.getElementById('c');const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.outputEncoding=THREE.sRGBEncoding;renderer.physicallyCorrectLights=false;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x9fd0f5);scene.fog=new THREE.Fog(0x9fd0f5,60,220);
const camera=new THREE.PerspectiveCamera(60,1,0.05,400);
scene.add(new THREE.HemisphereLight(0xbfe3ff,0x4a6a2a,0.8));
const sun=new THREE.DirectionalLight(0xfff2d8,1.5);sun.position.set(-20,40,25);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.bias=-0.0004;
Object.assign(sun.shadow.camera,{left:-25,right:25,top:25,bottom:-25,near:1,far:120});scene.add(sun);scene.add(sun.target);
const fill=new THREE.DirectionalLight(0xffffff,0.35);fill.position.set(10,8,-12);scene.add(fill);
const xmax=Math.max(60,...tr.map(r=>Math.max(...r.state.map(s=>s[0]))))+80;
const mat=(c,o={})=>new THREE.MeshStandardMaterial(Object.assign({color:c,roughness:.8,metalness:.05},o));
const ground=new THREE.Mesh(new THREE.PlaneGeometry(xmax+200,400),mat(0x5d8a3a));ground.rotation.x=-Math.PI/2;ground.position.x=xmax/2;ground.receiveShadow=true;scene.add(ground);
const road=new THREE.Mesh(new THREE.PlaneGeometry(xmax+100,7),mat(0x3a3d44,{roughness:.95}));road.rotation.x=-Math.PI/2;road.position.set(xmax/2,0.005,0);road.receiveShadow=true;scene.add(road);
const dash=new THREE.InstancedMesh(new THREE.PlaneGeometry(2,0.12),mat(0xf2f2f2),Math.ceil(xmax/6)+20);let m4=new THREE.Matrix4();
for(let i=0;i<dash.count;i++){m4.makeRotationX(-Math.PI/2);m4.setPosition(i*6-20,0.01,0);dash.setMatrixAt(i,m4)}scene.add(dash);
for(const z of[-3.2,3.2]){const e=new THREE.Mesh(new THREE.PlaneGeometry(xmax+100,0.12),mat(0xf2f2f2));e.rotation.x=-Math.PI/2;e.position.set(xmax/2,0.01,z);scene.add(e)}
const trunkG=new THREE.CylinderGeometry(0.12,0.18,1.6,6),crownG=new THREE.ConeGeometry(1.6,4.2,7),trunkM=mat(0x6b4a2b),crownM=mat(0x2f6b2a);
let seed=7;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647;
for(let x=-30;x<xmax+60;x+=7+rnd()*6){for(const side of[-1,1]){if(rnd()<0.35)continue;const z=side*(6+rnd()*14),s=0.7+rnd()*0.8;const t=new THREE.Mesh(trunkG,trunkM);t.position.set(x,0.8*s,z);t.scale.setScalar(s);t.castShadow=true;scene.add(t);const c=new THREE.Mesh(crownG,crownM);c.position.set(x,(1.6+2.1)*s,z);c.scale.setScalar(s);c.castShadow=true;scene.add(c)}}
for(let x=0;x<xmax;x+=10){const p=new THREE.Mesh(new THREE.BoxGeometry(0.08,0.9,0.08),mat(0xffffff));p.position.set(x,0.45,-3.8);scene.add(p)}
const hillM=mat(0x7fa25a);for(let i=0;i<14;i++){const h=new THREE.Mesh(new THREE.SphereGeometry(30+rnd()*40,16,10),hillM);h.position.set(rnd()*xmax,-25-rnd()*10,(rnd()<.5?-1:1)*(70+rnd()*60));scene.add(h)}
// ---------- materials and helpers
const paint=mat(0xd21f2b,{roughness:.35,metalness:.3});paint.userData.paint=true;
const black=mat(0x151515,{roughness:.6}),skin=mat(0xe0b08a),jersey=mat(0xf6f6f6,{roughness:.7}),shorts=mat(0x1a1a22),helmetM=mat(0xf4f4f4,{roughness:.3});
const tan=mat(0xc9a36b,{roughness:.85}),rimM=mat(0x101010,{roughness:.35,metalness:.4}),steel=mat(0xd8d8d8,{metalness:.9,roughness:.25});
function tube(g,a,b,r,m){const d=new THREE.Vector3().subVectors(b,a),l=Math.max(d.length(),1e-4);const c=new THREE.Mesh(new THREE.CylinderGeometry(r,r,1,10),m);c.scale.y=l;c.position.copy(a).addScaledVector(d,0.5);c.quaternion.setFromUnitVectors(V(0,1,0),d.clone().normalize());c.castShadow=true;g.add(c);return c}
function setTube(c,a,b){const d=new THREE.Vector3().subVectors(b,a),l=Math.max(d.length(),1e-4);c.position.copy(a).addScaledVector(d,0.5);c.scale.set(1,l,1);c.quaternion.setFromUnitVectors(V(0,1,0),d.divideScalar(l))}
function lathe(pts,m,seg=48){const c=new THREE.Mesh(new THREE.LatheGeometry(pts.map(p=>new THREE.Vector2(p[0],p[1])),seg),m);c.rotation.x=Math.PI/2;c.castShadow=true;return c}
function wheel(depth,R){const g=new THREE.Group();const rOut=R-0.025,rIn=rOut-depth;
 const tread=new THREE.Mesh(new THREE.TorusGeometry(R-0.012,0.012,12,56),black);tread.castShadow=true;g.add(tread);
 g.add(new THREE.Mesh(new THREE.TorusGeometry(R-0.021,0.012,10,56),tan));
 g.add(lathe([[rIn,-0.010],[rOut,-0.012],[rOut+0.004,0],[rOut,0.012],[rIn,0.010],[rIn,-0.010]],rimM));
 for(let i=0;i<21;i++){const a=i/21*Math.PI*2,z=(i%2?1:-1)*0.018;const sp=new THREE.Mesh(new THREE.BoxGeometry(0.003,rIn-0.03,0.0015),steel);sp.position.set(Math.cos(a)*(rIn+0.03)/2,Math.sin(a)*(rIn+0.03)/2,z);sp.rotation.z=a-Math.PI/2;g.add(sp)}
 const hub=new THREE.Mesh(new THREE.CylinderGeometry(0.02,0.02,0.1,12),black);hub.rotation.x=Math.PI/2;g.add(hub);
 const rotor=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.08,0.002,40),steel);rotor.rotation.x=Math.PI/2;rotor.position.z=-0.058;g.add(rotor);return g}
function decal(text,w,h){const c=document.createElement('canvas');c.width=1024;c.height=Math.round(1024*h/w);const x=c.getContext('2d');x.fillStyle='#fff';x.font='italic bold '+Math.round(c.height*0.78)+'px Helvetica,Arial,sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(text,c.width/2,c.height/2);
 const t=new THREE.CanvasTexture(c);t.encoding=THREE.sRGBEncoding;return new THREE.MeshBasicMaterial({map:t,transparent:true,depthWrite:false})}
const Z=V(0,0,1),_q=new THREE.Quaternion();
function spin(p,a){p.quaternion.copy(p.userData.q0).multiply(_q.setFromAxisAngle(Z,a))}
// ---------- bike geometry. Roll frame: origin at the rear contact point, x forward, y up, z to the rider's right.
let MODEL=null,G=null,FIT=null;
const SN=n=>THREE.PropertyBinding.sanitizeNodeName(n);  // GLTFLoader renames nodes: spaces -> _, drops []:./
function procGeo(){ // S-Works Tarmac SL9, 56 cm: wheelbase 981, head angle 73.5, BB drop 72, 172.5 cranks
 return {name:'Specialized S-Works Tarmac SL9',decal:'S-WORKS',paint:0xd21f2b,R:0.336,LAM:16.5*Math.PI/180,
  RH:V(0,0.336),FH:V(0.981,0.336),BB:V(0.405,0.264),SAD:V(0.20,0.975),HT:V(0.785,0.85),HB:V(0.826,0.71),HOOD:V(0.96,0.905),hoodZ:0.19,crankLen:0.1725,crank0:0}}
function modelGeo(){ // every point read off the model's own anchors and part bounding boxes
 const m=MODEL;m.updateMatrixWorld(true);const by=n=>m.getObjectByName(SN(n));const wp=n=>by(n).getWorldPosition(V(0,0));
 const bb=names=>{const b=new THREE.Box3();for(const n of names){const o=by(n);if(o)b.expandByObject(o)}return b};
 const RH=wp('anchor_frame_rearhub'),FH=wp('anchor_frame_fronthub'),BB=wp('anchor_frame_bottom_bracket');
 const tyre=bb(['Tyre Pzero001','Tyre Pzero']);const R=(tyre.max.y-tyre.min.y)/2;const off=V(-RH.x,-(RH.y-R),-RH.z);
 const o2=v=>V(v.x+off.x,v.y+off.y,0);
 const sad=bb(['Dummy_Saddle_ProLogo_Scratch']),cap=bb(['head tube cap001 RVBU']).getCenter(V(0,0)),hood=bb(['RightHandle']),arm=bb(['crank armLogo2']).getCenter(V(0,0));
 const LAM=18*Math.PI/180;  // 72 deg head angle: puts the front hub 48.6 mm (fork rake) ahead of the steering axis through the headset cap
 const HT=o2(cap),HB=HT.clone().add(V(Math.sin(LAM)*0.15,-Math.cos(LAM)*0.15));
 return {name:'Colnago V4Rs',decal:'COLNAGO',paint:null,R,LAM,off,RH:o2(RH),FH:o2(FH),BB:o2(BB),SAD:o2(V((sad.min.x+sad.max.x)/2,sad.max.y)),HT,HB,
  HOOD:o2(V((hood.min.x+hood.max.x)/2,hood.max.y)),hoodZ:(hood.min.z+hood.max.z)/2,crankLen:0.1725,crank0:Math.atan2(arm.y-BB.y,arm.x-BB.x)}}
function fitFrom(G){ // rider sized to the bike: hip over the saddle, hands on the hoods, knee slightly bent at the bottom of the stroke
 const hip=V(G.SAD.x-0.03,G.SAD.y+0.085),H=V(G.HOOD.x,G.HOOD.y+0.02),torso=0.56,armLen=0.60,reach=armLen*0.93;
 const d=hip.distanceTo(H),a=(torso*torso-reach*reach+d*d)/(2*d),h=Math.sqrt(Math.max(torso*torso-a*a,0));
 const ex=H.clone().sub(hip).normalize(),ey=V(-ex.y,ex.x),sh=hip.clone().addScaledVector(ex,a).addScaledVector(ey,h);
 const t=sh.clone().sub(hip).normalize(),head=sh.clone().addScaledVector(t,0.13).add(V(0.06,0.05));
 return {hip,sh,head,armLen,legLen:1.04*(hip.distanceTo(G.BB)+G.crankLen)}}
// ---------- the real model, rigged: fork + bars + front wheel steer about the head-tube axis; wheels and crank spin
function rigModel(){const m=MODEL,g=G,root=new THREE.Group();root.add(m);m.position.copy(g.off);root.updateMatrixWorld(true);
 const by=n=>m.getObjectByName(SN(n));
 const mk=(parent,at)=>{const p=new THREE.Group();p.position.copy(at);parent.add(p);root.updateMatrixWorld(true);return p};
 const group=(at,names)=>{const p=mk(root,at);for(const n of names){const o=by(n);if(o){root.updateMatrixWorld(true);p.attach(o)}}p.userData.q0=p.quaternion.clone();return p};
 const tilt=mk(root,g.HB);tilt.rotation.z=g.LAM;const steer=new THREE.Group();tilt.add(steer);root.updateMatrixWorld(true);
 const fs=group(g.FH,['Dummy_Wheel_BoraWTO_Front','disc_break_front01','disc_break_front02']);
 for(const n of['V4_510_SDM3_Fork004_RVBU','anchor_frame_fronthub','anchor_frame_handlebar','Stem top bracket RVBU','V4_510_SDM3_Club_Front_004_RVBU']){const o=by(n);if(o){root.updateMatrixWorld(true);steer.attach(o)}}
 root.updateMatrixWorld(true);steer.attach(fs);fs.userData.q0=fs.quaternion.clone();
 const rs=group(g.RH,['Dummy_Wheel_BoraWTO_Back','reardiscA','reardiscB','disc_break_rear03']);
 const cs=group(g.BB,['crank arm part 01','crank armLogo','crank armLogo2','CogOuter','CrankInner','CrankOuter','chain ring text']);
 const handL=mk(root,V(g.HOOD.x,g.HOOD.y,-g.hoodZ)),handR=mk(root,V(g.HOOD.x,g.HOOD.y,g.hoodZ));steer.attach(handL);steer.attach(handR);
 const paintMats=[];m.traverse(o=>{if(o.isMesh){o.castShadow=true;if(o.material&&o.material.name==='RVBU'&&!paintMats.includes(o.material))paintMats.push(o.material)}});
 for(const pm of paintMats){pm.userData.paint=true;pm.userData.orig=pm.color.getHex()}
 return {root,steer,fs,rs,cs,hands:[handL,handR]}}
// ---------- the procedural bike (ghost riders, or everyone when no model is embedded)
function procBike(bk){const g=G,roll=bk.roll,{RH,FH,BB,SAD,HT,HB,HOOD}=g;
 const SP=V(SAD.x+0.01,SAD.y-0.035),STt=BB.clone().lerp(SP,0.72),SSJ=BB.clone().lerp(STt,0.66);
 const rw=wheel(0.060,g.R);rw.position.copy(RH);roll.add(rw);
 const aero=new THREE.Group();aero.scale.z=0.5;roll.add(aero);
 tube(aero,HB,BB,0.038,paint);tube(aero,BB,STt,0.028,paint);tube(aero,STt,SP,0.017,black);tube(aero,STt,HT,0.024,paint);tube(aero,HT,HB,0.034,paint);
 for(const z of[-0.045,0.045]){tube(roll,V(BB.x,BB.y,z*0.6),V(RH.x,RH.y,z),0.011,paint);tube(roll,V(RH.x,RH.y+0.01,z),V(SSJ.x,SSJ.y,z*0.4),0.008,paint)}
 const saddle=new THREE.Mesh(new THREE.BoxGeometry(0.26,0.03,0.135),black);saddle.position.copy(SAD).add(V(-0.02,-0.015));roll.add(saddle);
 const bbShell=new THREE.Mesh(new THREE.CylinderGeometry(0.036,0.036,0.09,12),paint);bbShell.rotation.x=Math.PI/2;bbShell.position.copy(BB);roll.add(bbShell);
 const ring=lathe([[0.08,-0.0015],[0.105,-0.0015],[0.105,0.0015],[0.08,0.0015],[0.08,-0.0015]],steel,56);ring.position.copy(BB).add(V(0,0,0.062));roll.add(ring);
 const cass=new THREE.Mesh(new THREE.CylinderGeometry(0.048,0.03,0.04,24),steel);cass.rotation.x=Math.PI/2;cass.position.copy(RH).add(V(0,0,0.06));roll.add(cass);
 const chainPts=[];for(let i=0;i<=40;i++){const c=i<20?BB:RH,cr=i<20?0.105:0.045,ang=i<20?(Math.PI/2+i/19*Math.PI):(3*Math.PI/2+(i-20)/20*Math.PI);chainPts.push(V(c.x+cr*Math.cos(ang),c.y+cr*Math.sin(ang),0.06))}
 roll.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(chainPts,true),80,0.004,6,true),mat(0x777777,{metalness:.8,roughness:.4})));
 const dtDir=HB.clone().sub(BB).normalize(),dtMid=HB.clone().lerp(BB,0.55),dm=decal(g.decal,0.34,0.06);
 for(const side of[1,-1]){const pl=new THREE.Mesh(new THREE.PlaneGeometry(0.34,0.06),dm);pl.position.copy(dtMid).add(V(0,0,side*0.0205));pl.rotation.z=Math.atan2(dtDir.y,dtDir.x);if(side<0)pl.rotation.y=Math.PI;roll.add(pl)}
 const tilt=new THREE.Group();tilt.position.copy(HB);tilt.rotation.z=g.LAM;roll.add(tilt);const steer=new THREE.Group();tilt.add(steer);const un=new THREE.Group();un.rotation.z=-g.LAM;steer.add(un);
 const rel=v=>V(v.x-HB.x,v.y-HB.y,v.z||0),CR=V(0.012,-0.05);
 tube(un,V(0,0),CR,0.03,paint);for(const z of[-0.045,0.045])tube(un,V(CR.x,CR.y,z),V(FH.x-HB.x,FH.y-HB.y,z),0.012,paint);
 const fw=wheel(0.051,g.R);fw.position.copy(rel(FH));un.add(fw);
 const S0=rel(HT).add(V(-0.005,0.03)),SE=rel(V(HOOD.x-0.08,HOOD.y-0.01));tube(un,rel(HB),S0,0.02,black);tube(un,S0,SE,0.016,black);
 const tops=new THREE.Mesh(new THREE.BoxGeometry(0.05,0.02,2*g.hoodZ+0.02),black);tops.position.copy(SE);un.add(tops);
 const hands=[];for(const s of[-1,1]){const z=s*g.hoodZ,c0=V(SE.x,SE.y,z);const pts=[c0,c0.clone().add(V(0.09,0.005)),c0.clone().add(V(0.13,-0.03)),c0.clone().add(V(0.125,-0.09)),c0.clone().add(V(0.07,-0.13))];
  un.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),24,0.012,8,false),black));
  const hood=new THREE.Mesh(new THREE.BoxGeometry(0.08,0.035,0.03),black);hood.position.copy(rel(V(HOOD.x,HOOD.y-0.02,z)));un.add(hood);
  const mk=new THREE.Group();mk.position.copy(rel(V(HOOD.x,HOOD.y,z)));un.add(mk);hands.push(mk)}
 bk.proc={steer,rw,fw};bk.hands=hands;bk.crankTubes=true}
// ---------- rider + fly
function buildRider(bk){const F=FIT,roll=bk.roll,U=V(0,1,0);
 tube(roll,F.hip,F.sh,0.075,jersey);tube(roll,V(F.hip.x-0.05,F.hip.y+0.01),V(F.hip.x+0.05,F.hip.y-0.01),0.085,shorts);
 const helmet=new THREE.Mesh(new THREE.SphereGeometry(0.11,16,12),helmetM);helmet.position.copy(F.head).add(V(0,0.02));helmet.scale.set(1.15,0.95,1);helmet.castShadow=true;roll.add(helmet);
 const face=new THREE.Mesh(new THREE.SphereGeometry(0.085,12,10),skin);face.position.copy(F.head).add(V(0.02,-0.03));roll.add(face);
 for(const s of[-1,1]){const L={s,zH:0.09*s,zP:0.11*s,zC:0.075*s};L.thigh=tube(roll,F.hip,F.hip.clone().add(U),0.05,shorts);L.shin=tube(roll,F.hip,F.hip.clone().add(U),0.038,skin);
  L.shoe=new THREE.Mesh(new THREE.BoxGeometry(0.26,0.055,0.09),black);L.shoe.castShadow=true;roll.add(L.shoe);if(bk.crankTubes)L.crank=tube(roll,G.BB,G.BB.clone().add(U),0.012,black);bk.legs.push(L);
  const A={s,zS:0.18*s};A.upper=tube(roll,F.sh,F.sh.clone().add(U),0.036,jersey);A.fore=tube(roll,F.sh,F.sh.clone().add(U),0.03,skin);bk.arms.push(A)}
 const fly=new THREE.Group();fly.position.copy(F.head).add(V(-0.02,0.14));roll.add(fly);const body=new THREE.Mesh(new THREE.SphereGeometry(0.035,10,8),mat(0x3a2a1a,{roughness:.5}));body.scale.set(1.5,0.8,0.9);fly.add(body);
 const eyeM=mat(0xc0201a,{emissive:0x600000});for(const s of[-1,1]){const e=new THREE.Mesh(new THREE.SphereGeometry(0.012,8,6),eyeM);e.position.set(0.045,0.012,0.018*s);fly.add(e)}
 const wingM=new THREE.MeshStandardMaterial({color:0xdfe9ff,transparent:true,opacity:.45,side:THREE.DoubleSide});for(const s of[-1,1]){const w=new THREE.Mesh(new THREE.PlaneGeometry(0.075,0.03),wingM);w.position.set(-0.03,0.02,0.035*s);w.rotation.x=Math.PI/2;w.rotation.z=-0.5;fly.add(w)}
 const glow=new THREE.PointLight(0xff8040,0,0.6);glow.position.set(0,0.05,0);fly.add(glow);const halo=new THREE.Mesh(new THREE.SphereGeometry(0.06,10,8),new THREE.MeshBasicMaterial({color:0xffa060,transparent:true,opacity:0}));fly.add(halo);
 Object.assign(bk,{fly,glow,halo})}
function buildBike(ghost,real){const root=new THREE.Group(),roll=new THREE.Group();root.add(roll);const bk={root,roll,ghost,legs:[],arms:[],hands:[]};
 if(real){bk.rig=rigModel();roll.add(bk.rig.root);bk.hands=bk.rig.hands}else procBike(bk);
 buildRider(bk);
 if(ghost){root.traverse(o=>{if(o.isMesh){o.material=o.material.clone();o.material.transparent=true;o.material.opacity=0.15;o.castShadow=false}});bk.glow.intensity=0}
 return bk}
const _w=new THREE.Vector3();
function pose(bk,s,done,pop){const [x,y,psi,v,phi,delta]=s,F=FIT;bk.root.position.set(x,0,y);bk.root.rotation.y=-psi;
 bk.roll.rotation.x=done?(phi>=0?1:-1)*1.35:phi;
 const wa=x/G.R,ca=-wa/1.5;  // wheels and crank turn clockwise seen from the drive side; 1.5 gear ratio
 if(bk.rig){bk.rig.steer.rotation.y=-delta;spin(bk.rig.fs,-wa);spin(bk.rig.rs,-wa);spin(bk.rig.cs,ca)}
 else{bk.proc.steer.rotation.y=-delta;bk.proc.rw.rotation.z=-wa;bk.proc.fw.rotation.z=-wa}
 bk.root.updateMatrixWorld(true);
 for(const L of bk.legs){const a=G.crank0+ca+(L.s>0?0:Math.PI),ped=V(G.BB.x+G.crankLen*Math.cos(a),G.BB.y+G.crankLen*Math.sin(a),L.zP);
  if(L.crank)setTube(L.crank,V(G.BB.x,G.BB.y,L.zC),V(ped.x,ped.y,L.zC));
  const hip=V(F.hip.x,F.hip.y,L.zH),d=ped.clone().sub(hip),seg=F.legLen/2,half=Math.min(d.length()/2,seg*0.999);
  const knee=hip.clone().lerp(ped,0.5).addScaledVector(V(-d.y,d.x,0).normalize(),Math.sqrt(seg*seg-half*half));
  setTube(L.thigh,hip,knee);setTube(L.shin,knee,ped);L.shoe.position.copy(ped).add(V(0.035,-0.012,0))}
 for(const A of bk.arms){const shp=V(F.sh.x,F.sh.y,A.zS);bk.hands[A.s>0?1:0].getWorldPosition(_w);const hw=bk.roll.worldToLocal(_w.clone());
  const d=hw.clone().sub(shp),seg=F.armLen/2,half=Math.min(d.length()/2,seg*0.999);
  const elbow=shp.clone().lerp(hw,0.5).addScaledVector(V(d.y,-d.x,0).normalize(),Math.sqrt(seg*seg-half*half)).add(V(0,0,A.s*0.03));
  setTube(A.upper,shp,elbow);setTube(A.fore,elbow,hw)}
 const g=Math.min(1,(pop||0)/150000);if(!bk.ghost)bk.glow.intensity=0.4+1.2*g;bk.halo.material.opacity=0.08+0.3*g;bk.halo.scale.setScalar(1+0.6*g)}
function setPaint(hex){scene.traverse(o=>{if(o.isMesh&&o.material&&o.material.userData.paint){const m=o.material;if(hex==='orig'){if(m.userData.orig!==undefined)m.color.setHex(m.userData.orig);else m.color.setHex(G.paint||0xd21f2b).convertSRGBToLinear()}else m.color.setHex(hex).convertSRGBToLinear()}})}
// ---------- playback + cameras
let bikes=[],hero=null,shown=Math.min(B,24),rider=0,k=0,playing=false,camMode=+(Q.get('cam')||0),last=0,snap=true,showGhosts=Q.get('ghosts')!=='0';
const slider=document.getElementById('time'),rsel=document.getElementById('rider'),psel=document.getElementById('paint');
const barsEl=document.getElementById('bars'),pairs=[];for(let i=0;i<dnNames.length;i+=2){const row=document.createElement('div');row.className='bar';row.innerHTML='<span>'+dnNames[i].replace(/_L$/,'')+'</span><i></i><i class="r"></i>';barsEl.appendChild(row);pairs.push({l:i,r:i+1,el:row.querySelectorAll('i')})}
const camPos=V(-6,2,2),camTgt=V(0,0),tmp=V(0,0);
function frame(){const row=tr[k];for(let i=0;i<shown;i++){bikes[i].root.visible=showGhosts&&(i!==rider);if(bikes[i].root.visible)pose(bikes[i],row.state[i],row.done[i],0)}pose(hero,row.state[rider],row.done[rider],row.pop_hz?row.pop_hz[rider]:0);
 const s=row.state[rider],yaw=-s[2],fwd=V(Math.cos(yaw),0,-Math.sin(yaw)),right=V(Math.sin(yaw),0,Math.cos(yaw)),base=V(s[0],0,s[1]);
 if(camMode===0){tmp.copy(base).addScaledVector(fwd,-5.5).addScaledVector(right,1.6).add(V(0,1.9,0));camPos.lerp(tmp,snap?1:0.08);camTgt.lerp(base.clone().add(V(0,0.9,0)).addScaledVector(fwd,1.5),snap?1:0.15);camera.position.copy(camPos);camera.lookAt(camTgt)}
 else if(camMode===1){tmp.copy(base).addScaledVector(right,7).addScaledVector(fwd,1.5).add(V(0,1.3,0));camPos.lerp(tmp,snap?1:0.1);camTgt.lerp(base.clone().add(V(0,0.8,0)),snap?1:0.2);camera.position.copy(camPos);camera.lookAt(camTgt)}
 else if(camMode===3){tmp.copy(base).addScaledVector(right,2.6).addScaledVector(fwd,0.55).add(V(0,0.75,0));camPos.lerp(tmp,snap?1:0.1);camTgt.lerp(base.clone().add(V(0,0.62,0)).addScaledVector(fwd,0.5),snap?1:0.2);camera.position.copy(camPos);camera.lookAt(camTgt)}
 else{hero.fly.updateWorldMatrix(true,false);const p=V(0.05,0.06).applyMatrix4(hero.fly.matrixWorld),f=V(3,0.3).applyMatrix4(hero.fly.matrixWorld);camera.position.copy(p);camera.up.set(0,1,0).applyQuaternion(new THREE.Quaternion().setFromRotationMatrix(hero.roll.matrixWorld));camera.lookAt(f);camera.up.set(0,1,0)}
 sun.position.set(s[0]-20,40,25);sun.target.position.set(s[0],0,0);
 const $=id=>document.getElementById(id);
 $('v').textContent=(s[3]*3.6).toFixed(1);$('phi').textContent=(s[4]*57.3).toFixed(1)+'°';$('delta').textContent=(s[5]*57.3).toFixed(1)+'°';
 $('T').textContent=row.steer[rider].toFixed(2)+' Nm';$('P').textContent=row.power[rider].toFixed(0)+' W';$('cad').textContent=(row.done[rider]?0:s[3]/(2*Math.PI*G.R)/1.5*60).toFixed(0)+' rpm';$('x').textContent=s[0].toFixed(1)+' m';
 $('alive').textContent=row.done.filter(d=>!d).length+' / '+B;$('pop').textContent=row.pop_hz?Math.round(row.pop_hz[rider]).toLocaleString():'—';
 $('fall').style.display=row.done[rider]?'block':'none';$('clock').textContent=row.t.toFixed(2)+' s';slider.value=k;
 if(row.dn_hz){const d=row.dn_hz[rider];for(const p of pairs){p.el[0].style.transform='scaleX('+Math.min(1,d[p.l]/120)+')';p.el[1].style.transform='scaleX('+Math.min(1,d[p.r]/120)+')'}}
 snap=false;renderer.render(scene,camera);window.DONE=1}
function resize(){const w=innerWidth,h=innerHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}addEventListener('resize',resize);resize();
function loop(ts){if(playing){const sp=parseFloat(document.getElementById('speed').value);if(ts-last>dt*1000/sp){last=ts;k=Math.min(k+1,tr.length-1);if(k===tr.length-1){playing=false;document.getElementById('play').textContent='↺ 重播'}}}frame();requestAnimationFrame(loop)}
function start(){G=MODEL?modelGeo():procGeo();FIT=fitFrom(G);
 if(!MODEL)paint.color.setHex(G.paint).convertSRGBToLinear();else paint.color.setHex(0x2440c8).convertSRGBToLinear();
 for(let i=0;i<shown;i++){const bk=buildBike(true,false);scene.add(bk.root);bikes.push(bk)}hero=buildBike(false,!!MODEL);scene.add(hero.root);
 document.title='🪰 Fly rides a '+G.name;
 document.getElementById('title').textContent=G.name+' · '+D.mode+' · '+B+' 名骑手 · '+tr[tr.length-1].t.toFixed(1)+' s · 男性果蝇全中枢神经系统 166,700 神经元';
 const paints=MODEL?[['orig','原厂蓝'],[0xf2f2f2,'UAE 白'],[0xc8102e,'赛车红'],[0x16181c,'碳黑']]:[['orig','S-Works 红'],[0xf2f2f2,'白'],[0x16181c,'碳黑'],[0x2440c8,'蓝']];
 for(const [v,t] of paints){const o=document.createElement('option');o.value=v;o.textContent=t;psel.appendChild(o)}psel.onchange=e=>setPaint(e.target.value==='orig'?'orig':+e.target.value);
 if(Q.get('paint'))setPaint(+Q.get('paint'));
 for(let i=0;i<B;i++){const o=document.createElement('option');o.value=i;o.textContent='骑手 #'+i;rsel.appendChild(o)}
 rider=Math.min(B-1,+(Q.get('rider')??(D.best_rider||0)));rsel.value=rider;
 k=Math.min(tr.length-1,Math.round((+Q.get('t')||0)/dt));slider.max=tr.length-1;
 for(const j of[0,1,2,3])document.getElementById('cam'+j).classList.toggle('on',j===camMode);document.getElementById('ghosts').classList.toggle('on',showGhosts);
 requestAnimationFrame(loop)}
document.getElementById('play').onclick=()=>{if(k>=tr.length-1)k=0;playing=!playing;document.getElementById('play').textContent=playing?'⏸ 暂停':'▶ 播放'};
slider.oninput=e=>{k=+e.target.value;snap=true};rsel.onchange=e=>{rider=+e.target.value;snap=true};
document.getElementById('ghosts').onclick=e=>{showGhosts=!showGhosts;e.target.classList.toggle('on',showGhosts)};
for(const i of[0,1,2,3]){document.getElementById('cam'+i).onclick=()=>{camMode=i;snap=true;for(const j of[0,1,2,3])document.getElementById('cam'+j).classList.toggle('on',j===i)}}
addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();document.getElementById('play').click()}if('1234'.includes(e.key))document.getElementById('cam'+(+e.key-1)).click()});
// ---------- boot: decode the embedded model (if any), then build the scene
const b64=document.getElementById('bikeglb').textContent.trim();
if(b64){const bin=atob(b64),buf=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
 const gl=new THREE.GLTFLoader(),dl=new THREE.DRACOLoader();dl.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/libs/draco/');gl.setDRACOLoader(dl);
 gl.parse(buf.buffer,'',g=>{MODEL=g.scene;start()},e=>{console.warn('bike model failed, using the procedural bike',e);start()})}
else start();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default="results/ride_trace.json")
    ap.add_argument("out", nargs="?", default="results/ride_3d.html")
    ap.add_argument("--bike", default=str(DEFAULT_BIKE), help="GLB (Draco ok) to ride, or 'none' for the procedural Tarmac SL9")
    a = ap.parse_args()
    src, dst = ROOT / a.trace, ROOT / a.out
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
    bike = "" if a.bike == "none" or not Path(a.bike).exists() else base64.b64encode(Path(a.bike).read_bytes()).decode()
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")).replace("__BIKE__", bike)
    dst.write_text(html)
    print(f"{dst}  ({dst.stat().st_size / 1e6:.1f} MB, {len(data['trace'])} steps, {len(data['trace'][0]['state'])} riders, "
          f"bike: {Path(a.bike).name if bike else 'procedural Tarmac SL9'})")


if __name__ == "__main__":
    main()
