import { useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Auxiliary, SceneGeometry, SceneRequest, Vec3 } from "../workbenchTypes";

const xyz = (p: number[]) => new THREE.Vector3(p[1], -p[2], -p[0]);
const palette = [0xf6b855, 0x77b7eb, 0xc096ec, 0x7ed8b2, 0xed9d9f];
export type SceneLayer = "height" | "classes" | "illuminated" | "visible" | "echo";
interface Props { loading?: boolean; request: SceneRequest; geometry: SceneGeometry | null; auxiliary: Auxiliary | null; time: number; selected: string; hidden: string[]; layer: SceneLayer; wireframe: boolean; showBeam: boolean; showTrajectory: boolean; onSelect: (id: string) => void }
export default function Scene3D({ loading=false, request, geometry, auxiliary, time, selected, hidden, layer, wireframe, showBeam, showTrajectory, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null), controller = useRef<{ reset: (mode: string) => void; render: () => void }>();
  const [error, setError] = useState(""), [readout, setReadout] = useState("");
  const choose = useRef(onSelect); choose.current = onSelect;
  // A playback frame must not replace its canvas, WebGL context or camera.
  const surface = useRef<{renderer:THREE.WebGLRenderer;camera:THREE.PerspectiveCamera;controls:OrbitControls;initialized:boolean}>();
  useLayoutEffect(()=>{
    if(!host.current)return;
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});setError("");}
    catch{setError("当前浏览器无法创建 WebGL 三维视图，请启用硬件加速。");return;}
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.current.appendChild(renderer.domElement);
    const camera=new THREE.PerspectiveCamera(42,1,.01,1e8);
    const controls=new OrbitControls(camera,renderer.domElement);
    surface.current={renderer,camera,controls,initialized:false};
    return()=>{controller.current=undefined;surface.current=undefined;controls.dispose();renderer.dispose();renderer.domElement.remove();};
  },[]);
  useLayoutEffect(() => {
    if (!host.current || !surface.current) return;
    const element=host.current, {renderer,camera,controls}=surface.current;
    const scene=new THREE.Scene();scene.background=new THREE.Color(0x0e1c2c);
    const lights = new THREE.HemisphereLight(0xd9edff, 0x263928, 2.4);scene.add(lights);
    const sun = new THREE.DirectionalLight(0xffffff, 2);sun.position.set(100, 200, 80);scene.add(sun);
    const group = new THREE.Group(); scene.add(group);
    const clickable: THREE.Object3D[] = [];
    const platform = xyz(auxiliary?.platform_position_ned_m ?? request.platform.position_ned_m.map((v, i) => v + time * request.platform.velocity_ned_mps[i]));
    const all = [platform, ...request.scene.objects.map(o => xyz(o.position_ned_m.map((v,i)=>v+time*o.velocity_ned_mps[i])))];
    if (geometry?.terrain) { const t = geometry.terrain; all.push(xyz([t.north_m[0], t.east_m[0], 0]), xyz([t.north_m[t.north_m.length-1], t.east_m[t.east_m.length-1], 0])); }
    const bounds = new THREE.Box3().setFromPoints(all), center = bounds.getCenter(new THREE.Vector3());
    const extent = Math.max(10, bounds.getSize(new THREE.Vector3()).length()), markerSize = extent * .007;
    function mesh(geo: THREE.BufferGeometry, color: number, id?: string, material?: THREE.Material) {
      const m = new THREE.Mesh(geo, material ?? new THREE.MeshStandardMaterial({ color, roughness: .8, wireframe, side: THREE.DoubleSide }));
      if (id) { m.userData.id=id; clickable.push(m); }group.add(m);return m;
    }
    function line(points: THREE.Vector3[], color: number) { const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .8 }));group.add(l);return l; }
    const terrain = geometry?.terrain;
    if (terrain && !hidden.includes(request.scene.terrain?.id ?? "terrain")) {
      const { north_m: north, east_m: east, elevation_up_m: z } = terrain;
      const stride = Math.max(1, Math.ceil(Math.sqrt(north.length * east.length / 20000)));
      const ns = Array.from({length:Math.ceil((north.length-1)/stride)+1},(_,i)=>Math.min(i*stride,north.length-1));
      const es = Array.from({length:Math.ceil((east.length-1)/stride)+1},(_,i)=>Math.min(i*stride,east.length-1));
      const heights = z.flat();let min=Infinity,max=-Infinity;for(const h of heights){min=Math.min(min,h);max=Math.max(max,h);}
      const positions:number[] = [], colors:number[] = [];
      const classMap = request.scene.terrain?.land_cover_map, labels = [...new Set(classMap?.flat() ?? [])].sort();
      for(let j=0;j<es.length-1;j++)for(let i=0;i<ns.length-1;i++) {
        const cells = [[j,i],[j,i+1],[j+1,i],[j,i+1],[j+1,i+1],[j+1,i]];
        const label=classMap?.[es[j]]?.[ns[i]], ci=label?labels.indexOf(label):0;
        for(const [ej,ni] of cells){const h=z[es[ej]][ns[ni]],p=xyz([north[ns[ni]],east[es[ej]],-h]);positions.push(...p.toArray());const c=layer==="classes"?new THREE.Color().setHSL((ci*.173+.22)%1,.42,.5):new THREE.Color().setHSL(.43-(h-min)/Math.max(max-min,1)*.25,.27,.25+(h-min)/Math.max(max-min,1)*.25);colors.push(c.r,c.g,c.b);}
      }
      const geo = new THREE.BufferGeometry();geo.setAttribute("position",new THREE.Float32BufferAttribute(positions,3));geo.setAttribute("color",new THREE.Float32BufferAttribute(colors,3));geo.computeVertexNormals();
      mesh(geo,0xffffff,request.scene.terrain?.id,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,wireframe,transparent:layer!=="height"&&layer!=="classes",opacity:layer!=="height"&&layer!=="classes"?.35:1}));
    }
    request.scene.objects.forEach((o, index) => {
      if(hidden.includes(o.id))return;
      const pos=xyz(auxiliary?.objects.find(x=>x.id===o.id)?.position_ned_m ?? o.position_ned_m.map((v,i)=>v+time*o.velocity_ned_mps[i]));
      let geo:THREE.BufferGeometry;
      if(o.kind==="box"&&o.size_m)geo=new THREE.BoxGeometry(o.size_m[1],o.size_m[2],o.size_m[0]);
      else if(o.kind==="plate"&&o.size_m)geo=new THREE.PlaneGeometry(o.size_m[0],o.size_m[1]);
      else if(o.kind==="mesh"&&o.vertices_local_m&&o.faces){geo=new THREE.BufferGeometry();geo.setAttribute("position",new THREE.Float32BufferAttribute(o.vertices_local_m.flatMap(p=>xyz(p).toArray()),3));geo.setIndex(o.faces.flat());geo.computeVertexNormals();}
      else geo=new THREE.SphereGeometry(markerSize,12,8);
      const m=mesh(geo,selected===o.id?0xffffff:palette[index%palette.length],o.id);m.position.copy(pos);
      if(o.kind==="box"||o.kind==="mesh")m.rotateY(-(o.yaw_deg??0)*Math.PI/180);
      if(o.kind==="plate"&&o.normal_ned){const n=new THREE.Vector3(...o.normal_ned).normalize();const values=[Math.abs(n.x),Math.abs(n.y),Math.abs(n.z)];const index=values.indexOf(Math.min(...values));const hint=new THREE.Vector3(index===0?1:0,index===1?1:0,index===2?1:0);const u=hint.addScaledVector(n,-hint.dot(n)).normalize(),v=new THREE.Vector3().crossVectors(n,u);const basis=new THREE.Matrix4().makeBasis(xyz(u.toArray()),xyz(v.toArray()),xyz(n.toArray()));m.quaternion.setFromRotationMatrix(basis);m.applyMatrix4(new THREE.Matrix4().makeRotationY(-(o.yaw_deg??0)*Math.PI/180));m.position.copy(pos);}
      if(o.kind==="cloud"&&o.points_local_m){m.visible=false;const points=new THREE.Points(new THREE.BufferGeometry().setFromPoints(o.points_local_m.map(p=>xyz(p).add(pos))),new THREE.PointsMaterial({color:palette[index%palette.length],size:3,sizeAttenuation:false}));points.userData.id=o.id;group.add(points);clickable.push(points);}
      if(selected===o.id){line([platform,pos],0xffc675);setReadout(`${o.id} · 斜距 ${platform.distanceTo(pos).toFixed(2)} m`);}
      if(showTrajectory){const end=xyz(o.position_ned_m.map((v,i)=>v+(request.sequence.start_time_s+(request.sequence.frame_count-1)*request.sequence.frame_interval_s)*o.velocity_ned_mps[i]));line([xyz(o.position_ned_m),end],0x71899d);}
    });
    const radar=mesh(new THREE.OctahedronGeometry(markerSize*1.5),0x68e0d0,"platform");radar.position.copy(platform);radar.visible=!hidden.includes("platform");
    const velocity=xyz(auxiliary?.platform_velocity_ned_mps??request.platform.velocity_ned_mps);
    if(velocity.length()>0&&!hidden.includes("platform"))group.add(new THREE.ArrowHelper(velocity.clone().normalize(),platform,extent*.12,0x72d3c4));
    if(showTrajectory){const p=request.platform;const positions=p.trajectory?.position_ned_m ?? [p.position_ned_m,p.position_ned_m.map((v,i)=>v+(request.sequence.start_time_s+(request.sequence.frame_count-1)*request.sequence.frame_interval_s)*p.velocity_ned_mps[i])];line(positions.map(xyz),0x7ccec7);}
    if(showBeam&&!hidden.includes("platform")){
      if(auxiliary?.pattern){const p=auxiliary.pattern,geo=new THREE.BufferGeometry(),vertices=p.directions_ned.flatMap((d,i)=>xyz(d).multiplyScalar(p.normalized_power[i]*extent*.25).add(platform).toArray()),idx:number[]=[];for(let j=0;j<p.rows-1;j++)for(let i=0;i<p.cols-1;i++){const k=j*p.cols+i;idx.push(k,k+1,k+p.cols,k+1,k+p.cols+1,k+p.cols);}geo.setAttribute("position",new THREE.Float32BufferAttribute(vertices,3));geo.setIndex(idx);geo.computeVertexNormals();mesh(geo,0x4ed8d0,undefined,new THREE.MeshBasicMaterial({color:0x4ed8d0,transparent:true,opacity:.22,side:THREE.DoubleSide,depthWrite:false}));}
      else if(!request.radiation){const direction=xyz(auxiliary?.basis_ned.map(row=>row[0])??request.platform.boresight_ned).normalize();const length=extent*.5,angle=Math.min(85,request.platform.beam_half_angle_deg)*Math.PI/180;const cone=mesh(new THREE.ConeGeometry(length*Math.tan(angle),length,48,1,true),0x4ed8d0,undefined,new THREE.MeshBasicMaterial({color:0x4ed8d0,transparent:true,opacity:.08,side:THREE.DoubleSide,depthWrite:false}));cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,-1,0),direction);cone.position.copy(platform).addScaledVector(direction,length/2);line([platform,platform.clone().addScaledVector(direction,length)],0x65c9c4);}
    }
    if(auxiliary?.coverage&&!["height","classes"].includes(layer)){
      const c=auxiliary.coverage,counts=layer==="illuminated"?c.illuminated_tx_slots:layer==="visible"?c.unblocked_channel_slots:c.sampled_echo_channel_slots,den=layer==="illuminated"?c.tx_slot_denominator:c.channel_slot_denominator;
      const points:number[]=[],colors:number[]=[];c.positions_at_frame_start_ned_m.forEach((p,i)=>{if(hidden.includes(c.objects[c.object_index[i]]))return;points.push(...xyz(p).toArray());const color=new THREE.Color().setHSL(.02+counts[i]/den*.43,.9,.55);colors.push(color.r,color.g,color.b);});
      const geo=new THREE.BufferGeometry();geo.setAttribute("position",new THREE.Float32BufferAttribute(points,3));geo.setAttribute("color",new THREE.Float32BufferAttribute(colors,3));group.add(new THREE.Points(geo,new THREE.PointsMaterial({vertexColors:true,size:4,sizeAttenuation:false,depthTest:false})));
    }
    const grid=new THREE.GridHelper(extent,10,0x395067,0x23384b);grid.position.set(center.x,0,center.z);group.add(grid);
    const axes=new THREE.AxesHelper(extent*.12);axes.position.set(center.x-extent*.3,0,center.z+extent*.3);group.add(axes);
    function render(){renderer.render(scene,camera);}
    function reset(mode:string){controls.target.copy(mode==="follow"?platform:center);camera.up.set(0,1,0);const target=controls.target;if(mode==="top"){camera.position.copy(target).add(new THREE.Vector3(0,extent*1.9,.0001));camera.up.set(0,0,-1);}else if(mode==="side")camera.position.copy(target).add(new THREE.Vector3(extent*1.9,extent*.15,0));else camera.position.copy(target).add(new THREE.Vector3(.8,.6,.8).normalize().multiplyScalar(extent*1.9));camera.near=Math.max(.01,extent/100000);camera.far=extent*50;camera.updateProjectionMatrix();controls.update();render();}
    if(!surface.current.initialized){reset("default");surface.current.initialized=true;}
    controller.current={reset,render};
    const resize=new ResizeObserver(()=>{const w=element.clientWidth,h=element.clientHeight;renderer.setSize(w,h);camera.aspect=w/Math.max(1,h);camera.updateProjectionMatrix();render();});resize.observe(element);
    controls.addEventListener("change",render);const ray=new THREE.Raycaster();ray.params.Points!.threshold=markerSize;
    let down=[0,0];const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];};
    const click=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;const r=renderer.domElement.getBoundingClientRect();ray.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2),camera);const hit=ray.intersectObjects(clickable)[0];if(hit){choose.current(hit.object.userData.id);const p=hit.point;setReadout(`${hit.object.userData.id} · N ${(-p.z).toFixed(1)} / E ${p.x.toFixed(1)} / 高程 ${p.y.toFixed(1)} m`);}};
    renderer.domElement.addEventListener("pointerdown",pointerDown);renderer.domElement.addEventListener("pointerup",click);render();
    return()=>{resize.disconnect();controls.removeEventListener("change",render);renderer.domElement.removeEventListener("pointerdown",pointerDown);renderer.domElement.removeEventListener("pointerup",click);scene.traverse(o=>{const m=o as THREE.Mesh;m.geometry?.dispose();if(m.material)(Array.isArray(m.material)?m.material:[m.material]).forEach(x=>x.dispose());});};
  },[request,geometry,auxiliary,time,selected,hidden,layer,wireframe,showBeam,showTrajectory]);
  return <div className="lab-scene" data-testid="scene-3d"><div ref={host} className="lab-three-host" />{loading&&<div className="lab-frame-loading">正在载入所选帧的场景状态…</div>}{error&&<div className="lab-scene-error">{error}</div>}<div className="lab-camera-tools">{[["default","复位"],["top","俯视"],["side","侧视"],["follow","定位平台"]].map(([mode,label])=><button key={mode} onClick={()=>controller.current?.reset(mode)}>{label}</button>)}<button onClick={()=>{const c=host.current?.querySelector("canvas");if(c){const a=document.createElement("a");a.href=c.toDataURL();a.download="scene.png";a.click();}}}>截图</button></div><div className="lab-scene-caption">N 北 · E 东 · 高程向上 · 比例 1:1<br/>点目标与平台符号放大 · 速度箭头示方向{geometry&&` · 散射点预览 1/${geometry.preview_stride}`}<br/>{readout}</div></div>;
}
