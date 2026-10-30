#!/usr/bin/env python3
"""Render actual acquired radar frames with continuous simulated flight truth."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.navigation.dynamics import body_to_neu
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import object_mesh


def db_image(power):
    return 10*np.log10(np.maximum(power,1e-30)/max(float(np.max(power)),1e-30))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--preview',type=int)
    args=parser.parse_args();root=local_path(args.run)
    protocol=read_json(root/'protocol.json');cfg=Simulation.model_validate(read_json(root/'scene.json'))
    scope_file=root/'render_scope.json'
    if scope_file.exists():
        rendering_scope=read_json(scope_file)
    else:
        rendering_scope=dict(development_only=False)
        if protocol['mode']=='learned':
            training=read_json(local_path(protocol['model_dir'])/'training_protocol.json')
            if 'development_only' in training:
                rendering_scope['development_only']=training['development_only']
            else:
                source_protocol=read_json(local_path(training['data'])/'protocol.json')
                rendering_scope['development_only']=any(r.get('formal_split')=='train' and r['split']=='val' for r in source_protocol['records'])
        write_json(scope_file,rendering_scope)
    complete=read_json(root/'complete.json') if (root/'complete.json').exists() else None
    if complete is None and args.preview is None:raise ValueError('episode not complete; use --preview for a marked development image')
    frames=sorted((root/'frames').glob('*.npz'))
    if args.preview is not None:frames=frames[:args.preview]
    controls=[json.loads(line) for line in (root/'controls.jsonl').read_text().splitlines() if line]
    truths={int(f.stem):read_json(f) for f in (root/'truth').glob('*.json')}
    trace=read_json(root/'truth_trajectory.json') if (root/'truth_trajectory.json').exists() else [truths[k] for k in sorted(truths)]
    positions=np.array([t['position_neu_m'] for t in trace]);times=np.array([t['time_s'] for t in trace])
    velocities=np.array([t['velocity_neu_mps'] for t in trace])
    predictions=[]
    for row in controls:
        if 'relative_position_body_m' not in row['prediction']:continue
        truth=truths[row['frame']];r=np.array(row['prediction']['relative_position_body_m'])
        # Display-only world rotation; neither this transform nor true yaw enters inference/control.
        displayed=-body_to_neu(truth['yaw_rad'])@r
        predictions.append((row['time_s'],displayed,row))
    terrain=cfg.scene.terrain;n=np.array(terrain.north_m);e=np.array(terrain.east_m);h=np.array(terrain.elevation_up_m)
    ni=np.flatnonzero(abs(n)<=60);ei=np.flatnonzero(abs(e)<=60)
    nn,ee=np.meshgrid(n[ni],e[ei]);hh=h[np.ix_(ei,ni)]
    meshes=[]
    for obj in cfg.scene.objects:
        if obj.kind=='box':meshes.append(object_mesh(obj).triangles*[1,1,-1])
    out=root/('preview_frames' if args.preview is not None else 'rendered_frames');out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.facecolor':'#f5f7fb','axes.facecolor':'white'})
    paths=[]
    for frame_path in frames:
        index=int(frame_path.stem);truth=truths[index];now=truth['time_s'];p=np.array(truth['position_neu_m'])
        with np.load(frame_path,allow_pickle=False) as f:
            power=f['magnitude']**2;raxis=f['range_m'];vaxis=f['velocity_mps'];az=f['azimuth_deg'];el=f['elevation_deg']
        mask=times<=now+1e-8;past=positions[mask]
        available=[row for row in controls if row['time_s']<=now+1e-8]
        latest=available[-1] if available else None
        fig=plt.figure(figsize=(16,9),dpi=100);grid=fig.add_gridspec(2,3,hspace=.31,wspace=.48)
        ax=fig.add_subplot(grid[0,0],projection='3d')
        ax.plot_surface(nn,ee,hh,cmap='terrain',alpha=.58,linewidth=.15,edgecolor='#9da791')
        for triangles in meshes:ax.add_collection3d(Poly3DCollection(triangles,facecolor='#b9c8d9',edgecolor='#344759',linewidth=.35,alpha=.9))
        ax.plot(past[:,0],past[:,1],past[:,2],color='#1660c9',lw=2,label='Actual flight')
        ax.scatter(*p,color='#e87425',s=40);ax.scatter(0,0,30,marker='*',s=100,color='#159b61',label='Hover target')
        for local in [np.array([[2.,2.,0.],[-2.,-2.,0.]]),np.array([[2.,-2.,0.],[-2.,2.,0.]])]:
            arms=local@body_to_neu(truth['yaw_rad']).T+p
            ax.plot(arms[:,0],arms[:,1],arms[:,2],color='#d05e18',lw=2.2)
            ax.scatter(arms[:,0],arms[:,1],arms[:,2],s=22,facecolors='none',edgecolors='#d05e18')
        pred=[q for q in predictions if q[0]<=now+1e-8]
        if pred:
            label='RDA estimate*' if protocol['mode']=='learned' else 'Truth reference'
            pp=np.array([q[1] for q in pred]);ax.plot(pp[:,0],pp[:,1],pp[:,2],color='#bf4b95',ls='--',lw=1,label=label)
        ax.set(xlim=(-60,60),ylim=(-60,60),zlim=(-7,65),xlabel='North [m]',ylabel='East [m]',zlabel='Up [m]',title='Real DSM + 48-triangle landmark')
        ax.set_box_aspect((120,120,72));ax.view_init(elev=29,azim=-55);ax.legend(loc='upper left',fontsize=7)
        ax=fig.add_subplot(grid[0,1]);ax.contourf(ee,nn,hh,levels=16,cmap='terrain',alpha=.5)
        if meshes:ax.add_patch(Rectangle((-4,-3),8,6,facecolor='#a8b9cc',edgecolor='#33485a',lw=1.5))
        ax.plot(past[:,1],past[:,0],color='#1660c9',lw=2);ax.scatter(0,0,marker='*',s=110,c='#159b61')
        ax.scatter(p[1],p[0],color='#e87425',s=45)
        heading=body_to_neu(truth['yaw_rad'])@np.array([8.,0,0])
        ax.arrow(p[1],p[0],heading[1],heading[0],width=.5,color='#e87425',length_includes_head=True)
        ax.set(xlim=(-60,60),ylim=(-60,60),xlabel='East [m]',ylabel='North [m]',title='Heading rotates with the radar');ax.set_aspect('equal');ax.grid(alpha=.2)
        ax=fig.add_subplot(grid[0,2]);angular=db_image(power.max(axis=(0,1)))
        im=ax.imshow(angular,origin='lower',aspect='auto',extent=[az[0],az[-1],el[0],el[-1]],vmin=-45,vmax=0,cmap='magma')
        ax.set(xlabel='Radar azimuth [deg]',ylabel='Radar elevation [deg]',title='Two-angle radar image (max over R,D)')
        fig.colorbar(im,ax=ax,label='Relative power [dB]',fraction=.046)
        ax=fig.add_subplot(grid[1,0]);rd=db_image(power.max(axis=(2,3)))
        im=ax.imshow(rd,origin='lower',aspect='auto',extent=[vaxis[0],vaxis[-1],raxis[0],raxis[-1]],vmin=-45,vmax=0,cmap='magma')
        ax.set(xlabel='Closing velocity [m/s]',ylabel='Range [m]',title='Range-Doppler (max over both angles)')
        fig.colorbar(im,ax=ax,label='Relative power [dB]',fraction=.046)
        ax=fig.add_subplot(grid[1,1]);tt=times[mask]
        horizontal=np.linalg.norm(past[:,:2],axis=1);height=abs(past[:,2]-30)
        ax.plot(tt,horizontal,label='Horizontal error',color='#1660c9');ax.plot(tt,height,label='Height error',color='#bf4b95')
        ax.axhline(2,color='#1660c9',ls=':',alpha=.4);ax.axhline(1.5,color='#bf4b95',ls=':',alpha=.4)
        ax.set(xlim=(0,max(16,float(times[-1]))),ylim=(0,max(35,float(np.max(horizontal))*1.1)),xlabel='Simulation time [s]',ylabel='Error [m]',title='Closed-loop convergence (evaluation truth)')
        speed_ax=ax.twinx();speed_ax.plot(tt,np.linalg.norm(velocities[mask],axis=1),color='#de8f1d',alpha=.7,label='Speed');speed_ax.set_ylabel('Speed [m/s]')
        ax.legend(loc='upper right',fontsize=8);speed_ax.legend(loc='center right',fontsize=8);ax.grid(alpha=.2)
        ax=fig.add_subplot(grid[1,2]);ra=db_image(power.max(axis=(1,2)))
        im=ax.imshow(ra,origin='lower',aspect='auto',extent=[az[0],az[-1],raxis[0],raxis[-1]],vmin=-45,vmax=0,cmap='magma')
        ax.set(xlabel='Radar azimuth [deg]',ylabel='Range [m]',title='Range-azimuth (max over D,elevation)')
        fig.colorbar(im,ax=ax,label='Relative power [dB]',fraction=.046)
        mode={'learned':'RDA-only learned guidance','oracle':'TRUTH CONTROL REFERENCE','no_correction':'NO CORRECTION CONTROL'}[protocol['mode']]
        if rendering_scope['development_only']:mode+=' — DEVELOPMENT MODEL'
        if complete is None:mode+=' — DEVELOPMENT PREVIEW'
        status='initial acquisition' if latest is None else latest['control']['mode']
        fig.suptitle(f'{mode}  |  t = {now:.3f} s  |  {status}\n'
                     f'horizontal error {np.linalg.norm(p[:2]):.2f} m  |  height error {abs(p[2]-30):.2f} m  |  speed {np.linalg.norm(truth["velocity_neu_mps"]):.2f} m/s',fontsize=13,y=.99)
        note='Every radar panel is a physically synthesized acquired frame. '
        note+=('*World display of body-frame estimates uses simulation yaw only for plotting; guidance receives no true yaw.'
               if protocol['mode']=='learned' else 'This control uses truth or zero commands, and is not evidence of radar-only localization.')
        fig.text(.02,.012,note,fontsize=8,color='#526071')
        path=out/f'{index:04d}.png';fig.savefig(path);plt.close(fig);paths.append(path)
    if args.preview is not None:
        print(json.dumps(dict(preview_frames=[str(p) for p in paths])));return
    images=[Image.open(path).convert('RGB') for path in paths]
    duration=round(1000*protocol['spec']['radar_interval_s'])
    images[0].save(root/'flight.gif',save_all=True,append_images=images[1:],duration=duration,loop=0,optimize=False)
    import imageio_ffmpeg
    writer=imageio_ffmpeg.write_frames(str(root/'flight.mp4'),images[0].size,fps=1000/duration,codec='libx264',
                                      pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,macro_block_size=2)
    writer.send(None)
    for image in images:writer.send(np.asarray(image))
    writer.close()
    relative=[str(path.relative_to(root)) for path in paths]
    cards=''.join(f'<div class="card"><strong>{html.escape(key)}</strong><br>{html.escape(str(value))}</div>' for key,value in complete.items() if key in
        ['success','termination','horizontal_error_m','height_error_m','hover_target_error_3d_m','ground_target_distance_3d_m','final_speed_mps','truth_hold_duration_s','corrections','elapsed_simulation_s'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实DEM雷达闭环纠偏</title>
<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px;background:#f3f6fb;color:#203049}h1{font-size:28px}.cards{display:flex;flex-wrap:wrap;gap:12px}.card{background:white;padding:14px;border-radius:10px;min-width:190px}img{width:100%;border-radius:12px}button{padding:8px 20px}input{width:70%}pre{white-space:pre-wrap;background:white;padding:20px}a{color:#125ab0}</style>
<h1>真实 DEM 与中心建筑上的雷达闭环纠偏</h1><p>MODE</p><p>地面中心为定位原点，空中目标位于其上方30米。位置估计和控制使用机体系前／右／上坐标。世界轨迹中的估计位置仅在绘图时借助模拟航向转换，真实航向不输入定位器或控制器。</p>
<div class="cards">CARDS</div><p><button id="play">播放 / 暂停</button> <input id="slider" type="range" min="0" value="0"><span id="counter"></span></p><img id="frame">
<p>每一帧雷达图均来自实际回波合成；图像未做时间插值。动画播放速度为模拟时间。二维投影使用最大功率投影，颜色按每帧相对峰值标度。</p>
<p><a href="flight.mp4">MP4 视频</a> · <a href="flight.gif">GIF 动画</a> · <a href="complete.json">完整指标</a> · <a href="controls.jsonl">估计与控制日志</a> · <a href="protocol.json">实验配置</a></p>
<details><summary>实验边界与参数</summary><pre>PROTOCOL</pre></details>
<script>const frames=FRAMES;const slider=document.getElementById('slider'),img=document.getElementById('frame'),counter=document.getElementById('counter');slider.max=frames.length-1;function show(){img.src=frames[+slider.value];counter.textContent=` ${+slider.value+1} / ${frames.length}`;}slider.oninput=show;let playing=false;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%frames.length;show();}},DURATION);show();</script></html>'''
    mode_label={'learned':'仅雷达图驱动的学习定位与闭环控制。','oracle':'真值控制参考：用于验证运动与采样，不计入仅雷达定位的成功率。','no_correction':'无纠偏对照：不依据雷达观测调整运动。'}[protocol['mode']]
    if rendering_scope['development_only']:mode_label+=' 此处使用开发模型，开发留出样本来自正式训练池；不作为最终独立测试或批量验收结果。'
    page=page.replace('MODE',mode_label).replace('CARDS',cards).replace('PROTOCOL',html.escape(json.dumps(protocol,ensure_ascii=False,indent=2))).replace('FRAMES',json.dumps(relative)).replace('DURATION',str(duration))
    (root/'report.html').write_text(page)
    write_json(root/'render_manifest.json',dict(frames=len(paths),simulation_interval_s=duration/1000,
        radar_frames_interpolated=False,world_estimate_rotation='simulation yaw for display only',
        development_only=rendering_scope['development_only'],
        gif='flight.gif',mp4='flight.mp4',html='report.html'))
    print(json.dumps(dict(frames=len(paths),report=str(root/'report.html'))))


if __name__=='__main__':main()
