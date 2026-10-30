"""Plot the actual configured synthetic ridge; no new terrain is generated."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.font_manager import FontProperties
import numpy as np
from radar_platform.unified.schema import Terrain
from radar_platform.unified.scene import terrain_mesh

ROOT=Path(__file__).resolve().parents[1]
raw=json.loads((ROOT/'configs/unified/high_altitude_ridge_visibility.json').read_text())
terrain=Terrain.model_validate(raw['scene']['terrain'])
mesh,_=terrain_mesh(terrain)
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
plt.rcParams['axes.unicode_minus']=False
fig,(ax,side)=plt.subplots(1,2,figsize=(12,7),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained')
fig.suptitle('遮挡测试实际使用的理想山脊：高程图与中央剖面',fontsize=18)
v=mesh.vertices_m
tri=mtri.Triangulation(v[:,1],v[:,0],mesh.faces)
im=ax.tricontourf(tri,-v[:,2],levels=np.arange(0,601,25),cmap='YlOrBr')
ax.triplot(tri,color='0.35',lw=.5,alpha=.5)
ax.scatter(v[:,1],v[:,0],s=13,c='black',zorder=4,label='原始地形网格节点')
contours=ax.tricontour(tri,-v[:,2],levels=[200,400],colors='#705430',linewidths=.7)
ax.clabel(contours,fmt='%d m',fontsize=8)
for name,label,color in [('front','前方目标','#1476ba'),('behind','后方目标','#b91c38')]:
    target=next(t for t in raw['scene']['objects'] if t['id']==name)
    n,e,_=target['position_ned_m']
    ax.scatter(e,n,s=130,marker='*',c=color,edgecolors='white',linewidths=.7,zorder=6)
    ax.annotate(f'{label}\nN={n} m，E=0',xy=(e,n),xytext=(45,n+38),fontsize=10,
                color=color,bbox=dict(fc='white',ec='none',alpha=.85))
ax.set(xlabel='东向 E / m',ylabel='北向 N / m',title='俯视高程图（横纵比例相同）',xlim=(-330,330),ylim=(570,1530))
ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=9)
bar=fig.colorbar(im,ax=ax,shrink=.72,pad=.02,ticks=np.arange(0,601,100))
bar.set_label('相对地面高程 / m')
n=np.array(terrain.north_m);h=np.array(terrain.elevation_up_m)[1]
side.fill_between(n,0,h,color='#ce934e',alpha=.7)
side.plot(n,h,color='#713e18',lw=2)
side.scatter(n,h,s=25,c='black',zorder=4)
side.scatter([1000,1300],[0,0],s=130,marker='*',c=['#1476ba','#b91c38'],zorder=6)
side.annotate('前方目标',xy=(1000,0),xytext=(900,75),color='#1476ba',arrowprops=dict(arrowstyle='-',color='#1476ba'))
side.annotate('后方目标',xy=(1300,0),xytext=(1340,95),color='#b91c38',arrowprops=dict(arrowstyle='-',color='#b91c38'))
side.annotate('峰顶 600 m\nN=1200 m',xy=(1200,600),xytext=(1320,520),
              arrowprops=dict(arrowstyle='->',color='#713e18'),fontsize=11)
side.annotate('',xy=(1100,45),xytext=(1300,45),arrowprops=dict(arrowstyle='<->',color='black'))
side.text(1200,65,'山脊底宽 200 m',ha='center',fontsize=10)
side.set(xlabel='北向 N / m',ylabel='相对地面高程 / m',title='E=0 中央剖面（无垂直夸大）',xlim=(570,1530),ylim=(-30,690))
side.set_aspect('equal');side.grid(alpha=.2)
side.text(.02,-.22,'沿东西方向完全相同，形成一条长 600 m 的直线山脊。\n原始网格为 3×6 个节点、20 个三角面；色阶按面内线性插值显示。\n这是人为构造的陡坡压力测试，不是真实 DEM，也不是平滑山丘。',
          transform=side.transAxes,fontsize=11,va='top')
out=ROOT/'.runtime/suitability_20261022/ideal_ridge_elevation.png'
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180,bbox_inches='tight')
print(out)
