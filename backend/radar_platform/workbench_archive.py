"""Read-only recovery of owned jobs and a small, explicit offline example registry."""
import json
import os
import re
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
DEMO_PATHS={
    **{f'demo-{name}':PROJECT/'.runtime/workbench_examples/2026-10-29'/name for name in ['points','terrain','occlusion','array','scan']},
    'demo-real-dem':PROJECT/'.runtime/stage2_20261023/campaign_v1/alpine_uniform_0_s11',
    'demo-land-cover':PROJECT/'.runtime/stage2_20261023/campaign_v1/alpine_tree_plus6_s11',
}

def runtime():return Path(os.environ.get('RADAR_RUNTIME_ROOT',PROJECT/'.runtime')).resolve()

def owned_root(job_id):
    if job_id in DEMO_PATHS:return DEMO_PATHS[job_id]
    if not re.fullmatch(r'[a-f0-9]{32}',job_id):raise KeyError(job_id)
    base=runtime()/'scene_jobs';candidate=base/job_id
    if candidate.is_symlink() or candidate.resolve().parent!=base.resolve():raise KeyError(job_id)
    return candidate

def recover(job_id):
    from .unified.jobs import Job
    from .unified.schema import Simulation
    root=owned_root(job_id)
    try:
        state_path=root/('run.json' if job_id in DEMO_PATHS else 'job.json')
        state=json.loads(state_path.read_text())
        request=Simulation.model_validate_json((root/'request.json').read_text())
        frames=[json.loads(p.read_text()) for p in sorted(root.glob('frame_*/metadata.json'))]
    except (OSError,ValueError) as exc:raise KeyError(job_id) from exc
    status=state.get('status','failed')
    interrupted=status in ('queued','running')
    if interrupted:status='cancelled'
    return Job(job_id,request,root,state.get('estimate',{}),request.sequence.frame_count,
               state.get('data_origin','ideal_simulation'),status=status,frames=frames,
               error='服务重启，计算已中断；保留已完成帧。' if interrupted else state.get('error'),
               created_at=state.get('created_at',state.get('created_utc','')))
