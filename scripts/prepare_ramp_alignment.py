#!/usr/bin/env python3
"""Metadata-only selection, then bounded extraction from the existing local ZIP."""
import argparse,csv,hashlib,io,json
from collections import Counter
from pathlib import Path
import zipfile

SEQUENCES={
    '2019_04_09_bms1000':'calibration',
    '2019_04_09_cms1000':'calibration',
    '2019_04_09_pms1000':'calibration',
    '2019_04_09_css1000':'development',
    '2019_04_30_mlms000':'development',
    '2019_05_09_cm1s003':'heldout_record',
    '2019_05_09_pcms002':'heldout_record',
    '2019_05_29_bcms000':'heldout_record',
}


def frame_number(name):return int(Path(name).stem)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    if a.output.resolve()==project or project in a.output.resolve().parents:raise ValueError('dataset extraction must remain outside source')
    a.output.mkdir(parents=True,exist_ok=True)
    records=[];inventory=[];labels={};radar_names={}
    with zipfile.ZipFile(a.root/'raw/Automotive.zip') as z:
        names=z.namelist()
        for seq,role in SEQUENCES.items():
            prefix='Automotive/'+seq+'/'
            ln=sorted(n for n in names if n.startswith(prefix+'text_labels/') and n.endswith('.csv'))
            rn=sorted((n for n in names if n.startswith(prefix+'radar_raw_frame/') and n.endswith('.mat')),key=frame_number)
            radar_names[seq]=rn
            available_labels={frame_number(n):n for n in ln};rn=[n for n in rn if frame_number(n) in available_labels]
            counts=Counter()
            for n in ln:
                content=z.read(n);labels[n]=content
                for row in csv.reader(io.StringIO(content.decode('utf-8-sig'))):
                    if len(row)>=6:counts[int(float(row[1]))]+=1
            # Three nonoverlapping 20-frame windows centered on fixed quantiles.
            selected=set()
            for q in (.25,.5,.75):
                center=int(round((len(rn)-1)*q))
                selected.update(rn[max(0,center-10):min(len(rn),center+10)])
            # Preserve the previously viewed 60 cyclist frames as regression only,
            # never silently call them a new final holdout.
            if seq=='2019_04_09_bms1000':selected.update(n for n in rn if 3<=frame_number(n)<=62)
            for n in sorted(selected,key=frame_number):
                index=frame_number(n)
                records.append({'sequence':seq,'role':role,'frame_id':n,'frame_index':index,
                                'label_file':available_labels[index],'previously_viewed':seq=='2019_04_09_bms1000' and 3<=index<=62})
            inventory.append({'sequence':seq,'role':role,'available_radar_label_pairs':len(rn),'selected_frames':len(selected),'label_instances_by_class':dict(counts)})
        plan={'selection_rule':'sequences fixed by class/date before radar scoring; 20 frames at 25/50/75 percent plus old 60-frame cyclist regression',
              'annotation_context':'all labels in selected sequences extracted for trajectory INPUT, not radar fitting',
              'split_unit':'recording sequence; heldout radar must not be used to choose alignment parameters',
              'inventory':inventory,'frames':records}
        (a.output/'selection.json').write_text(json.dumps(plan,indent=2)+'\n')
        wanted={r['frame_id'] for r in records}|set(labels)
        files=[]
        for i,name in enumerate(sorted(wanted)):
            path=(a.output/name).resolve()
            if a.output.resolve() not in path.parents:raise ValueError('unsafe archive member')
            if not path.exists():
                content=labels[name] if name in labels else z.read(name)
                path.parent.mkdir(parents=True,exist_ok=True)
                part=path.with_suffix(path.suffix+'.part');part.write_bytes(content);part.replace(path)
            files.append({'file':name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            if i%1000==0:print('extracted',i,'/',len(wanted),flush=True)
        (a.output/'manifest.json').write_text(json.dumps({'archive':str(a.root/'raw/Automotive.zip'),'files':files,
            'radar_frames':len(records),'source':'https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection'},indent=2)+'\n')
    print(json.dumps({'frames':len(records),'sequences':inventory},indent=2))


if __name__=='__main__':main()
