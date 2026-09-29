"""Validate a two-row manifest and describe the expected inference workflow."""
import csv
from pathlib import Path

def main(manifest=Path(__file__).with_name('patient_manifest.csv')):
    rows=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))
    required={'patient_id','mri_dir','tumor_mask','lymph_node_mask','wsi_path','annotation_path','label'}
    if not rows or any(set(r)!=required for r in rows): raise ValueError('Manifest columns are invalid')
    if any(r['label'] not in {'0','1'} for r in rows): raise ValueError('Labels must be 0 or 1')
    missing=[(r['patient_id'],k) for r in rows for k in required-{ 'patient_id','label'} if not Path(r[k]).exists()]
    if missing:
        print('Example raw images are not included. Replace placeholder paths in patient_manifest.csv with local study data.')
        return 0
    print('Manifest validated; construct PatientDataset, encode MRI and WSI branches, then call MCMINet.')
    return 0
if __name__=='__main__': raise SystemExit(main())
