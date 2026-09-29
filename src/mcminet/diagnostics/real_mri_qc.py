"""MRI-only geometry audit and display of exact validated preprocessing outputs."""
from dataclasses import asdict
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
from PIL import Image,ImageDraw
import torch

from mcminet.data.mri_preprocessing import load_mri_volume,align_mask,normalize_volume,peritumoral_ring


def roi_statistics(roi):
    return dict(shape=list(roi.shape),dtype=str(roi.dtype),device=str(roi.device),
                finite=bool(torch.isfinite(roi).all()),min=float(roi.min()),mean=float(roi.mean()),
                std=float(roi.std(unbiased=False)),max=float(roi.max()))


def write_mri_qc(sources,result,output):
    """Re-read using validated helpers for context; verify every display crop exactly.

    No new alignment, slice selection, ring generation or model-input transform.
    Percentile windowing and display enlargement only affect the PNG.
    """
    volume=load_mri_volume(sources.mri_dicom_dir)
    tumor,tr=align_mask(sources.tumor_mask_path,volume)
    ln,lr=align_mask(sources.lymph_node_mask_path,volume)
    normalized,mean,std=normalize_volume(volume.values)
    zt,zl=result.tumor_slice_index,result.lymph_node_slice_index
    ring,clipped=peritumoral_ring(tumor[:,:,zt],volume.spacing_mm[:2])
    if tr!=result.tumor_mask_resampled or lr!=result.lymph_node_mask_resampled or int(ring.sum())!=result.peritumoral_area:
        raise ValueError('QC reread differs from validated preprocessing')
    for roi,box,z in [(result.tumor_roi,result.tumor_bbox,zt),(result.peritumoral_roi,result.peritumoral_bbox,zt),(result.lymph_node_roi,result.lymph_node_bbox,zl)]:
        r0,r1,c0,c1=box
        expected=torch.from_numpy(normalized[r0:r1,c0:c1,z].copy()).float().unsqueeze(0)
        if not torch.equal(roi,expected):raise ValueError('QC/model crop mismatch')
    files=sorted(f for f in Path(sources.mri_dicom_dir).rglob('*') if f.is_file())
    header=pydicom.dcmread(str(files[0]),stop_before_pixels=True,force=True,specific_tags=['SeriesInstanceUID'])
    masks={}
    for name,path in [('tumor',sources.tumor_mask_path),('lymph_node',sources.lymph_node_mask_path)]:
        image=nib.load(str(path),mmap='r',keep_file_open=False)
        masks[name]=dict(shape=list(image.shape),spacing=list(map(float,image.header.get_zooms())),
                         spatial_unit=image.header.get_xyzt_units()[0],affine=image.affine.tolist())
    metadata={k:v for k,v in asdict(result).items() if k not in ('tumor_roi','peritumoral_roi','lymph_node_roi')}
    metadata.update(dicom_file_count=len(files),selected_series_identifier=str(header.SeriesInstanceUID),
        selected_dce_phase=None,dce_selection='No DCE identification; manifest directory is upstream-selected',
        source_masks=masks,working_shape=list(volume.values.shape),working_spacing_mm=list(volume.spacing_mm),
        working_affine_ras_mm=volume.affine_ras_mm.tolist(),mri_resampling_performed=False,
        mask_alignment='Validated affine-based signed permutation/flip or nearest-neighbor; flags indicate interpolation only',
        normalization='Stored slope/intercept then whole-volume population z-score including finite zero background',
        peritumoral_definition='2D center-distance <=4 mm, excludes tumor; ROI is full rectangular crop without pixel masking',
        crop_values_exactly_verified=True)
    lo,hi=np.percentile(normalized,[1,99])
    def gray(array):
        values=np.clip((array-lo)/(hi-lo),0,1)
        return Image.fromarray(np.rint(values*255).astype(np.uint8)).convert('RGB')
    def overlay(image,mask,color):
        a=np.array(image);a[mask]=np.rint(.55*a[mask]+.45*np.array(color)).astype(np.uint8)
        return Image.fromarray(a)
    ct=overlay(overlay(gray(normalized[:,:,zt]),ring,(0,255,255)),tumor[:,:,zt],(255,60,60))
    cl=overlay(gray(normalized[:,:,zl]),ln[:,:,zl],(30,255,80))
    for im,boxes in [(ct,[(result.tumor_bbox,'red'),(result.peritumoral_bbox,'cyan')]),(cl,[(result.lymph_node_bbox,'lime')])]:
        d=ImageDraw.Draw(im)
        for (r0,r1,c0,c1),color in boxes:d.rectangle((c0,r0,c1-1,r1-1),outline=color,width=1)
    canvas=Image.new('RGB',(1200,960),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f'{sources.patient_id} validated MRI ROI QC; manual review; no response labels',fill='black')
    draw.text((12,27),f'Display only: volume P1/P99 window [{lo:.3f},{hi:.3f}]; native array orientation (rows down, columns right)',fill='black')
    for i,(im,title) in enumerate([(ct,f'Tumor slice {zt}; red=tumor, cyan=4 mm ring'),(cl,f'LN slice {zl}; green=LN')]):
        im.thumbnail((570,430),Image.Resampling.NEAREST);canvas.paste(im,(i*600+15,75))
        draw.text((i*600+15,55),title,fill='black')
    items=[('Tumor',result.tumor_roi,result.tumor_bbox),('Peritumoral',result.peritumoral_roi,result.peritumoral_bbox),('Lymph node',result.lymph_node_roi,result.lymph_node_bbox)]
    for i,(name,roi,box) in enumerate(items):
        x=i*400+12;image=gray(roi[0].numpy());scale=min(375/image.width,300/image.height)
        image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.NEAREST)
        canvas.paste(image,(x,555));draw.text((x,520),f'{name} exact native ROI {list(roi.shape)}',fill='black')
        draw.text((x,865),f'bbox [r0,r1,c0,c1): {box}\nDisplay enlarged only; no model resize',fill='black')
    draw.text((12,920),'Peritumoral input is the ring-derived RECTANGLE, including every enclosed MRI pixel; no ring-only masking.',fill='black')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);filename=sources.patient_id+'_mri_roi_qc.png'
    canvas.save(output/filename)
    return metadata,filename
