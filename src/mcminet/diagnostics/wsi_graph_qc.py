"""Read-only spatial graph diagnostics in level-0 pixels; no policy selection."""
from mcminet.config import default

from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
import torch

from mcminet.data.qupath_annotation_adapter import bounded_overview, overlay_point
from mcminet.data.wsi_preprocessing import to_rgb, STRIDE, NATIVE_SIZE

LONG_EDGE_MULTIPLES = (2, 3, 5, 10)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    names = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max')
    if not values.size:
        return dict.fromkeys((*names, 'mean'), None)
    result = dict(zip(names, map(float, np.percentile(values, [0,10,25,50,75,90,95,99,100]))))
    result['mean'] = float(values.mean())
    return result


def graph_qc(coordinates, graph, *, mpp_x=None, mpp_y=None):
    """Canonical node order is untouched. Repeated input edges count in degrees.

    Unique undirected pairs are used only for connectivity. Empty edge distance
    distributions are reported as null; isolated nodes have max distance zero.
    Scalar micron conversion requires finite positive, effectively equal x/y MPP.
    """
    if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
        raise ValueError('coordinates must be nonempty [N,2]')
    if coordinates.is_complex() or coordinates.dtype == torch.bool or not torch.isfinite(coordinates).all():
        raise ValueError('coordinates must be finite real values')
    points = coordinates.detach().cpu().double().numpy()
    n = len(points)
    edge, distance = graph['edge_index'], graph['edge_distance']
    if edge.device.type != 'cpu' or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError('edge_index must be CPU long [2,E]')
    e = edge.shape[1]
    if distance.device.type != 'cpu' or distance.shape != (e,) or not torch.isfinite(distance).all() or (distance<0).any():
        raise ValueError('edge_distance must be finite nonnegative CPU [E]')
    if e and (int(edge.min()) < 0 or int(edge.max()) >= n):
        raise ValueError('edge index out of range')
    source, target = edge.numpy()
    dist = distance.detach().double().numpy()
    expected = np.linalg.norm(points[source]-points[target], axis=1)
    if not np.allclose(expected, dist, rtol=1e-10, atol=1e-7):
        raise ValueError('Distances do not match original coordinate units')
    outgoing = np.bincount(source, minlength=n)
    incoming = np.bincount(target, minlength=n)
    pairs = sorted({tuple(sorted((int(a),int(b)))) for a,b in zip(source,target)})
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a,b in pairs:
        ra,rb=find(a),find(b)
        if ra != rb:parent[max(ra,rb)] = min(ra,rb)
    groups = {}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    components = sorted(groups.values(), key=lambda x:(-len(x),x[0]))
    component_ids = np.empty(n,dtype=np.int64)
    for label,indices in enumerate(components):component_ids[indices]=label
    max_distance = np.zeros(n,dtype=np.float64)
    np.maximum.at(max_distance,source,dist)
    representative_mpp = None
    warnings=[]
    if mpp_x is not None and mpp_y is not None and np.isfinite([mpp_x,mpp_y]).all() and min(mpp_x,mpp_y)>0:
        if math.isclose(mpp_x,mpp_y,rel_tol=1e-9,abs_tol=0):
            representative_mpp=(mpp_x+mpp_y)/2
        else:warnings.append('Anisotropic MPP: scalar micron conversion omitted; manual review required')
    else:warnings.append('Missing/invalid MPP: physical-distance conversion omitted')
    longest=[]
    for i in sorted(range(n),key=lambda i:(-max_distance[i],i)):
        indices=np.flatnonzero(source==i)
        if not len(indices):continue
        j=min(indices,key=lambda j:(-dist[j],int(target[j]),int(j)))
        longest.append(dict(node_index=i,coordinate=points[i].tolist(),neighbor_index=int(target[j]),
            neighbor_coordinate=points[target[j]].tolist(),distance_pixels=float(dist[j]),
            distance_um=float(dist[j]*representative_mpp) if representative_mpp else None))
        if len(longest)==20:break
    long_edges={str(m):dict(threshold_pixels=m*STRIDE,count=int((dist>m*STRIDE).sum()),
                   percent=float(100*(dist>m*STRIDE).sum()/e) if e else 0.0) for m in LONG_EDGE_MULTIPLES}
    steps={name:dict(distance_pixels=STRIDE*multiple,
           directed_edge_count=int(np.isclose(dist,STRIDE*multiple,rtol=1e-10,atol=1e-6).sum()))
           for name,multiple in [('1',1),('sqrt2',math.sqrt(2)),('2',2),('sqrt5',math.sqrt(5)),
                                  ('sqrt8',math.sqrt(8)),('3',3),('5',5),('10',10)]}
    summary=dict(node_count=n,directed_edge_count=e,unique_undirected_pair_count=len(pairs),
        duplicate_directed_edge_count=e-len(set(zip(source.tolist(),target.tolist()))),
        input_self_loop_count=int((source==target).sum()),in_degree=distribution(incoming),
        out_degree=distribution(outgoing),zero_degree_nodes=int(((incoming+outgoing)==0).sum()),
        zero_in_degree_nodes=int((incoming==0).sum()),zero_out_degree_nodes=int((outgoing==0).sum()),
        connected_component_count=len(components),component_sizes=[len(x) for x in components],
        largest_component_size=len(components[0]),largest_component_fraction=len(components[0])/n,
        singleton_components=sum(len(x)==1 for x in components),edge_distance_pixels=distribution(dist),
        representative_mpp=representative_mpp,
        edge_distance_um=distribution(dist*representative_mpp) if representative_mpp else None,
        stride_pixels=STRIDE,grid_step_counts=steps,long_edges=long_edges,
        per_node_max_outgoing_distance_pixels=distribution(max_distance),longest_nodes=longest,
        graph_policy_review=('B. LONG-EDGE / CONNECTIVITY PATTERN REQUIRES MANUAL REVIEW'
            if long_edges['3']['count'] or len(components)>1 else 'A. NO OBVIOUS STRUCTURAL CONCERN FOR MANUAL REVIEW'),
        warnings=warnings)
    return dict(summary=summary,component_ids=component_ids,out_degree=outgoing,
                in_degree=incoming,per_node_max_outgoing_distance=max_distance)


def unique_edge_indices(graph):
    """Display only: remove reverse/duplicate pairs without changing input graph."""
    edge=graph['edge_index'].numpy();seen=set();indices=[]
    for i,(a,b) in enumerate(edge.T):
        pair=tuple(sorted((int(a),int(b))))
        if pair not in seen:seen.add(pair);indices.append(i)
    return indices


def write_graph_qc_images(patient_id, slide, coordinates, graph, qc, output,
                          *, roi_geometry=None, rejected_coordinates=None, graph_config=None):
    """Overview display uses centers; canonical graph always uses top-left xy.

    Native pyramid zooms show eight longest unique edges. Tissue/annotation-gap
    explanations are left for manual review, never inferred from these images.
    """
    graph_config = default("graph") if graph_config is None else graph_config
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    overview=bounded_overview(slide)
    points=coordinates.detach().cpu().numpy()
    centers=points+NATIVE_SIZE / 2
    display=[overlay_point(c,slide.dimensions,overview.size) for c in centers]
    edges=graph['edge_index'].numpy();dist=graph['edge_distance'].numpy()
    unique=unique_edge_indices(graph)
    sample=[unique[i] for i in np.linspace(0,len(unique)-1,min(600,len(unique))).round().astype(int)] if unique else []
    long=[i for i in unique if dist[i]>3*STRIDE]
    colors=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#ffe119']
    def draw_roi(draw, transform):
        if roi_geometry is None:return
        polygons=list(roi_geometry.geoms) if hasattr(roi_geometry,'geoms') else [roi_geometry]
        for polygon in polygons:
            if polygon.geom_type!='Polygon':continue
            for ring in (polygon.exterior,*polygon.interiors):
                draw.line([transform(xy) for xy in ring.coords],fill='#00ffff',width=1)
    rejected=np.asarray(rejected_coordinates).reshape(-1,2) if rejected_coordinates is not None else np.empty((0,2))
    paths=[]
    for name,selection in [('graph_overview',sample),('long_edge_overview',long),('connected_component_overview',[])]:
        im=overview.copy();draw=ImageDraw.Draw(im)
        draw_roi(draw,lambda xy:overlay_point(xy,slide.dimensions,overview.size))
        for i in selection:
            a,b=edges[:,i];draw.line((*display[a],*display[b]),fill='red' if name=='long_edge_overview' else '#00bfff',width=2)
        for i,(x,y) in enumerate(display):
            color=colors[int(qc['component_ids'][i])%len(colors)] if name=='connected_component_overview' else '#ffff00'
            draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
        if name=='long_edge_overview':
            # Draw on top of nodes and add a locator so a short physical edge is
            # visible even at whole-slide scale. The locator is not an extra edge.
            for i in selection:
                a,b=edges[:,i];pa,pb=display[a],display[b]
                draw.line((*pa,*pb),fill='red',width=3)
                x,y=(pa[0]+pb[0])/2,(pa[1]+pb[1])/2
                draw.ellipse((x-10,y-10,x+10,y+10),outline='red',width=2)
                label=f'{a}-{b}: {dist[i]:.1f} px'
                box=draw.textbbox((x+12,y-12),label)
                draw.rectangle(box,fill='white');draw.text((x+12,y-12),label,fill='red')
        header=f'{patient_id} {name}; centers DISPLAY ONLY; cyan=effective ROI; graph={graph_config}'
        detail=(f'All {len(long)} unique edges >{3*STRIDE} px; diagnostic only' if name=='long_edge_overview' else
                f'{len(sample)} deterministic unique edges shown' if name=='graph_overview' else
                'Component sizes: '+str(qc['summary']['component_sizes']))
        canvas=Image.new('RGB',(max(900,im.width),im.height+50),'white');canvas.paste(im,(0,50))
        d=ImageDraw.Draw(canvas);d.text((8,6),header,fill='black');d.text((8,25),detail,fill='black')
        f=f'{patient_id}_{name}.png';canvas.save(output/f);paths.append(f)
    longest=sorted(unique,key=lambda i:(-dist[i],int(edges[0,i]),int(edges[1,i])))[:8]
    canvas=Image.new('RGB',(1200,max(1,math.ceil(len(longest)/2))*440+30),'white')
    draw=ImageDraw.Draw(canvas);draw.text((8,8),f'{patient_id}: longest unique edges; native pyramid context; manual review only',fill='black')
    for slot,i in enumerate(longest):
        a,b=edges[:,i];lo=np.maximum(np.floor(np.minimum(centers[a],centers[b])-2*STRIDE),0).astype(int)
        hi=np.minimum(np.ceil(np.maximum(centers[a],centers[b])+2*STRIDE),slide.dimensions).astype(int)
        span=hi-lo; desired=max(span[0]/560,span[1]/340)
        level=max([j for j,d in enumerate(slide.level_downsamples) if d<=max(1,desired)] or [0])
        down=float(slide.level_downsamples[level]);size=tuple(np.maximum(1,np.ceil(span/down).astype(int)))
        im=to_rgb(slide.read_region(tuple(map(int,lo)),level,size));im.thumbnail((560,340),Image.Resampling.BILINEAR)
        d=ImageDraw.Draw(im)
        transform=lambda xy:((xy[0]-lo[0])/down*im.width/size[0],(xy[1]-lo[1])/down*im.height/size[1])
        draw_roi(d,transform)
        for center in centers:
            if np.all(center>=lo) and np.all(center<=hi):
                x0,y0=transform(center);d.ellipse((x0-1,y0-1,x0+1,y0+1),fill='yellow')
        for xy in rejected:
            center=xy+NATIVE_SIZE / 2
            if np.all(center>=lo) and np.all(center<=hi):
                x0,y0=transform(center);d.rectangle((x0-3,y0-3,x0+3,y0+3),outline='magenta',width=2)
        positions=[((centers[node][0]-lo[0])/down*im.width/size[0],(centers[node][1]-lo[1])/down*im.height/size[1]) for node in (a,b)]
        d.line((*positions[0],*positions[1]),fill='red',width=2)
        for node,(x,y) in zip((a,b),positions):
            d.ellipse((x-4,y-4,x+4,y+4),outline='yellow',width=2);d.text((x+4,y+4),str(node),fill='red')
        x,y=(slot%2)*600,30+(slot//2)*440;canvas.paste(im,(x,y))
        mpp=qc['summary']['representative_mpp'];um=f'{dist[i]*mpp:.2f} um' if mpp else 'um unavailable'
        draw.text((x+5,y+345),f'Nodes {a} -> {b}; {dist[i]:.2f} px; {um}\nxy {points[a].tolist()} -> {points[b].tolist()}\nCyan ROI; yellow retained; magenta background-rejected',fill='black')
    if not longest:draw.text((8,45),'No edges',fill='black')
    f=f'{patient_id}_longest_edge_examples.png';canvas.save(output/f);paths.append(f)
    return paths
