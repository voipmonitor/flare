"""Match a stored model DAG to a CUDA trace without loading checkpoint tensors."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys


def tensor(path, name, offset=0, length=None):
    with path.open('rb') as file:
        size, = struct.unpack('<Q',file.read(8))
        header = json.loads(file.read(size))
        begin,end = header[name]['data_offsets']
        count = end-begin-offset if length is None else length
        if offset<0 or count<0 or offset+count>end-begin:
            raise ValueError('resource lies outside the stored tensor')
        file.seek(8+size+begin+offset)
        data = file.read(count)
        if len(data)!=count:
            raise ValueError('stored tensor is truncated')
        return data


def decode_frame(data, expected):
    result = subprocess.run(['/usr/bin/zstd','-q','-d','-c'],input=data,stdout=subprocess.PIPE,check=True).stdout
    if len(result)!=expected:
        raise ValueError('decoded resource length differs from metadata')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--schema',type=Path,required=True)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--graph',default='speculative1x8-tokens4')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.schema.resolve()))
    import package_pb2 as package
    import runtime_package_pb2 as runtime

    primary=args.package/'model.safetensors'
    manifest=package.ProtectedPackage.FromString(tensor(primary,'flare.manifest'))
    locations=package.LocationSet.FromString(tensor(primary,'flare.locations'))
    locations={x.payload_id:x for x in locations.locations}
    rank=package.RankBody.FromString(manifest.ranks[0].body)
    graph,=[g for g in rank.graphs if g.name==args.graph]
    resource=rank.resources[graph.resource]
    location=locations[resource.stored.payload_id]
    path=primary if not location.file else args.package/location.file
    if not path.resolve().is_relative_to(args.package.resolve()):
        raise ValueError('graph resource leaves the package directory')
    data=tensor(path,location.tensor,resource.stored.offset,resource.stored.nbytes)
    chunks=[data[i:i+resource.chunk_nbytes] for i in range(0,len(data),resource.chunk_nbytes)]
    if [hashlib.sha256(x).digest() for x in chunks]!=list(resource.chunk_sha256):
        raise ValueError('graph resource hash differs from the verified manifest')
    data=decode_frame(data,resource.decoded_nbytes)
    cursor=0
    def record(cls):
        nonlocal cursor
        size,=struct.unpack_from('<I',data,cursor);cursor+=4
        value=cls.FromString(data[cursor:cursor+size]);cursor+=size
        return value
    nodes=[record(runtime.Node) for _ in range(graph.node_count)]
    edges=[record(runtime.Edge) for _ in range(graph.edge_count)]
    if cursor!=len(data):
        raise ValueError('graph resource has unparsed trailing data')
    symbols=[f.symbol for f in rank.functions]
    names=subprocess.run(['c++filt','-n'],input='\n'.join(symbols)+'\n',text=True,
                         stdout=subprocess.PIPE,check=True).stdout.splitlines()
    trace=json.loads(args.trace.read_text())
    events={int(e['args']['graph node id'])&0xffffffff:e for e in trace['traceEvents']
            if e.get('cat') in ('kernel','gpu_memcpy') and 'graph node id' in e.get('args',{})}
    mismatches=[]
    output=[]
    for index,node in enumerate(nodes):
        kind=node.WhichOneof('operation')
        value=dict(index=index,kind=kind)
        event=events.get(index)
        if kind=='kernel':
            k=node.kernel
            value.update(name=names[k.function],grid=list(k.grid),block=list(k.block),shared=k.shared)
            if event is None or event['cat']!='kernel':
                mismatches.append(dict(index=index,reason='kernel event missing'))
            else:
                fields=[value['name']==event['name'],value['grid']==event['args']['grid'],
                        value['block']==event['args']['block'],value['shared']==event['args']['shared memory']]
                if not all(fields):
                    mismatches.append(dict(index=index,reason='kernel signature differs',fields=fields,
                                           stored=value['name'],traced=event['name']))
        elif kind=='copy':
            value['bytes']=node.copy.nbytes
            if event is None or event['cat']!='gpu_memcpy' or event['args'].get('bytes')!=node.copy.nbytes:
                mismatches.append(dict(index=index,reason='copy differs',stored=node.copy.nbytes,event=event))
        if event is not None:
            value.update(start_us=event['ts'],duration_us=event['dur'],stream=event['args']['stream'])
        output.append(value)
    report=dict(package=str(args.package),graph=args.graph,trace=str(args.trace),
                node_count=len(nodes),edge_count=len(edges),trace_events=len(events),
                kinds=dict(Counter(n['kind'] for n in output)),signature_mismatches=len(mismatches),
                edge_data=dict(Counter(e.data.hex() for e in edges)))
    (args.output/'match.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'mismatches.json').write_text(json.dumps(mismatches,indent=2)+'\n')
    (args.output/'dag.json').write_text(json.dumps(dict(nodes=output,
        edges=[dict(source=e.source,destination=e.destination,data=e.data.hex()) for e in edges]),indent=2)+'\n')
    c=rank.compressed_prepared_weights
    weights=package.PreparedWeightCatalog.FromString(decode_frame(c.zstd,c.decoded_nbytes)) if c.zstd else rank.prepared_weights
    descriptions=[dict(name=v.name,group=v.group,dtype=v.dtype,shape=list(v.shape),stride=list(v.stride),
                       allocation=v.allocation,offset=v.offset,nbytes=v.nbytes,layout=v.layout_type,
                       padding_bytes=sum(x.nbytes for x in v.padding)) for v in weights.views]
    (args.output/'weight-views.json').write_text(json.dumps(descriptions,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
