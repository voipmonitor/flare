"""Native coding decode with grouped speculative token timestamps and acceptance counters."""

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request


PROMPTS = (
    ("lru", "Write a complete Python LRU cache implementation with get, put, eviction and capacity validation. "
        "Include unittest coverage for edge cases. Return the complete module as code."),
    ("csv", "Implement an incremental Python CSV parser that accepts arbitrary text chunks, quoted fields, "
        "escaped quotes, CRLF and embedded newlines. Include tests for chunk-boundary cases. Return code."),
    ("html", "Create a single self-contained HTML file showing an interactive animated particle field with "
        "mouse attraction, attractive colors, a pause button and responsive resizing. Use no external assets."),
)


def request(base,path,payload=None):
    req=urllib.request.Request(base.rstrip('/')+path,
        data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as response:
        data=response.read().decode()
    return data if path=='/metrics' else json.loads(data)


def metrics(text):
    values={}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            key,value=line.rsplit(' ',1)
            values[key]=float(value)
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:8016')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--seed',type=int,default=0,help='explicit uint64 request seed (default: 0)')
    args=p.parse_args()
    if not 0 <= args.seed < 2**64:
        p.error('--seed must be a uint64 integer')
    args.output.mkdir(parents=True,exist_ok=False)
    model=request(args.base,'/v1/models')['data'][0]['id']
    results=[]
    for name,prompt in PROMPTS:
        before_text=request(args.base,'/metrics')
        before=metrics(before_text)
        if before.get('flare_active_requests',0) or before.get('flare_pending_requests',0):
            raise RuntimeError('Decode measurement requires an idle server')
        payload=dict(model=model,messages=[dict(role='user',content=prompt)],reasoning_effort='low',
            max_tokens=args.max_tokens,seed=args.seed,tool_choice='none',return_token_ids=True,return_token_timestamps=True)
        start=time.monotonic()
        response=request(args.base,'/v1/chat/completions',payload)
        wall=time.monotonic()-start
        after_text=request(args.base,'/metrics')
        after=metrics(after_text)
        (args.output/(name+'.request.json')).write_text(json.dumps(payload,indent=2))
        (args.output/(name+'.response.json')).write_text(json.dumps(response,indent=2))
        (args.output/(name+'.metrics-before.txt')).write_text(before_text)
        (args.output/(name+'.metrics-after.txt')).write_text(after_text)
        choice=response['choices'][0]
        ids,times=choice['token_ids'],choice['token_timestamps_ns']
        if len(ids)!=len(times) or len(ids)<32 or len(ids)!=response['usage']['completion_tokens']:
            raise ValueError('Raw token observations must close against completion usage')
        if any(b<a for a,b in zip(times,times[1:])) or times[-1]<=times[0]:
            raise ValueError('Token completion timestamps must be nondecreasing with positive decode time')
        first=sum(t==times[0] for t in times)
        seconds=(times[-1]-times[0])/1e9
        delta={key:value-before.get(key,0) for key,value in after.items() if key.startswith('flare_speculative_')}
        def total(name):
            return sum(value for key,value in delta.items() if key.split('{',1)[0]=='flare_speculative_'+name+'_total')
        rounds,emitted,offered,accepted=(total(name) for name in ('rounds','emitted_tokens','offered_drafts','accepted_drafts'))
        content=choice['message'].get('content') or ''
        (args.output/(name+'.content.txt')).write_text(content)
        result=dict(workload=name,generated_tokens=len(ids),first_timestamp_tokens=first,decode_seconds=seconds,
            decode_tokens_per_second=(len(ids)-first)/seconds,wall_seconds=wall,finish_reason=choice['finish_reason'],
            usage=response['usage'],speculative_rounds=rounds,mean_emitted_per_round=emitted/rounds if rounds else None,
            offered=offered,accepted=accepted,acceptance=accepted/offered if offered else None,
            mean_round_gpu_ms=total('gpu_nanoseconds')/rounds/1e6 if rounds else None,counters=delta)
        results.append(result)
        print(json.dumps({key:value for key,value in result.items() if key!='counters'}),flush=True)
    summary=dict(model=model,mode=args.mode,method='(N-first timestamp group)/(last-first native completion timestamp)',
        measured_interval='decode only; excludes prefill',sampling='recipe temperature/top-p defaults; explicit request seed',seed=args.seed,
        reasoning_effort='low',tool_choice='none',workloads=results,
        median_decode_tokens_per_second=statistics.median(x['decode_tokens_per_second'] for x in results))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'median_decode_tokens_per_second':summary['median_decode_tokens_per_second']}),flush=True)


if __name__=='__main__':
    main()
