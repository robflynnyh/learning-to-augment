import json, torch, sys, traceback
from pathlib import Path
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

def normalize(x):
    if OmegaConf is not None:
        try:
            if OmegaConf.is_config(x):
                return OmegaConf.to_container(x, resolve=False)
        except Exception:
            pass
    return x

def slim(x, depth=0):
    x = normalize(x)
    if depth > 7: return "..."
    if isinstance(x, (str,int,float,bool)) or x is None: return x
    if isinstance(x, dict):
        out={}
        for k,v in x.items():
            ks=str(k)
            if any(s in ks.lower() for s in ["state", "weight", "bias"]):
                continue
            out[ks]=slim(v, depth+1)
        return out
    if isinstance(x, (list,tuple)):
        if len(x)>50: return [slim(v, depth+1) for v in x[:50]] + [f"...({len(x)} items)"]
        return [slim(v, depth+1) for v in x]
    return str(type(x).__name__)

def summarize(path):
    rec={"path":path}
    try:
        p=Path(path); rec["size"]=p.stat().st_size
        ck=torch.load(path, map_location="cpu", weights_only=False)
        rec["type"]=type(ck).__name__
        if isinstance(ck, dict):
            rec["keys"]=list(map(str, ck.keys()))[:50]
            for cfgkey in ["config","cfg","args","hparams","hyper_parameters"]:
                if cfgkey in ck:
                    rec[cfgkey]=slim(ck[cfgkey])
            sd=None
            for k in ["model_state_dict","state_dict","model","net","module"]:
                if k in ck and isinstance(ck[k], dict): sd=ck[k]; rec["state_dict_key"]=k; break
            if sd:
                rec["num_tensors"]=sum(1 for v in sd.values() if hasattr(v,"shape"))
                rec["tensor_prefixes"]=sorted({str(k).split('.')[0] for k in sd.keys()})[:30]
        else:
            rec["repr"]=repr(ck)[:500]
    except Exception as e:
        rec["error"]=repr(e)
        rec["traceback"]=traceback.format_exc()[-1000:]
    return rec

paths=[line.strip().split(maxsplit=1)[1] for line in sys.stdin if line.strip() and line.strip()[0].isdigit()]
for path in paths:
    print(json.dumps(summarize(path), default=str))
