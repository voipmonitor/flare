"""Identify a single CUDA kernel's ELF64 text section by its exact machine bytes."""
import hashlib
import struct

def cubin_code(image):
    if image[:6]!=b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    start,=struct.unpack_from("<Q",image,40)
    step,n,name_index=struct.unpack_from("<HHH",image,58)
    sections=[struct.unpack_from("<IIQQQQIIQQ",image,start+step*i) for i in range(n)]
    strings=sections[name_index];names=image[strings[4]:strings[4]+strings[5]]
    found=[]
    for name,kind,flags,address,offset,length,*rest in sections:
        end=names.index(b"\0",name)
        symbol=names[name:end].decode()
        if symbol.startswith(".text.") and length:found.append((symbol[6:],hashlib.sha256(image[offset:offset+length]).hexdigest()))
    if len(found)!=1:raise ValueError(f"expected one executable CUDA section, got {len(found)}")
    return found[0]
