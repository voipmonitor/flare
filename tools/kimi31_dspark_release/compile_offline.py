"""Compile the recorded SM120 targets using the provider's device-free compiler worker."""
from pathlib import Path
import multiprocessing
import os
import runpy
import sys

if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("offline workers must start with CUDA devices hidden")
    from b12x._lib.compile_pool import _initialize_worker
    _initialize_worker(0,(12,0),"offline-frank1","NVIDIA RTX PRO 6000 Blackwell Workstation Edition",188,101376,102400,
                       multiprocessing.Array("i",[0,0,0]))
    runpy.run_path(str(Path(__file__).with_name("compile_collectives.py")),run_name="__main__")
