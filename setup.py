"""
Build the int8_gemm CUDA extension.

Usage (development install):
    pip install -e .

Or build-only:
    python setup.py build_ext --inplace

Requires: CUDA toolkit, torch with matching CUDA version.

GPU arch flags are auto-detected from the installed CUDA toolkit.
You can override with:
    TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install -e .
"""

import os
from setuptools import find_packages, setup


def get_arch_flags() -> list:
    """Return nvcc -gencode flags for all GPUs visible at build time.
    Falls back to a reasonable default set covering Volta, Turing, Ampere, Ada.
    Torch is imported lazily so setup.py can be parsed without torch installed.
    """
    # User-specified override
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    if arch_list:
        flags = []
        for arch in arch_list.replace(",", ";").split(";"):
            arch = arch.strip()
            if not arch:
                continue
            sm = arch.replace(".", "")
            flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
        return flags

    # Auto-detect from visible GPUs (requires torch+CUDA at build time)
    try:
        import torch
        n = torch.cuda.device_count()
        archs = set()
        for i in range(n):
            cap = torch.cuda.get_device_capability(i)
            archs.add(f"{cap[0]}{cap[1]}")
        if archs:
            return [
                f"-gencode=arch=compute_{sm},code=sm_{sm}"
                for sm in sorted(archs)
            ]
    except Exception:
        pass

    # Default: Volta (7.0), Turing (7.5), Ampere (8.0 & 8.6), Ada (8.9)
    defaults = ["70", "75", "80", "86", "89"]
    return [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in defaults]


def get_ext_modules():
    """Build ext_modules list lazily so torch is only imported when compiling."""
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: F401
    except ImportError:
        return []

    return [
        CUDAExtension(
            name="int8_gemm",
            sources=[
                "csrc/int8_gemm_bind.cpp",
                "csrc/int8_gemm.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    *get_arch_flags(),
                ],
            },
        )
    ]


def get_cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
        return {"build_ext": BuildExtension}
    except ImportError:
        return {}


setup(
    name="nano-vllm",
    packages=find_packages(include=["nanovllm*"]),
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
)
