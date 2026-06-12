import os
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools_rust import Binding, RustExtension

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def get_ext_filename(self, ext_name):
        if ext_name == "gfxgraph._native.libhipgraph_bridge":
            return "gfxgraph/_native/libhipgraph_bridge.so"
        return super().get_ext_filename(ext_name)

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir, "-DBUILD_TESTS=OFF", "-DBUILD_BENCHMARKS=OFF"],
            cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "-j"],
            cwd=build_temp
        )
        
        lib_name = "libhipgraph_bridge.so"
        src_lib = os.path.join(build_temp, lib_name)
        if not os.path.exists(src_lib):
            if os.path.exists(os.path.join(build_temp, "lib", lib_name)):
                src_lib = os.path.join(build_temp, "lib", lib_name)
        
        dst_lib = os.path.abspath(self.get_ext_fullpath(ext.name))
        os.makedirs(os.path.dirname(dst_lib), exist_ok=True)
        shutil.copy2(src_lib, dst_lib)

setup(
    rust_extensions=[
        RustExtension(
            "rs_gfxgraph",
            path="rust/rs_gfxgraph/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        ),
        RustExtension(
            "rs_gfxgraph_stats",
            path="rust/rs_gfxgraph_stats/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    ext_modules=[CMakeExtension("gfxgraph._native.libhipgraph_bridge", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    include_package_data=True,
    package_data={"gfxgraph._native": ["*.so"]},
    zip_safe=False,
)
