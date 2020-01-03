# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from shutil import which

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = which('nvcc')
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        print("nvcc path is " + nvcc)     
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib','x64' )}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    #self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
   #default_compiler_so = self.compiler_so
    super = self.compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
        postfix=os.path.splitext(sources[0])[1]
        
        if postfix == '.cu':
            # use the cuda for .cu files
            #self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']


        return super(sources, output_dir, macros, include_dirs, debug, extra_preargs, postargs, depends)
        # reset the default compiler_so, which we might have changed for cuda
        #self.rc = default_compiler_so

    # inject our redefined _compile method into the class
    self.compile = compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


thisdir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(thisdir, 'utils')
ext_modules = [
    Extension(
        "utils.cython_bbox",
        [os.path.join(source_dir, "bbox.pyx")],
        extra_compile_args={'gcc': []},
        include_dirs=[numpy_include]
    ),

    Extension(
        "utils.nms.cpu_nms",
        [os.path.join(source_dir, "nms", 'cpu_nms.pyx')],
        extra_compile_args={'gcc': []},
        include_dirs=[numpy_include]
    ),
    # Extension('utils.nms.gpu_nms',
    #           [os.path.join(source_dir, "nms", 'gpu_nms.pyx'), os.path.join(source_dir, "nms", 'nms_kernel.cu')],
    #           library_dirs=[CUDA['lib64']],
    #           libraries=['cudart'],
    #           language='c++',
    #           runtime_library_dirs=[CUDA['lib64']],
    #           # this syntax is specific to this build system
    #           # we're only going to use certain compiler args with nvcc and not with gcc
    #           # the implementation of this trick is in customize_compiler() below
    #           extra_compile_args={'gcc': [],
    #                               'nvcc': ['-arch=sm_35',
    #                                        '--ptxas-options=-v',
    #                                        '-c']},
    #           include_dirs=[numpy_include, CUDA['include']]
    # ),
]


if __name__ == '__main__':
    setup(
        name='utils',
        ext_modules=ext_modules,
        # inject our custom trigger
        cmdclass={'build_ext': custom_build_ext},
    )
