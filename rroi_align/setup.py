from setuptools import setup, Extension, find_namespace_packages
from torch.utils import cpp_extension
from glob import glob

files = (glob('**/*.cpp', recursive=True)
       + glob('**/*.cu', recursive=True)
)

print('Source Files:')
print(files)

'''
cuda_architectures = ['-gencode=arch=compute_70,code=sm_70',
              '-gencode=arch=compute_75,code=compute_75',
              '-gencode=arch=compute_75,code=sm_75']
'''
cuda_architectures = ['-gencode=arch=compute_70,code=sm_70',
                      '-gencode=arch=compute_70,code=compute_70']

common_args = ['-std=c++14']
#common_args = ['-std=c++14', '-g', '-O0']
compile_args = {
  'cxx': common_args + ['-fopenmp', '-DOMP_NESTED=true', '-I/usr/local/cuda/include'],
  'nvcc': common_args + cuda_architectures
}

setup(name='rroi_align',
      ext_modules=[cpp_extension.CppExtension('rroi_align_cpp',
                          files,
                          extra_compile_args=compile_args),
          ],
      cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(parallel=True) },
      )
