project(
    'mkl_random',
    ['c', 'cpp', 'cython'],
    version: run_command(
        find_program('python3', 'python'),
        ['-c', 'exec(open("mkl_random/_version.py").read()); print(__version__)'],
        check: true
    ).stdout().strip(),
    meson_version: '>=1.8.3',
    default_options: [
        'cpp_std=c++11',
        'buildtype=release',
    ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# numpy includes
np_dir = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

inc_np = include_directories(np_dir, 'mkl_random/src')

# compiler/linker
cpp = meson.get_compiler('cpp')
cpp_args = [
    '-D_FILE_OFFSET_BITS=64',
    '-D_LARGEFILE_SOURCE=1',
    '-D_LARGEFILE64_SOURCE=1',
    '-DPY_ARRAY_UNIQUE_SYMBOL=mkl_random_ext',
    '-DNDEBUG'
]
link_args = []

if cpp.get_argument_syntax() == 'msvc'
    link_args += ['Advapi32.lib']
else
    cpp_args += ['-Wno-unused-but-set-variable', '-Wno-unused-function']
endif

rpath_link_args = []
if host_machine.system() != 'windows'
    if host_machine.system() == 'darwin'
        origin = '@loader_path'
    else
        origin = '$ORIGIN'
    endif
    rpath = origin / '../..' + ':' + origin / '../../..'
    rpath_link_args = ['-Wl,-rpath,' + rpath]
endif

mkl_dep = dependency('MKL', method: 'cmake',
    modules: ['MKL::MKL'],
    cmake_args: [
        '-DMKL_ARCH=intel64',
        '-DMKL_LINK=dynamic',
        '-DMKL_THREADING=' + get_option('mkl_threading'),
        '-DMKL_INTERFACE=lp64'
    ],
    required: true
)

# Cython extension
py.extension_module(
    'mklrand',
    sources: [
        'mkl_random/mklrand.pyx',
        'mkl_random/src/mkl_distributions.cpp',
        'mkl_random/src/randomkit.cpp'
    ],
    include_directories: inc_np,
    dependencies: [mkl_dep, py_dep],
    cpp_args: cpp_args,
    link_args: link_args + rpath_link_args,
    override_options: ['cython_language=cpp'],
    install: true,
    subdir: 'mkl_random'
)

# Python sources
py.install_sources(
    [
        'mkl_random/__init__.py',
        'mkl_random/__main__.py',
        'mkl_random/_init_helper.py',
        'mkl_random/_patch_numpy.py',
        'mkl_random/_patch_startup.py',
        'mkl_random/_version.py',
        'mkl_random/patch.py',
        'mkl_random/with_patch.py',
    ],
    subdir: 'mkl_random'
)

py.install_sources(
    [
        'mkl_random/interfaces/__init__.py',
        'mkl_random/interfaces/_numpy_random.py',
        'mkl_random/interfaces/numpy_random.py',
    ],
    subdir: 'mkl_random/interfaces'
)

install_subdir(
    'mkl_random/tests',
    install_dir: py.get_install_dir() / 'mkl_random'
)
