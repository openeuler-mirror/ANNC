"""Configure the production TF addon with synthetic SDKs. No binary is loaded."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def project(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("CMake is required")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.21)\n"
        "project(TfSdkTest LANGUAGES CXX)\n"
        "include(GNUInstallDirs)\n"
        "set(CMAKE_CXX_STANDARD 11)\n"
        f'set(PROJECT_SOURCE_DIR "{ROOT}")\n'
        'add_subdirectory("${PROJECT_SOURCE_DIR}/addons/tensorflow" tensorflow)\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sibling.cc" "int sibling;")\n'
        'add_library(sibling STATIC "${CMAKE_CURRENT_BINARY_DIR}/sibling.cc")\n'
    )
    query = tmp_path / "build/.cmake/api/v1/query"
    query.mkdir(parents=True)
    (query / "codemodel-v2").touch()
    return tmp_path


def sdk(path, version="1.15.0", flags=("-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0")):
    major, minor = version.split(".")[:2]
    header = path / "include/tensorflow/core/public/version.h"
    header.parent.mkdir(parents=True)
    header.write_text(
        f"#define TF_MAJOR_VERSION {major}\n#define TF_MINOR_VERSION {minor}\n"
    )
    lib = path / "lib"
    lib.mkdir()
    for name in ("cc", "framework"):
        (lib / f"libtensorflow_{name}.so.{major}").touch()
    (path / "tensorflow.py").write_text(
        f"__version__ = {version!r}\n"
        "class sysconfig:\n"
        f"    get_compile_flags = staticmethod(lambda: {flags!r})\n"
        f"    get_include = staticmethod(lambda: {str(path / 'include')!r})\n"
        f"    get_lib = staticmethod(lambda: {str(lib)!r})\n"
    )
    python = path / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        f"os.environ['PYTHONPATH'] = {str(path)!r}\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
    )
    python.chmod(0o755)
    return python


def configure(project, python, *, error=None, **options):
    build = project / "build"
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(project),
            "-B",
            str(build),
            f"-DANNC_TENSORFLOW_PYTHON_EXECUTABLE={python}",
        ]
        + [f"-DANNC_{key}={value}" for key, value in options.items()],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )
    if error:
        assert result.returncode != 0, result.stdout
        assert error in " ".join(result.stdout.split()), result.stdout
        return
    assert result.returncode == 0, result.stdout
    reply = build / ".cmake/api/v1/reply"
    index = json.loads(max(reply.glob("index-*.json")).read_text())
    model = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text())
    targets = {}
    for item in model["configurations"][0]["targets"]:
        target = json.loads((reply / item["jsonFile"]).read_text())
        if "nameOnDisk" in target:
            targets[Path(target["nameOnDisk"]).stem] = target
    return targets


def check_plugins(targets, include, standard, abi, libraries=(), unused_sdk=None):
    for name in ("libannc_fused_op", "libannc_optimizer"):
        for group in targets[name]["compileGroups"]:
            includes = {entry["path"] for entry in group["includes"]}
            assert str(include) in includes
            if unused_sdk:
                assert str(unused_sdk / "include") not in includes
            assert group["languageStandard"]["standard"] == str(standard)
            definitions = {entry["define"] for entry in group["defines"]}
            assert {
                d for d in definitions if d.startswith("_GLIBCXX_USE_CXX11_ABI=")
            } == {f"_GLIBCXX_USE_CXX11_ABI={abi}"}
        link = [
            entry["fragment"] for entry in targets[name]["link"]["commandFragments"]
        ]
        if unused_sdk:
            assert all(str(unused_sdk / "lib") not in fragment for fragment in link)
        for library in libraries:
            link_name = library.name.split(".so")[0][3:]
            assert str(library) in link or (
                f"-L{library.parent}" in link and f"-l{link_name}" in link
            ), link
    sibling = targets["libsibling"]["compileGroups"][0]
    assert all(
        "_GLIBCXX_USE_CXX11_ABI" not in flag["define"]
        for flag in sibling.get("defines", [])
    )
    assert str(include) not in [entry["path"] for entry in sibling.get("includes", [])]
    assert sibling["languageStandard"]["standard"] == "11"


def test_switch_sdk_in_one_build(project):
    previous = None
    for name, standard, abi in (("first", 14, 0), ("second", 17, 1)):
        path = project / name
        python = sdk(
            path, flags=(f"-std=c++{standard}", f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
        )
        targets = configure(project, python, TENSORFLOW_PRELOAD="OFF")
        check_plugins(
            targets,
            path / "include",
            standard,
            abi,
            sorted((path / "lib").glob("*.so.1")),
            unused_sdk=previous,
        )
        previous = path


@pytest.mark.parametrize("suffix", ["", ".1.15.0"])
def test_explicit_sdk_settings(project, suffix):
    target = sdk(project / "python-sdk")
    other = project / "explicit"
    sdk(other)
    libraries = [
        other / "lib" / f"libtensorflow_{name}.so{suffix}"
        for name in ("framework", "cc")
    ]
    for library in libraries:
        library.touch()
    targets = configure(
        project,
        target,
        TENSORFLOW_PRELOAD="OFF",
        TENSORFLOW_CXX_STANDARD=14,
        TENSORFLOW_CXX11_ABI=0,
        TENSORFLOW_INCLUDE_DIR=other / "include",
        TENSORFLOW_LIBRARIES=";".join(map(str, libraries)),
    )
    check_plugins(
        targets, other / "include", 14, 0, libraries, unused_sdk=target.parent
    )


def test_sdk_without_compile_flags(project):
    path = project / "sdk"
    python = sdk(path, flags=())
    targets = configure(
        project, python, TENSORFLOW_CXX_STANDARD=14, TENSORFLOW_CXX11_ABI=0
    )
    check_plugins(targets, path / "include", 14, 0)


def test_tf220_sdk_selection(project):
    path = project / "sdk"
    python = sdk(
        path, version="2.20.0", flags=("-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1")
    )
    targets = configure(project, python)
    check_plugins(targets, path / "include", 17, 1)


@pytest.mark.parametrize(
    "version, options, error",
    [
        ("2.19.0", {}, "Unsupported TensorFlow version"),
        ("1.15.0", {"ENABLE_ONNXRUNTIME_EP": "ON"}, "with ORT"),
        ("1.15.0", {"TENSORFLOW_CXX_STANDARD": 17}, "conflicts"),
        ("1.15.0", {"TENSORFLOW_CXX11_ABI": 1}, "conflicts"),
        ("1.15.0", {"TENSORFLOW_PRELOAD": "AUTO"}, "ANNC_TENSORFLOW_PRELOAD"),
    ],
)
def test_reject_incompatible_configuration(project, version, options, error):
    configure(project, sdk(project / "sdk", version=version), error=error, **options)


@pytest.mark.parametrize(
    "standard, abi, error",
    [
        (14, "", "ANNC_TENSORFLOW_CXX11_ABI"),
        ("", 0, "ANNC_TENSORFLOW_CXX_STANDARD"),
        (11, 0, "ANNC_TENSORFLOW_CXX_STANDARD"),
        (14, 2, "ANNC_TENSORFLOW_CXX11_ABI"),
    ],
)
def test_missing_or_invalid_sdk_flags(project, standard, abi, error):
    configure(
        project,
        sdk(project / "sdk", flags=()),
        TENSORFLOW_CXX_STANDARD=standard,
        TENSORFLOW_CXX11_ABI=abi,
        error=error,
    )


@pytest.mark.parametrize("version", ["1.14.0", "2.20.0"])
def test_reject_wrong_headers(project, version):
    sdk(project / "headers", version=version)
    configure(
        project,
        sdk(project / "sdk"),
        TENSORFLOW_INCLUDE_DIR=project / "headers/include",
        error="1.15 headers",
    )


@pytest.mark.parametrize(
    "mode, names, error",
    [
        ("ON", ["cc", "framework"], "ANNC_TENSORFLOW_LIBRARIES"),
        ("OFF", ["cc"], "exactly one"),
        ("OFF", ["framework"], "exactly one"),
        ("OFF", ["cc", "framework", "cc"], "exactly one"),
        ("OFF", ["cc", "framework", "framework"], "exactly one"),
        ("OFF", ["cc.so.2", "framework"], "Unexpected"),
        ("OFF", ["cc", "framework.so.2"], "Unexpected"),
        ("OFF", ["cc", "framework", "other"], "Unexpected"),
    ],
)
def test_reject_incompatible_libraries(project, mode, names, error):
    python = sdk(project / "sdk")
    libraries = []
    for name in names:
        filename = "libtensorflow_" + (name if ".so" in name else name + ".so.1")
        library = python.parent / "lib" / filename
        library.touch()
        libraries.append(str(library))
    configure(
        project,
        python,
        TENSORFLOW_PRELOAD=mode,
        TENSORFLOW_LIBRARIES=";".join(libraries),
        error=error,
    )
