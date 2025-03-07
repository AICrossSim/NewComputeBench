from importlib.util import find_spec


def package_is_available(pkg_name: str) -> bool:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = find_spec(pkg_name) is not None
    return package_exists


def all_packages_are_available(pkg_names: tuple[str]) -> bool:
    assert isinstance(pkg_names, tuple)
    return all(package_is_available(pkg_name) for pkg_name in pkg_names)
