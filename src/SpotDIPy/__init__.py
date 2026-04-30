import sys

try:
    import pkg_resources
except ImportError:
    try:
        import importlib.metadata
        import importlib.resources
        from types import ModuleType
        from pathlib import Path

        pkg_mock = ModuleType("pkg_resources")


        def resource_filename(package_or_requirement, resource_name):
            try:
                # Modern yol bulma yöntemi
                source = importlib.resources.files(package_or_requirement).joinpath(resource_name)
                return str(source)
            except Exception:
                return str(Path(resource_name).absolute())


        pkg_mock.resource_filename = resource_filename
        pkg_mock.get_distribution = importlib.metadata.distribution
        pkg_mock.version = importlib.metadata.version

        sys.modules["pkg_resources"] = pkg_mock
    except Exception:
        pass

from .SpotDIPy import SpotDIPy
from .utils import *
from .plot_GUI import PlotGUI
from .version import __version__
