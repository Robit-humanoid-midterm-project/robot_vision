"""User-relative model locations shared by the worker and preview."""
import os
from pathlib import Path


def default_weights():
    explicit = os.environ.get('ROBOT_VISION_WEIGHTS')
    if explicit:
        return os.path.expandvars(os.path.expanduser(explicit))
    source_model = Path(__file__).resolve().parent.parent / 'models/weights.pt'
    if source_model.is_file():
        return str(source_model)
    try:
        from ament_index_python.packages import get_package_share_directory
        installed_model = Path(get_package_share_directory('robot_vision')) / 'models/weights.pt'
        if installed_model.is_file():
            return str(installed_model)
    except (ImportError, LookupError):
        pass
    data_home = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    installed = data_home / 'robot_vision/models/weights.pt'
    return str(installed if installed.is_file() else Path.home() / 'Downloads/weights.pt')
