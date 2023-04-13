from ._version import __version__ as __version__
from ._types import *
from ._extract import *
from ._customization import *
from ._lowlevel import InspectionWarning as InspectionWarning

from . import _util

_util.fixup_module_metadata("stackscope", globals())

from . import lowlevel as lowlevel
