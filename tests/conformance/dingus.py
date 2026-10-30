"""`oztest` dingus: report whether an attributes file or zarr hierarchy is valid."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from pydantic import ValidationError

import yaozarrs
from yaozarrs._storage import StorageValidationError

path = Path(sys.argv[-1])
strict = "--strict" in sys.argv[1:-1]
warnings.simplefilter("ignore")
try:
    if path.is_dir():
        yaozarrs.validate_zarr_store(path, strict=strict)
    else:
        yaozarrs.validate_ome_object(json.loads(path.read_text()), strict=strict)
# anything else is a crash: oztest reports the nonzero exit as an `error`
except (ValidationError, StorageValidationError) as e:
    print(json.dumps({"validity": "invalid", "message": str(e)}))
else:
    print(json.dumps({"validity": "valid"}))
