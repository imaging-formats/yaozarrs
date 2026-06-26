from argparse import ArgumentParser
import json
from pathlib import Path

from pydantic import ValidationError
from yaozarrs import validate_ome_json, validate_ome_uri

def main(raw_args=None):
    parser = ArgumentParser(description="A thin wrapper for executing OME-Zarr conformance tests.")

    # any arguments required for yaozarrs validation
    # isn't this also definitely required by the dingus, so that we know which mode
    # the parent CLI is being run in?
    parser.add_argument("mode", choices=["attributes", "zarr"])

    # required argument to be satisfied by conformance CLI
    # will be path to zarr attributes file containing an OME group when `mode=='attributes'`
    # otherwise to a zarr hierarchy root when `mode=='zarr'`
    parser.add_argument("path", type=Path)
    args = parser.parse_args(raw_args)

    # need to check if args.path contains "/valid/" or "/invalid/" because
    # we now need to match the result string
    expects_exception = '/invalid/' in str(args.path)
    result: dict[str, bool | str] = {}
    try:
        if args.mode == "attributes":
            validate_ome_json(args.path.read_text())
        elif args.mode == "zarr":
            validate_ome_uri(args.path)
        result["validity"] = "valid"
    except Exception as e:
        result["validity"] = "invalid"
        if not expects_exception:
            result["message"] = str(e)
    print(json.dumps(result))

if __name__ == "__main__":
    main()

# need a better example dingus -- or more documentation for the jsonschema one