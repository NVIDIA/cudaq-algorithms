# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import the checkout's public APIs and check maintained skill absence claims.

Requires the library's runtime dependencies. This checks imports and named
exports, not kernel execution, signatures, scientific behavior, or every possible
spelling of a future API. Absence name patterns are explicit maintenance probes;
when an API lands, update the affected records/evals and this contract together.
"""
import argparse
import importlib
import json
from pathlib import Path
import pkgutil
import re
import sys
from types import ModuleType

REPOSITORY = Path(__file__).resolve().parents[3]
COVERAGE = REPOSITORY / "skills-dev/cudaq-algorithms-dev/coverage"
PACKAGE = "cudaq_algorithms"


def check(source_root, registry_path, contract_path):
    report = {
        "passed": False,
        "scope":
        "Real checkout imports and explicit public API/absence probes; no scientific execution",
        "package_origin": None,
        "supported_symbols_checked": 0,
        "absent_claims_checked": 0,
        "modules_imported": [],
        "errors": [],
    }

    def fail(kind, **details):
        report["errors"].append({"check": kind, **details})

    source_root = source_root.resolve()
    package_root = source_root / PACKAGE
    if not (package_root / "__init__.py").is_file():
        fail(
            "source_root",
            message=
            f"No {PACKAGE}/__init__.py under {source_root}; refusing an installed fallback"
        )
        return report
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        features = registry["features"]
        if not isinstance(features, list) or not features:
            raise ValueError(
                "Feature registry must contain documented symbols")
        symbols = {}
        for feature in features:
            if not feature["symbols"]:
                raise ValueError(
                    f"Feature {feature['id']} has no named public symbols")
            for symbol in feature["symbols"]:
                symbols.setdefault(symbol, []).append(feature["record"])
        for symbol in contract.get("additional_symbols", []):
            symbols.setdefault(symbol, [])
        for symbol in symbols:
            if not isinstance(symbol,
                              str) or not symbol.startswith(PACKAGE + "."):
                raise ValueError(f"Not a {PACKAGE} public symbol: {symbol!r}")
        absent = contract["absent"]
        if contract["schema_version"] != 1 or not isinstance(
                absent, list) or not absent:
            raise ValueError(
                "Contract needs schema_version 1 and nonempty absence claims")
        patterns = {}
        for claim in absent:
            if not claim["sources"] or not claim["claim"]:
                raise ValueError(
                    "Each absence claim needs its current record/eval sources")
            if claim["id"] in patterns:
                raise ValueError(f"Duplicate absence claim: {claim['id']}")
            patterns[claim["id"]] = [
                re.compile(pattern)
                for pattern in claim.get("name_patterns", [])
            ]
            if not patterns[claim["id"]] and not claim.get("symbols"):
                raise ValueError(f"Absence claim {claim['id']} has no probes")
    except (OSError, ValueError, KeyError, TypeError, re.error) as exc:
        fail("contract", message=str(exc))
        return report

    # The standalone CLI gets a fresh interpreter; always prefer this checkout.
    sys.path.insert(0, str(source_root))
    modules = {}

    def load_module(name):
        try:
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve()
            if not origin.is_relative_to(package_root):
                raise ValueError(
                    f"Imported {origin}, outside requested source {package_root}"
                )
            modules[name] = module
            report["modules_imported"].append(name)
            return module
        except Exception as exc:
            fail(
                "import_failed",
                module=name,
                message=
                f"{type(exc).__name__}: {exc}; install declared runtime dependencies or repair the checkout import"
            )
            return None

    package = load_module(PACKAGE)
    if package is None:
        return report
    report["package_origin"] = str(Path(package.__file__).resolve())

    def discover(parent):
        if not hasattr(parent, "__path__"):
            return
        for entry in sorted(pkgutil.iter_modules(parent.__path__),
                            key=lambda item: item.name):
            if entry.name.startswith("_"):
                continue
            module = load_module(parent.__name__ + "." + entry.name)
            if module is not None and entry.ispkg:
                discover(module)

    discover(package)

    def resolve(symbol, require_export=True):
        candidates = [
            name for name in modules if symbol.startswith(name + ".")
        ]
        if not candidates:
            raise AttributeError(
                "No successfully imported public module for this symbol")
        prefix = max(candidates, key=len)
        value = modules[prefix]
        for part in symbol[len(prefix) + 1:].split("."):
            if part.startswith("_"):
                raise AttributeError(
                    "Private names are outside this public contract")
            if (require_export and isinstance(value, ModuleType)
                    and hasattr(value, "__all__")
                    and part not in value.__all__):
                raise AttributeError(
                    f"{part} is absent from {value.__name__}.__all__")
            value = getattr(value, part)
        return value

    for symbol, records in sorted(symbols.items()):
        report["supported_symbols_checked"] += 1
        try:
            resolve(symbol)
        except Exception as exc:
            fail(
                "missing_public_symbol",
                symbol=symbol,
                records=records,
                message=
                f"{type(exc).__name__}: {exc}. Update the documented import/registry if this API intentionally changed."
            )

    public_names = {}
    for name, module in modules.items():
        public_names[name] = name.rsplit(".", 1)[-1]
        for attribute in dir(module):
            if not attribute.startswith("_"):
                public_names[name + "." + attribute] = attribute
    for claim in absent:
        report["absent_claims_checked"] += 1
        matches = {
            name
            for name, leaf in public_names.items() if any(
                pattern.fullmatch(leaf) for pattern in patterns[claim["id"]])
        }
        for symbol in claim.get("symbols", []):
            try:
                resolve(symbol, require_export=False)
                matches.add(symbol)
            except AttributeError:
                pass
            except Exception as exc:
                fail("import_failed",
                     module=symbol,
                     message=
                     f"Cannot establish absence: {type(exc).__name__}: {exc}")
        if matches:
            fail(
                "absent_api_appeared",
                claim_id=claim["id"],
                matches=sorted(matches),
                sources=claim["sources"],
                message=
                "Update the skill records/evaluation assertions and API contract for this newly public API; its appearance is a documentation change, not a permanent implementation ban."
            )
    report["passed"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",
                        type=Path,
                        default=REPOSITORY / "python")
    parser.add_argument("--registry",
                        type=Path,
                        default=COVERAGE / "features.json")
    parser.add_argument("--contract",
                        type=Path,
                        default=COVERAGE / "api_contract.json")
    args = parser.parse_args()
    report = check(args.source_root, args.registry, args.contract)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
