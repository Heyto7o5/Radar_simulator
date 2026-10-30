#!/usr/bin/env python3
"""Add declared synthetic buildings to a DEM simulation, without running radar."""
import argparse
import hashlib
import json
from pathlib import Path

from radar_platform.dem_buildings import BuildingSpec, add_dem_buildings
from radar_platform.unified.schema import Simulation
from radar_platform.unified.preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--buildings', type=Path, required=True, help='JSON list of BuildingSpec objects')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('existing output preserved')
    source = args.config.read_bytes()
    declarations = args.buildings.read_bytes()
    specs = [BuildingSpec.model_validate(x) for x in json.loads(declarations)]
    result, buildings = add_dem_buildings(Simulation.model_validate_json(source), specs)
    _, estimate = preflight(result)
    report = dict(buildings=buildings, estimate=estimate, simulation_performed=False,
                  source_sha256=hashlib.sha256(source).hexdigest(),
                  declarations_sha256=hashlib.sha256(declarations).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'simulation_config.json').write_text(result.model_dump_json(indent=2)+'\n')
    (args.output/'building_declarations.json').write_bytes(declarations)
    (args.output/'placement_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), buildings=len(buildings), simulation_performed=False)))


if __name__ == '__main__':
    main()
