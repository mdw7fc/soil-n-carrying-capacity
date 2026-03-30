"""
Soil Feedback Amplification of Nitrogen Supply Disruptions
==========================================================

Reproduces the core analysis from:
    Wallenstein, M. D. (2026). Soil feedback dynamics amplify the
    consequences of nitrogen supply disruptions. Nature Food.

Simulates a sustained 20% reduction in global synthetic nitrogen supply
over a 10-year horizon under three allocation scenarios:
    1. Uniform: every region loses 20%
    2. Trade-dependency: reductions proportional to Gulf N import reliance
    3. Price-mediated: market clearing allocates shortfall by elasticity

Prints summary tables to stdout and writes CSV results to output/.

Usage:
    python run_analysis.py

Author: Matthew Wallenstein
"""

import numpy as np
from pathlib import Path
from soil_n_model import (
    SoilNCarryingCapacityModel,
    get_default_regions,
    compute_price_mediated_reductions,
    run_scenario,
    run_price_mediated,
    aggregate_global,
    PRICE_PARAMS,
)


# Trade-dependency reductions (Gulf N export reliance, from manuscript).
# Note: weighted global average is ~19%, not exactly 20%, because this
# scenario allocates by physical trade exposure rather than market clearing.
TRADE_DEPENDENCY_REDUCTIONS = {
    'north_america':       0.05,
    'europe':              0.15,
    'east_asia':           0.10,
    'fsu_central_asia':    0.02,
    'south_asia':          0.40,
    'latin_america':       0.20,
    'southeast_asia':      0.30,
    'sub_saharan_africa':  0.10,
}


def main():
    regions = get_default_regions()
    t_max = 10
    outdir = Path(__file__).parent / 'output'
    outdir.mkdir(exist_ok=True)

    # ---- Baseline (no reduction) ----
    baseline = run_scenario(regions, 0.0, t_max)
    baseline_global = aggregate_global(baseline, regions)

    # ---- Scenario 1: Uniform 20% ----
    uniform = run_scenario(regions, 0.20, t_max)
    uniform_global = aggregate_global(uniform, regions)

    # ---- Scenario 2: Trade-dependency ----
    trade_dep = {}
    for rn, region in regions.items():
        model = SoilNCarryingCapacityModel(
            region=region,
            reduction_fraction=TRADE_DEPENDENCY_REDUCTIONS[rn],
            t_max=t_max,
        )
        trade_dep[rn] = model.run()
    trade_dep_global = aggregate_global(trade_dep, regions)

    # ---- Scenario 3: Price-mediated ----
    reductions, price_increase = compute_price_mediated_reductions(
        regions, global_supply_cut=0.20
    )
    price_med = run_price_mediated(regions, reductions, t_max)
    price_med_global = aggregate_global(price_med, regions)

    # ================================================================
    # PRINT RESULTS
    # ================================================================

    print("=" * 70)
    print("SOIL FEEDBACK AMPLIFICATION OF NITROGEN SUPPLY DISRUPTIONS")
    print("=" * 70)

    # Price-mediated allocation summary
    print(f"\nPrice-mediated allocation (equilibrium price increase: {price_increase:.0%})")
    print(f"{'Region':<32s}  {'Elasticity':>10s}  {'Subsidy':>8s}  {'N cut':>7s}")
    print("-" * 65)
    for rn, region in regions.items():
        e = PRICE_PARAMS[rn]['elasticity']
        s = PRICE_PARAMS[rn]['subsidy_buffer']
        red = reductions[rn]
        print(f"  {region.name:<30s}  {e:>10.2f}  {s:>7.0%}  {red:>6.0%}")

    # Global trajectories
    print(f"\n{'':=<70}")
    print("GLOBAL PRODUCTION LOSS TRAJECTORIES")
    print(f"{'':=<70}")
    print(f"{'Year':>5}  {'Uniform':>10}  {'Trade-dep':>10}  {'Price-med':>10}")
    print("-" * 40)
    for yr in [0, 1, 3, 5, 7, 10]:
        bp = baseline_global.loc[baseline_global['year'] == yr, 'pop_total_millions'].values[0]
        u = uniform_global.loc[uniform_global['year'] == yr, 'pop_total_millions'].values[0]
        t = trade_dep_global.loc[trade_dep_global['year'] == yr, 'pop_total_millions'].values[0]
        p = price_med_global.loc[price_med_global['year'] == yr, 'pop_total_millions'].values[0]
        print(f"  {yr:3d}   {(1-u/bp)*100:9.1f}%  {(1-t/bp)*100:9.1f}%  {(1-p/bp)*100:9.1f}%")

    # Amplification factors at year 10
    print(f"\n{'':=<70}")
    print("AMPLIFICATION AT YEAR 10 (dynamic / static)")
    print(f"{'':=<70}")
    for label, scenario_global in [
        ('Uniform 20%', uniform_global),
        ('Trade-dependency', trade_dep_global),
        ('Price-mediated', price_med_global),
    ]:
        bp0 = baseline_global.loc[baseline_global['year'] == 0, 'pop_total_millions'].values[0]
        s0 = scenario_global.loc[scenario_global['year'] == 0, 'pop_total_millions'].values[0]
        bp10 = baseline_global.loc[baseline_global['year'] == 10, 'pop_total_millions'].values[0]
        s10 = scenario_global.loc[scenario_global['year'] == 10, 'pop_total_millions'].values[0]
        static_loss = (1 - s0 / bp0) * 100
        dynamic_loss = (1 - s10 / bp10) * 100
        amp = dynamic_loss / static_loss if static_loss > 0 else float('nan')
        print(f"  {label:<20s}  static={static_loss:.1f}%  dynamic={dynamic_loss:.1f}%  amp={amp:.1f}x")

    # Regional breakdown (price-mediated, year 10)
    print(f"\n{'':=<70}")
    print("REGIONAL BREAKDOWN: PRICE-MEDIATED, YEAR 10")
    print(f"{'':=<70}")
    print(f"{'Region':<32s}  {'N cut':>6s}  {'Loss':>7s}  {'People (M)':>11s}")
    print("-" * 62)
    total_people = 0
    for rn, region in regions.items():
        b10 = baseline[rn].loc[baseline[rn]['year'] == 10, 'carrying_capacity_fraction'].values[0]
        p10 = price_med[rn].loc[price_med[rn]['year'] == 10, 'carrying_capacity_fraction'].values[0]
        loss = (1 - p10 / b10) * 100
        pop_loss = (b10 - p10) * region.pop_supported
        total_people += pop_loss
        print(f"  {region.name:<30s}  {reductions[rn]:>5.0%}  {loss:>6.1f}%  {pop_loss:>10.0f}")
    print(f"  {'TOTAL':<30s}  {'':>5s}  {'':>6s}  {total_people:>10.0f}")

    # ================================================================
    # SAVE CSV OUTPUT
    # ================================================================

    baseline_global.to_csv(outdir / 'baseline_global.csv', index=False)
    uniform_global.to_csv(outdir / 'uniform_20pct_global.csv', index=False)
    trade_dep_global.to_csv(outdir / 'trade_dependency_global.csv', index=False)
    price_med_global.to_csv(outdir / 'price_mediated_global.csv', index=False)

    for label, scenario_results in [
        ('baseline', baseline),
        ('uniform_20pct', uniform),
        ('trade_dependency', trade_dep),
        ('price_mediated', price_med),
    ]:
        for rn, df in scenario_results.items():
            df.to_csv(outdir / f'{label}_{rn}.csv', index=False)

    print(f"\nCSV results saved to {outdir}/")


if __name__ == '__main__':
    main()
