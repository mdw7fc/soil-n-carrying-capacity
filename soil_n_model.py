"""
Dynamic Soil Nitrogen Carrying Capacity Model
==============================================

A system dynamics model tracking soil organic matter pool depletion,
nitrogen mineralization, crop yield response, and carrying capacity
under a sustained reduction in synthetic nitrogen supply over a
10-year analytical horizon.

Framework: Three-pool SOM model (active, slow, passive) informed by
Century/RothC logic, with coupled feedback loops for residue return
and soil physical degradation.

Reference:
    Wallenstein, M. D. (2026). Soil feedback dynamics amplify the
    consequences of nitrogen supply disruptions. Nature Food.

Author: Matthew Wallenstein
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict


# ============================================================
# MODEL PARAMETERS
# ============================================================

@dataclass
class SOMPoolParams:
    """Three-pool SOM structure (Century/RothC-informed)."""
    # Pool fractions of total SOC (must sum to 1.0)
    f_active: float = 0.04      # 2-5% of total SOM
    f_slow: float = 0.38        # 25-50% of total SOM
    f_passive: float = 0.58     # 45-73% of total SOM

    # Decay constants (yr^-1) -- reciprocal of turnover time
    k_active: float = 0.33      # ~3 yr turnover
    k_slow: float = 0.03        # ~33 yr turnover (half-life ~23 yr)
    k_passive: float = 0.001    # ~1000 yr turnover

    # C:N ratios by pool
    cn_active: float = 8.0
    cn_slow: float = 12.0
    cn_passive: float = 12.0

    # Fraction of decomposed C transferred to next pool (humification)
    h_active_to_slow: float = 0.40
    h_slow_to_passive: float = 0.03


@dataclass
class CropParams:
    """Crop yield and nitrogen response parameters."""
    yield_max: float = 5.0          # t/ha grain, global average mix
    mitscherlich_c: float = 0.015   # Yield-N response curvature
    yield_min: float = 0.8          # t/ha subsistence floor
    residue_grain_ratio: float = 1.0
    residue_c_fraction: float = 0.42
    residue_cn: float = 60.0        # High C:N for cereal straw
    grain_n_fraction: float = 0.018
    harvest_index: float = 0.45
    n_uptake_efficiency: float = 0.55


@dataclass
class FeedbackParams:
    """Feedback loop strength parameters."""
    residue_feedback: bool = True
    physical_feedback: bool = True
    physical_strength: float = 1.0
    cn_coupling_feedback: bool = True
    cre_base: float = 0.11         # Carbon retention efficiency of residue
    cre_to_active: float = 0.60    # Fraction of CRE going to active pool
    cre_to_slow: float = 0.40      # Fraction of CRE going to slow pool


@dataclass
class RegionParams:
    """Region-specific soil and agricultural parameters."""
    name: str
    soc_initial: float          # Initial SOC stock (t C/ha, 0-30 cm)
    cn_bulk: float = 10.0
    cropland_mha: float = 100.0
    synth_n_current: float = 120.0  # kg N/ha/yr
    pop_supported: float = 500.0    # millions
    texture_class: int = 1
    whc_sensitivity: float = 8.4    # mm per % SOC in top 20 cm
    water_stress_coeff: float = 0.004
    baseline_water_deficit: float = 0.0
    residue_retention: float = 0.85


# ============================================================
# DEFAULT REGIONS
# ============================================================

def get_default_regions() -> Dict[str, RegionParams]:
    """Return eight regions covering global cropland.

    Calibrated against FAO FAOSTAT 2023, ISRIC SoilGrids, and IFA data.
    Total cropland: ~1,230 Mha; Total synthetic N: ~99 Tg/yr.
    """
    return {
        'north_america': RegionParams(
            name='North America',
            soc_initial=50.0, cn_bulk=10.0,
            cropland_mha=170.0, synth_n_current=76.0,
            pop_supported=900.0, texture_class=1,
            water_stress_coeff=0.003, baseline_water_deficit=0.0,
            residue_retention=0.90,
        ),
        'europe': RegionParams(
            name='Europe',
            soc_initial=42.0, cn_bulk=10.5,
            cropland_mha=130.0, synth_n_current=85.0,
            pop_supported=900.0, texture_class=1,
            water_stress_coeff=0.003, baseline_water_deficit=0.0,
            residue_retention=0.90,
        ),
        'east_asia': RegionParams(
            name='East Asia',
            soc_initial=35.0, cn_bulk=10.0,
            cropland_mha=120.0, synth_n_current=250.0,
            pop_supported=1875.0, texture_class=1,
            water_stress_coeff=0.004, baseline_water_deficit=5.0,
            residue_retention=0.75,
        ),
        'south_asia': RegionParams(
            name='South Asia',
            soc_initial=25.0, cn_bulk=9.5,
            cropland_mha=200.0, synth_n_current=110.0,
            pop_supported=1350.0, texture_class=1,
            water_stress_coeff=0.005, baseline_water_deficit=10.0,
            residue_retention=0.50,
        ),
        'southeast_asia': RegionParams(
            name='Southeast Asia',
            soc_initial=32.0, cn_bulk=10.0,
            cropland_mha=90.0, synth_n_current=89.0,
            pop_supported=750.0, texture_class=1,
            water_stress_coeff=0.004, baseline_water_deficit=5.0,
            residue_retention=0.70,
        ),
        'latin_america': RegionParams(
            name='Latin America',
            soc_initial=45.0, cn_bulk=11.0,
            cropland_mha=160.0, synth_n_current=50.0,
            pop_supported=900.0, texture_class=1,
            water_stress_coeff=0.003, baseline_water_deficit=0.0,
            residue_retention=0.80,
        ),
        'sub_saharan_africa': RegionParams(
            name='Sub-Saharan Africa',
            soc_initial=35.0, cn_bulk=11.0,
            cropland_mha=230.0, synth_n_current=7.0,
            pop_supported=600.0, texture_class=0,
            water_stress_coeff=0.005, baseline_water_deficit=15.0,
            residue_retention=0.55,
        ),
        'fsu_central_asia': RegionParams(
            name='Former Soviet Union & Central Asia',
            soc_initial=50.0, cn_bulk=10.0,
            cropland_mha=130.0, synth_n_current=38.0,
            pop_supported=375.0, texture_class=1,
            water_stress_coeff=0.004, baseline_water_deficit=10.0,
            residue_retention=0.85,
        ),
    }


# ============================================================
# CORE MODEL
# ============================================================

class SoilNCarryingCapacityModel:
    """
    System dynamics model of agricultural carrying capacity under
    a permanent partial reduction in synthetic nitrogen supply.

    State variables (per hectare):
        C_active, C_slow, C_passive: Carbon in three SOM pools (t C/ha)

    Key feedbacks:
        1. Residue return: lower yield -> less residue -> SOM loss -> less N
        2. Physical degradation: SOM loss -> water holding capacity loss -> yield penalty
        3. C-N coupling: severe SOC depletion reduces mineralization efficiency
    """

    def __init__(
        self,
        region: RegionParams,
        reduction_fraction: float = 0.20,
        som_params: SOMPoolParams = None,
        crop_params: CropParams = None,
        feedback_params: FeedbackParams = None,
        dt: float = 1.0,
        t_max: float = 10.0,
    ):
        self.region = region
        self.reduction_fraction = reduction_fraction
        self.som = som_params or SOMPoolParams()
        self.crop = crop_params or CropParams()
        self.fb = feedback_params or FeedbackParams()
        self.dt = dt
        self.t_max = t_max

        # Initialize state
        soc = region.soc_initial
        self.C_active = soc * self.som.f_active
        self.C_slow = soc * self.som.f_slow
        self.C_passive = soc * self.som.f_passive
        self.soc_initial = soc

    def _soc_to_percent(self, soc_tha: float) -> float:
        """Convert t C/ha (0-30 cm) to approximate % SOC."""
        return soc_tha / 39.0  # Assumes BD ~1.3, 30 cm depth

    def _n_mineralization(self, C_pool, cn_ratio, k):
        """Annual N mineralized from a single SOM pool (kg N/ha/yr)."""
        return k * C_pool / cn_ratio * 1000.0

    def _synthetic_n(self, t: float) -> float:
        """Synthetic N at (1 - reduction) of baseline, instantaneously at t=0."""
        if t < 0:
            return self.region.synth_n_current
        return self.region.synth_n_current * (1.0 - self.reduction_fraction)

    def _yield_from_n(self, n_available, water_stress_factor=1.0):
        """Crop yield from Mitscherlich response to available N."""
        n_eff = max(0.0, n_available)
        y = self.crop.yield_max * (1.0 - np.exp(-self.crop.mitscherlich_c * n_eff))
        y *= water_stress_factor
        return max(self.crop.yield_min, y)

    def _water_stress(self, soc_current):
        """Water stress factor (0-1) from SOC-driven water holding capacity loss."""
        if not self.fb.physical_feedback:
            return 1.0
        soc_pct = self._soc_to_percent(soc_current)
        soc_pct_init = self._soc_to_percent(self.soc_initial)
        delta_soc_pct = soc_pct_init - soc_pct
        whc_loss_mm = delta_soc_pct * self.region.whc_sensitivity * self.fb.physical_strength
        total_deficit = self.region.baseline_water_deficit + max(0, whc_loss_mm)
        stress = 1.0 - self.region.water_stress_coeff * total_deficit
        return max(0.3, min(1.0, stress))

    def _residue_c_input(self, yield_actual):
        """Carbon input from crop residue (t C/ha/yr)."""
        residue_mass = yield_actual * self.crop.residue_grain_ratio
        residue_mass *= self.region.residue_retention
        return residue_mass * self.crop.residue_c_fraction

    def _cn_coupling_factor(self, soc_current):
        """Modifier to N mineralization efficiency under severe SOC depletion."""
        if not self.fb.cn_coupling_feedback:
            return 1.0
        frac = soc_current / self.soc_initial
        if frac > 0.60:
            return 1.0
        elif frac < 0.30:
            return 0.6
        else:
            return 1.0 - 0.4 * (0.60 - frac) / 0.30

    def run(self) -> pd.DataFrame:
        """Run the simulation and return time series."""
        n_steps = int(self.t_max / self.dt) + 1
        times = np.arange(0, self.t_max + self.dt / 2, self.dt)[:n_steps]

        results = {col: np.zeros(n_steps) for col in [
            'year', 'C_active', 'C_slow', 'C_passive', 'SOC_total',
            'N_mineralized', 'N_synthetic', 'N_available',
            'yield_tha', 'yield_fraction', 'water_stress',
            'carrying_capacity_fraction',
        ]}
        results['year'] = times

        C_a, C_s, C_p = self.C_active, self.C_slow, self.C_passive
        baseline_bnf = 5.0  # kg N/ha/yr free-living fixation

        # Reference yield at full N
        soc_0 = C_a + C_s + C_p
        n_min_0 = (
            self._n_mineralization(C_a, self.som.cn_active, self.som.k_active)
            + self._n_mineralization(C_s, self.som.cn_slow, self.som.k_slow)
            + self._n_mineralization(C_p, self.som.cn_passive, self.som.k_passive)
        )
        n_avail_0 = (n_min_0 + self.region.synth_n_current + baseline_bnf) * self.crop.n_uptake_efficiency
        yield_0 = self._yield_from_n(n_avail_0)

        for i in range(n_steps):
            t = times[i]
            soc = C_a + C_s + C_p

            cn_factor = self._cn_coupling_factor(soc)
            n_mineralized = (
                self._n_mineralization(C_a, self.som.cn_active, self.som.k_active)
                + self._n_mineralization(C_s, self.som.cn_slow, self.som.k_slow)
                + self._n_mineralization(C_p, self.som.cn_passive, self.som.k_passive)
            ) * cn_factor

            n_synth = self._synthetic_n(t)
            n_available = (n_mineralized + n_synth + baseline_bnf) * self.crop.n_uptake_efficiency

            ws = self._water_stress(soc)
            y = self._yield_from_n(n_available, ws)
            res_c = self._residue_c_input(y)
            cc_frac = y / yield_0 if yield_0 > 0 else 0

            results['C_active'][i] = C_a
            results['C_slow'][i] = C_s
            results['C_passive'][i] = C_p
            results['SOC_total'][i] = soc
            results['N_mineralized'][i] = n_mineralized
            results['N_synthetic'][i] = n_synth
            results['N_available'][i] = n_available
            results['yield_tha'][i] = y
            results['yield_fraction'][i] = y / yield_0 if yield_0 > 0 else 0
            results['water_stress'][i] = ws
            results['carrying_capacity_fraction'][i] = cc_frac

            # Euler integration
            if i < n_steps - 1:
                d_active = self.som.k_active * C_a * self.dt
                d_slow = self.som.k_slow * C_s * self.dt
                d_passive = self.som.k_passive * C_p * self.dt

                h_a_to_s = d_active * self.som.h_active_to_slow
                h_s_to_p = d_slow * self.som.h_slow_to_passive

                if self.fb.residue_feedback:
                    c_in_active = res_c * self.fb.cre_base * self.fb.cre_to_active * self.dt
                    c_in_slow = res_c * self.fb.cre_base * self.fb.cre_to_slow * self.dt
                else:
                    fixed_res = self._residue_c_input(yield_0)
                    c_in_active = fixed_res * self.fb.cre_base * self.fb.cre_to_active * self.dt
                    c_in_slow = fixed_res * self.fb.cre_base * self.fb.cre_to_slow * self.dt

                C_a = max(0.0, C_a - d_active + c_in_active)
                C_s = max(0.0, C_s - d_slow + h_a_to_s + c_in_slow)
                C_p = max(0.0, C_p - d_passive + h_s_to_p)

        return pd.DataFrame(results)


# ============================================================
# PRICE-MEDIATED ALLOCATION
# ============================================================

# Regional price elasticities of nitrogen demand and subsidy buffers.
# Sources cited in manuscript references 14-17.
PRICE_PARAMS = {
    'north_america':       {'elasticity': -0.15, 'subsidy_buffer': 0.30},
    'europe':              {'elasticity': -0.20, 'subsidy_buffer': 0.25},
    'east_asia':           {'elasticity': -0.20, 'subsidy_buffer': 0.40},
    'fsu_central_asia':    {'elasticity': -0.10, 'subsidy_buffer': 0.20},
    'south_asia':          {'elasticity': -0.50, 'subsidy_buffer': 0.50},
    'latin_america':       {'elasticity': -0.45, 'subsidy_buffer': 0.10},
    'southeast_asia':      {'elasticity': -0.60, 'subsidy_buffer': 0.05},
    'sub_saharan_africa':  {'elasticity': -0.90, 'subsidy_buffer': 0.00},
}


def compute_price_mediated_reductions(
    regions: Dict[str, RegionParams],
    global_supply_cut: float = 0.20,
) -> tuple:
    """
    Solve for equilibrium price increase and regional demand reductions.

    When global supply falls by `global_supply_cut`, prices rise until
    the sum of regional demand reductions equals the shortfall (market
    clearing). Each region's reduction depends on its price elasticity
    and government subsidy buffer.

    Returns:
        reductions: dict of {region_name: reduction_fraction}
        price_increase: fractional equilibrium price increase
    """
    # Baseline N consumption per region (Tg N/yr)
    region_n = {
        rn: r.cropland_mha * r.synth_n_current / 1000.0
        for rn, r in regions.items()
    }
    total_n = sum(region_n.values())
    target_reduction_tg = global_supply_cut * total_n

    # Market clearing: P = target / sum(Q_i * |e_i| * (1 - s_i))
    denominator = sum(
        region_n[rn] * abs(PRICE_PARAMS[rn]['elasticity']) * (1 - PRICE_PARAMS[rn]['subsidy_buffer'])
        for rn in regions
    )
    price_increase = target_reduction_tg / denominator

    reductions = {}
    for rn in regions:
        e = abs(PRICE_PARAMS[rn]['elasticity'])
        s = PRICE_PARAMS[rn]['subsidy_buffer']
        red = min(e * price_increase * (1 - s), 0.95)
        reductions[rn] = red

    return reductions, price_increase


# ============================================================
# SCENARIO RUNNERS
# ============================================================

def run_scenario(regions, reduction_frac, t_max=10):
    """Run all regions with a uniform permanent N reduction."""
    results = {}
    for rn, region in regions.items():
        model = SoilNCarryingCapacityModel(
            region=region,
            reduction_fraction=reduction_frac,
            t_max=t_max,
        )
        results[rn] = model.run()
    return results


def run_price_mediated(regions, reductions, t_max=10):
    """Run each region with its price-mediated reduction fraction."""
    results = {}
    for rn, region in regions.items():
        model = SoilNCarryingCapacityModel(
            region=region,
            reduction_fraction=reductions[rn],
            t_max=t_max,
        )
        results[rn] = model.run()
    return results


def aggregate_global(results, regions):
    """Compute global population-weighted aggregates from regional results."""
    first_key = list(results.keys())[0]
    years = results[first_key]['year'].values
    total_pop = sum(r.pop_supported for r in regions.values())

    pop_total = np.zeros(len(years))
    yield_weighted = np.zeros(len(years))

    for rn, df in results.items():
        region = regions[rn]
        weight = region.pop_supported / total_pop
        yield_weighted += df['yield_fraction'].values * weight
        pop_total += df['carrying_capacity_fraction'].values * region.pop_supported

    return pd.DataFrame({
        'year': years,
        'pop_total_millions': pop_total,
        'yield_fraction_weighted': yield_weighted,
    })
