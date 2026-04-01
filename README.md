# Soil Feedback Amplification of Nitrogen Supply Disruptions

Code accompanying:

> Wallenstein, M. D. (2026). Soil feedback dynamics amplify the consequences of nitrogen supply disruptions. *Nature Food*.

## Overview

A system dynamics model coupling three-pool SOM decomposition (Century/RothC kinetics), nitrogen mineralization with stoichiometric immobilization, Mitscherlich crop yield response, and feedback loops (residue return including root C inputs, physical degradation, C-N coupling) to estimate how soil processes amplify the consequences of a sustained 20% reduction in global synthetic nitrogen supply over a 10-year horizon.

Key model features:
- **Regionalized parameters:** Eight regions with calibrated SOC, CRE, yield response, and root:shoot ratios so that initial SOC is at approximate equilibrium under current management
- **Nitrogen immobilization:** Iterative yield-residue-immobilization feedback accounts for mineral N drawn into SOM when high C:N residue is incorporated
- **Root carbon inputs:** Belowground C enters SOM independent of residue retention

Three allocation scenarios determine how the global shortfall is distributed across eight agricultural regions:

1. **Uniform** -- every region loses 20%
2. **Trade-dependency** -- reductions proportional to Gulf nitrogen import reliance
3. **Price-mediated** -- market clearing allocates shortfall by regional price elasticity of demand and government subsidy buffers

## Usage

```bash
pip install -r requirements.txt
python run_analysis.py
```

Results are printed to stdout and saved as CSV files in `output/`.

## Files

| File | Description |
|---|---|
| `soil_n_model.py` | Core model: SOM pools, feedbacks, regions, price-mediated allocation |
| `run_analysis.py` | Runs all three scenarios, prints summary tables, writes CSV |
| `requirements.txt` | Python dependencies |

## Key findings

- A static estimate of a 20% N supply cut predicts ~4.5-5.9% global production loss (higher than naive estimates because immobilization raises the effective N dependency of current yields)
- Soil feedbacks amplify this to ~5.7-8.1% within a decade (1.3-1.4x amplification)
- Price-mediated allocation concentrates damage in the poorest regions: Sub-Saharan Africa faces an 81% effective N reduction despite consuming only 1.6% of global supply

## License

MIT
