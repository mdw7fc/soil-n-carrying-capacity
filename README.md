# Soil Feedback Amplification of Nitrogen Supply Disruptions

Code accompanying:

> Wallenstein, M. D. (2026). Soil feedback dynamics amplify the consequences of nitrogen supply disruptions. *Nature Food*.

## Overview

A system dynamics model coupling three-pool SOM decomposition (Century/RothC kinetics), nitrogen mineralization, Mitscherlich crop yield response, and two feedback loops (residue return, physical degradation) to estimate how soil processes amplify the consequences of a sustained 20% reduction in global synthetic nitrogen supply.

Three allocation scenarios determine how the global shortfall is distributed across eight agricultural regions:

1. **Uniform** — every region loses 20%
2. **Trade-dependency** — reductions proportional to Gulf nitrogen import reliance
3. **Price-mediated** — market clearing allocates shortfall by regional price elasticity of demand and government subsidy buffers

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

- A static estimate of a 20% N supply cut predicts ~3-4% global production loss
- Soil feedbacks amplify this to ~7-8% within a decade (2.0x)
- Price-mediated allocation concentrates damage in the poorest regions: Sub-Saharan Africa faces an 81% effective N reduction despite consuming only 1.6% of global supply

## License

MIT
