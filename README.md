# PSGD_Nuon
Not Muon

Siren exmple with Nuon beats Muon tuned by Keller

Hyper-params by Keller for Muon: reaches loss of 0.000982 PSGD Nuon reaches loss of 0.000979
```python
    # Assuming Muon is defined elsewhere
    optimizer = Muon(
        muon_params,
        lr=0.005,
        momentum=0.9,
        adamw_params=adamw_params,
        adamw_lr=3e-4,
        adamw_betas=(0.90, 0.95),
        adamw_wd=0
    )
```
