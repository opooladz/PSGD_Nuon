# PSGD_Nuon
Not Muon

Use single sided whitening that is dynamic and learned instead of being instantanious like Muon. This means we don't have to do it every iteration -- think of the savings. 

## SIREN Example

![image](https://github.com/user-attachments/assets/28aab08c-57d3-425a-b123-1733bbb9e9c0)
![image](https://github.com/user-attachments/assets/7d7fbf87-53b7-4c2e-b683-de24d9c54f30)


Siren exmple with Nuon beats Muon tuned

Hyper-params for Muon(Keller): reaches loss of 0.000982 PSGD Nuon reaches loss of 0.000976
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
