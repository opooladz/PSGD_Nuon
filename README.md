# PSGD_Nuon
Not Muon

Use single sided whitening that is dynamic and learned instead of being instantanious like Muon. This means we don't have to do it every iteration -- think of the savings. 

## SIREN Example

![image](https://github.com/user-attachments/assets/89db50c0-d729-46d2-a501-85b774d03f62)
![image](https://github.com/user-attachments/assets/b6ee1f0f-96b4-4e70-9edf-56b0ecfa4186)


Siren exmple with Nuon beats Muon tuned

Hyper-params for Muon(Keller): reaches loss of 0.000982 PSGD Nuon reaches loss of 0.000898
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
