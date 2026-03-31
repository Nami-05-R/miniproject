# Integrated Road Safety Monitoring System

Streamlit-based civic safety dashboard for:

- pothole detection with YOLO
- day/night-aware road inspection
- streetlight analysis using a trained streetlight detector plus illumination logic
- GPS-tagged map visualization
- priority scoring and repair workflow
- citizen road-issue reporting

## Run locally

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

## Public deployment

This repo is prepared for Streamlit Community Cloud.

Main entry file:

```text
app.py
```

Important deploy files:

- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`

Detailed deployment steps:

- see `DEPLOYMENT_GUIDE.md`

## Citizen reporting story

The app includes a `Citizen Portal` tab where public users can:

1. capture or upload a road image
2. provide approximate location details
3. submit the issue into the same authority workflow used by operators

## Model files included for deployment

- `models/pothole_best.pt`
- `models/streetlight_best.pt`
