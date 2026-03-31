# Deployment Guide

## Goal

Publish the Streamlit app to a public URL so citizens can upload road photos from anywhere, not only from the same Wi-Fi network.

## Recommended Hosting Options

1. Streamlit Community Cloud
2. Render
3. Railway

## Core Files Added

- `requirements.txt`
- `runtime.txt`
- `Procfile`
- `.streamlit/config.toml`

## Streamlit Community Cloud

1. Push this project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

Notes:
- Make sure the trained model weights are included in the repo or downloaded during startup.
- SQLite will work for prototype usage, but a cloud database is better for multi-user production use.

## Render

1. Push the project to GitHub.
2. Create a new Web Service in Render.
3. Select the repository.
4. Use the start command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

5. Deploy.

## Public Citizen Reporting Story

Once hosted:

1. Citizens open the public app URL.
2. They use the `Citizen Portal` tab.
3. They capture or upload a road image.
4. They provide approximate location details.
5. The app generates a safety record with GPS, risk, map entry, and workflow status.

## Real-World Next Steps

1. Replace SQLite with PostgreSQL or Supabase for concurrent public usage.
2. Add authentication for authority dashboards.
3. Add reverse geocoding for locality names.
4. Add moderator review for public submissions.
5. Add external notification channels like Telegram, email, or SMS.
