# Deploy the VisionAssist demo on Render (free)

This deploys the **demo only**: a synthetic camera feed, canned OCR and scene text. The real app needs a camera, microphone and speakers on the user's own computer, so it cannot run on a cloud server (see [SETUP_LOCAL.md](SETUP_LOCAL.md)).

> Free-tier limits change. Check <https://render.com/docs/free> before relying on any number below.

## What you get

- A public URL like `https://visionassist-demo.onrender.com` showing the Hindi dashboard.
- Free web services spin down after 15 minutes without traffic and take about a minute to wake. The workspace gets 750 free instance hours per month.

## Before you start

- A GitHub account with this repo (fork it or use your own).
- A free Render account: <https://render.com> (sign in with GitHub).
- Do **not** add `OCR_API_KEY` or `OPENROUTER_API_KEY` to a public demo.

## Option A: Blueprint (fastest)

1. Push this repo to GitHub (it already contains `render.yaml`).
2. In Render: **New +** > **Blueprint**.
3. Connect your GitHub account and pick this repository.
4. Render reads `render.yaml`, shows `visionassist-demo` on the **Free** plan. Click **Apply** / **Deploy Blueprint**.
5. Wait for the build (several minutes the first time). When the status is **Live**, open the URL at the top of the service page.

## Option B: Manual web service

1. **New +** > **Web Service** > connect this repository.
2. Settings:
   - **Language / Runtime:** Docker
   - **Branch:** `main`
   - **Dockerfile path:** `./Dockerfile`
   - **Instance type:** Free
3. **Environment variables:** add `PORT` = `8000` and `LOCAL_ONLY` = `true`.
4. **Health check path:** `/`
5. Click **Create Web Service**.

## Verify

- Logs show `dashboard on http://0.0.0.0:8000`.
- The URL shows the Hindi dashboard with a synthetic video feed and a "demo" notice.
- Pressing the dashboard buttons (read text, describe scene) returns the canned demo results.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Deploy stuck on "port scan" / "no open ports" | Set env var `PORT=8000`, or make sure the Dockerfile `CMD` uses `${PORT:-8000}` and `--host 0.0.0.0` |
| Build fails installing packages | Open the build log; the Dockerfile has not been tested on a real Docker daemon by the author, so please open an issue with the log |
| First visit is very slow | The free service was asleep; wait about a minute |
| Service suspended | Free hours or bandwidth are used up for the month; wait or upgrade |
| Video does not load | Hard-refresh; check the logs for errors |
| Changes do not appear | Render redeploys on each push to `main`; check the **Events** tab |

## Optional: keep it awake

Free services sleep when idle. A free uptime monitor (for example UptimeRobot) pinging the URL every 10 to 14 minutes keeps it awake, but it uses your 750 monthly hours faster, so a single always-on service will use most of them.

## Security notes

- The demo exposes only synthetic frames. Never deploy the real camera mode publicly.
- If you ever expose a real instance, set `WEB_TOKEN` first (see the README Security section).

## Add the link to the README

After it is live, add a badge or link near the top of `README.md`:

```markdown
[**Live demo**](https://YOUR-SERVICE.onrender.com)
```
