# Hugging Face Space deployment

Current GitHub `main` is intentionally **undeployed**. The existing
[Supplymind Space](https://huggingface.co/spaces/Shaurya-Noodle/Supplymind)
serves an older build; a healthy `/health` response does not verify it against
the current source. Pushes to `main` do not deploy.

To deploy intentionally:

1. Create or renew a Hugging Face token with write access to the Space. Save
   it as the GitHub Actions repository secret `HF_TOKEN`; do not put it in the
   repository or in a chat. The previous secret was rejected with HTTP 401.
2. In GitHub Actions, select **Deploy to HuggingFace Space** and choose
   **Run workflow** on `main`.
3. The workflow validates `HF_TOKEN` with Hugging Face before running the
   locked test suite. It uploads only if both pass, then checks the Space's
   `/health` endpoint. A failed preflight does not change the Space.
4. Review the workflow log for the uploaded Space commit ID and verify the
   Space build finished before describing it as current. The `/health` check
   alone proves availability, not parity with `main`.

The historical v3 deployment notes are in [v3/DEPLOY_HF_SPACE.md](v3/DEPLOY_HF_SPACE.md).
