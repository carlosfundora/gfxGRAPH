<!-- Publisher: Carlos Fundora · GitHub: @carlosfundora · Hugging Face: @carlosfundora · MIT -->
# Publishing gfxGRAPH to PyPI (first-time friendly)

Two ways. **Path A (Trusted Publishing) is recommended** — no token to create, store, or leak.

---

## Path A — Trusted Publishing (no token, set up once)

PyPI lets GitHub Actions publish directly via a cryptographic handshake (OIDC). The workflow is
already in this repo (`.github/workflows/publish.yml`); you just register the publisher on PyPI once.

1. **Make a PyPI account:** https://pypi.org/account/register/ → verify email.
2. **Turn on 2FA** (PyPI requires it): Account settings → "Add 2FA with an authenticator app".
   This is mandatory before you can publish at all.
3. **Add the Trusted Publisher** (this is the whole trick — it pre-authorizes our GitHub workflow):
   go to https://pypi.org/manage/account/publishing/ → "Add a new pending publisher" and enter
   **exactly**:
   | Field | Value |
   |---|---|
   | PyPI Project Name | `gfxgraph` |
   | Owner | `carlosfundora` |
   | Repository name | `gfxGRAPH` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |
4. **Publish:** in the GitHub repo → Releases → "Draft a new release" → tag `v0.5.0`, title
   `gfxGRAPH 0.5.0`, Publish. The Actions workflow builds + uploads automatically. Done →
   `uv pip install gfxgraph`.

(After the first release the "pending" publisher becomes a normal one; future releases just need a
new tag.)

---

## Path B — Manual API token (publish right now)

1. Register + enable 2FA (steps 1–2 above).
2. **Create a token:** Account settings → API tokens → "Add API token". For the *first* upload of a
   brand-new project, scope = **"Entire account"** (the project doesn't exist yet to scope to). Copy
   the token — it starts with `pypi-` and is shown **once**.
3. **Upload** (artifacts are already built in `dist/`):
   ```bash
   cd ~/ai/projects/gfxGRAPH
   uv publish --token <PASTE_TOKEN> dist/gfxgraph-0.5.0*
   ```
   Keep the token secret — run this in your own terminal; don't paste it into chat/commits.
4. **After it works:** create a new **project-scoped** token (`gfxgraph` only) and delete the
   account-scoped one. Or switch to Path A and never touch a token again.

**Practice first (optional):** TestPyPI is a separate sandbox (own account/token):
```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token <TEST_TOKEN> dist/gfxgraph-0.5.0*
```
