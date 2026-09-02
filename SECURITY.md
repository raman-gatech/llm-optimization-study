# Security Policy

## Supported version

The `main` branch is the supported development version.

## Reporting a vulnerability

Do not open a public issue for credentials, arbitrary code execution, dependency
compromise, or private-data exposure. Contact the repository owner privately
through the security-reporting option on the GitHub repository. Include a
minimal reproduction, affected files, impact, and any suggested mitigation.

## Operational boundaries

- Hugging Face and deployment tokens must be supplied through environment
  variables or an approved secret manager.
- This repository does not serve untrusted model prompts as a public API.
- The public deployment is a read-only static results dashboard.
- Model checkpoints and gated weights are not distributed here.
- Nginx serves the dashboard with a restrictive Content Security Policy and
  standard clickjacking/content-type protections.
