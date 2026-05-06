# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report vulnerabilities through [GitHub Security Advisories](https://github.com/Reflective-Lab/mnemos-knowledge/security/advisories/new) or by emailing **Kenneth Pernyer** at [kenneth@reflective.se](mailto:kenneth@reflective.se).

You should receive a response within 48 hours. If you do not, please follow up via email.

Please include:

- The version of mnemos you're using
- A description of the vulnerability
- Steps to reproduce
- Any relevant logs or error messages
- Your assessment of the impact (CVSS score if possible)

## Built-in Security Practices

- `unsafe_code = "forbid"` across the workspace
- No hard-coded secrets; embedding API keys are sourced from the environment
- gRPC server defaults to localhost binding
- Storage paths are caller-provided — no implicit filesystem writes outside the configured root

## Shared Responsibility

mnemos provides a knowledgebase runtime, not a hardened production deployment. Operators are responsible for:

- Encrypting storage at rest
- Authenticating gRPC clients (TLS, mTLS, or upstream proxy auth)
- Restricting network exposure of the gRPC and CLI surfaces
- Vetting embedding provider data flows for sensitive content
- Backups and retention policy
