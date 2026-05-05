## Summary

- Describe the user-visible or operator-visible change.

## Checks

- [ ] `cargo check --workspace` passes
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] Documentation updated if behavior, setup, or proto schema changed

## Layer Discipline

- [ ] No new Backend type leaks into the Suggestor surface
- [ ] No Suggestor imports a vendor adapter directly
- [ ] Capability declarations remain declarative (no concrete adapter types)

## Security

- [ ] No hard-coded secrets
- [ ] No new network egress without configuration
- [ ] No new filesystem writes outside the configured storage root
