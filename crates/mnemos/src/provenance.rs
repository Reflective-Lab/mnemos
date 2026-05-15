//! Mnemos's `ProvenanceSource` marker.
//!
//! Migrated to the [`converge_pack::ProvenanceSource`] trait. Public
//! surface is unchanged at call sites:
//! `MNEMOS_PROVENANCE.proposed_fact(...)` reads the same.
//!
//! The `converge-core` engine emits the uniform `suggestor.execute`
//! tracing span automatically around every `Suggestor::execute`
//! call. Suggestors override `Suggestor::provenance()` to return
//! `MNEMOS_PROVENANCE.as_str()` so the engine's span carries the
//! right origin.

use converge_pack::ProvenanceSource;

/// Marker type identifying mnemos-emitted facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mnemos;

impl ProvenanceSource for Mnemos {
    fn as_str(&self) -> &'static str {
        "mnemos"
    }
}

/// Canonical provenance constant for mnemos.
pub const MNEMOS_PROVENANCE: Mnemos = Mnemos;

#[cfg(test)]
mod tests {
    use super::*;
    use converge_pack::ContextKey;

    #[test]
    fn provenance_string_is_stable() {
        assert_eq!(MNEMOS_PROVENANCE.as_str(), "mnemos");
    }

    #[test]
    fn proposed_fact_uses_canonical_source_string() {
        let fact = MNEMOS_PROVENANCE.proposed_fact(
            ContextKey::Diagnostic,
            "diagnostic",
            converge_pack::TextPayload::new("content"),
        );
        assert_eq!(fact.provenance(), "mnemos");
    }
}
