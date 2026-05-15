//! Mnemos's `ProvenanceSource` marker.
//!
//! Migrated to the [`converge_pack::ProvenanceSource`] trait. Public
//! surface is unchanged at call sites:
//! `MNEMOS_PROVENANCE.proposed_fact(...)` reads the same.

use converge_pack::{ContextKey, ProvenanceSource};
use tracing::info_span;

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

/// Legacy per-crate suggestor span helper.
///
/// Internal-only transitional shim. The engine emits the canonical
/// `suggestor.execute` span automatically; existing call sites in this
/// crate are migrated as they're touched.
pub(crate) fn suggestor_span(
    name: &str,
    input_key: ContextKey,
    output_key: ContextKey,
    input_count: usize,
) -> tracing::Span {
    info_span!(
        "mnemos.suggestor.execute",
        provenance = MNEMOS_PROVENANCE.as_str(),
        suggestor = name,
        input_key = ?input_key,
        output_key = ?output_key,
        input_count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
