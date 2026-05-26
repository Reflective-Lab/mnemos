//! Agentic memory consultation policy.
//!
//! The policy answers whether a host must consult prior episodes before
//! promoting load-bearing reasoning. Mnemos owns this vocabulary because it
//! owns agentic memory and recall; host runtimes decide how to enforce it.

use serde::{Deserialize, Serialize};

/// How strongly a run should depend on relevant prior episodes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorConsultationPolicy {
    /// Consult priors whenever available; continue without them when the
    /// episodic store is unavailable.
    #[default]
    RequireWhenAvailable,
    /// Block promotion of load-bearing claims if relevant priors cannot be
    /// consulted.
    RequireForLoadBearingClaims,
    /// Surface priors when recalled, but never block if recall is unavailable.
    OptionalAdvisory,
    /// Explicitly do not consult prior episodes for this run.
    IgnorePriors,
}

impl PriorConsultationPolicy {
    /// Returns whether missing prior recall should block load-bearing claims.
    #[must_use]
    pub const fn blocks_without_priors(self) -> bool {
        matches!(self, Self::RequireForLoadBearingClaims)
    }

    /// Returns whether the policy permits prior recall.
    #[must_use]
    pub const fn consults_priors(self) -> bool {
        !matches!(self, Self::IgnorePriors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_requires_priors_when_available() {
        assert_eq!(
            PriorConsultationPolicy::default(),
            PriorConsultationPolicy::RequireWhenAvailable
        );
    }

    #[test]
    fn only_load_bearing_policy_blocks_when_priors_are_missing() {
        assert!(PriorConsultationPolicy::RequireForLoadBearingClaims.blocks_without_priors());
        assert!(!PriorConsultationPolicy::RequireWhenAvailable.blocks_without_priors());
        assert!(!PriorConsultationPolicy::OptionalAdvisory.blocks_without_priors());
        assert!(!PriorConsultationPolicy::IgnorePriors.blocks_without_priors());
    }

    #[test]
    fn ignore_priors_disables_recall() {
        assert!(!PriorConsultationPolicy::IgnorePriors.consults_priors());
        assert!(PriorConsultationPolicy::OptionalAdvisory.consults_priors());
    }

    #[test]
    fn policy_serializes_as_snake_case() {
        let json =
            serde_json::to_string(&PriorConsultationPolicy::RequireForLoadBearingClaims).unwrap();
        assert_eq!(json, "\"require_for_load_bearing_claims\"");

        let back: PriorConsultationPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, PriorConsultationPolicy::RequireForLoadBearingClaims);
    }
}
