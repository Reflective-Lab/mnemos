//! gRPC service implementation.

#[allow(clippy::all, clippy::pedantic)]
#[allow(missing_docs)]
mod knowledge {
    tonic::include_proto!("mnemos.v1");
}

pub mod client;
pub mod server;

pub use client::KnowledgeClient;
pub use knowledge::*;
pub use server::KnowledgeServiceImpl;
