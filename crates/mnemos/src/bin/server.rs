//! mnemos gRPC Server

use mnemos::core::{KnowledgeBase, KnowledgeBaseConfig};
use mnemos::grpc::KnowledgeServiceImpl;
use mnemos::grpc::knowledge_service_server::KnowledgeServiceServer;

use clap::Parser;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// mnemos gRPC Server
#[derive(Parser)]
#[command(name = "mnemos-server")]
#[command(author, version, about = "gRPC server for the knowledge base")]
struct Args {
    /// Address to bind to
    #[arg(short, long, default_value = "0.0.0.0:50051")]
    address: String,

    /// Path to knowledge base storage
    #[arg(short, long, default_value = "./knowledge.db")]
    storage: String,

    /// Embedding dimensions
    #[arg(short, long, default_value = "384")]
    dimensions: usize,

    /// Disable self-learning
    #[arg(long)]
    no_learning: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize logging
    let filter = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!(
        "Starting mnemos Server v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Create knowledge base configuration
    let config = KnowledgeBaseConfig {
        dimensions: args.dimensions,
        storage_path: args.storage.clone(),
        learning_enabled: !args.no_learning,
        ..Default::default()
    };

    // Initialize knowledge base
    let kb = KnowledgeBase::with_config(config).await?;
    let kb = Arc::new(RwLock::new(kb));

    // Create gRPC service
    let service = KnowledgeServiceImpl::from_shared(kb);

    // Parse address
    let addr = args.address.parse()?;

    info!("Listening on {}", addr);
    info!("Storage: {}", args.storage);
    info!("Dimensions: {}", args.dimensions);
    info!(
        "Learning: {}",
        if args.no_learning {
            "disabled"
        } else {
            "enabled"
        }
    );

    // Start server
    Server::builder()
        .add_service(KnowledgeServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
