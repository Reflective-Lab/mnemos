//! mnemos CLI
//!
//! A self-learning knowledge base with vector search capabilities.

use clap::{Parser, Subcommand};
use mnemos::grpc::{AddEntryRequest, KnowledgeClient, SearchRequest};
use std::collections::HashMap;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// mnemos - Self-learning knowledge base CLI
#[derive(Parser)]
#[command(name = "mnemos")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// gRPC server address
    #[arg(short, long, default_value = "http://localhost:50051")]
    server: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a new knowledge entry
    Add {
        /// Title of the entry
        #[arg(short, long)]
        title: String,

        /// Content of the entry
        #[arg(short, long)]
        content: String,

        /// Category for classification
        #[arg(long)]
        category: Option<String>,

        /// Tags (comma-separated)
        #[arg(long)]
        tags: Option<String>,

        /// Source URL or reference
        #[arg(long)]
        source: Option<String>,
    },

    /// Search the knowledge base
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: u32,

        /// Minimum similarity threshold
        #[arg(long, default_value = "0.0")]
        min_similarity: f32,

        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        /// Disable learning-based ranking
        #[arg(long)]
        no_learning: bool,
    },

    /// Get an entry by ID
    Get {
        /// Entry ID (UUID)
        id: String,
    },

    /// Delete an entry
    Delete {
        /// Entry ID (UUID)
        id: String,
    },

    /// Provide feedback on a search result
    Feedback {
        /// Entry ID
        entry_id: String,

        /// Whether the result was helpful
        #[arg(long)]
        helpful: bool,
    },

    /// Get knowledge base statistics
    Stats,

    /// Health check
    Health,

    /// Import entries from a JSON file
    Import {
        /// Path to JSON file
        file: String,
    },

    /// Export all entries to a JSON file
    Export {
        /// Path to output JSON file
        file: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Connect to server
    let mut client = KnowledgeClient::connect(&cli.server).await?;

    match cli.command {
        Commands::Add {
            title,
            content,
            category,
            tags,
            source,
        } => {
            let tags: Vec<String> = tags
                .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            let request = AddEntryRequest {
                title: title.clone(),
                content,
                category,
                tags,
                source,
                metadata: HashMap::new(),
            };

            let id = client.add_entry_full(request).await?;
            println!("Added entry '{}' with ID: {}", title, id);
        }

        Commands::Search {
            query,
            limit,
            min_similarity,
            category,
            no_learning,
        } => {
            let request = SearchRequest {
                query: query.clone(),
                limit,
                min_similarity,
                category,
                tags: Vec::new(),
                use_learning: !no_learning,
                include_related: false,
                diversity: 0.0,
                hybrid: false,
                keyword_weight: 0.3,
            };

            let response = client.search_full(request).await?;

            println!(
                "Search results for '{}' ({} found, {:.2}ms):\n",
                query, response.total_results, response.search_time_ms
            );

            for (i, result) in response.results.iter().enumerate() {
                if let Some(entry) = &result.entry {
                    println!("{}. {} (score: {:.3})", i + 1, entry.title, result.score);
                    println!("   ID: {}", entry.id);
                    println!(
                        "   Similarity: {:.3}, Boost: {:.3}",
                        result.similarity, result.relevance_boost
                    );
                    let preview = if entry.content.len() > 100 {
                        format!("{}...", &entry.content[..100])
                    } else {
                        entry.content.clone()
                    };
                    println!("   {}\n", preview.replace('\n', " "));
                }
            }
        }

        Commands::Get { id } => match client.get_entry(&id).await? {
            Some(entry) => {
                println!("Title: {}", entry.title);
                println!("ID: {}", entry.id);
                println!("Category: {}", entry.category.as_deref().unwrap_or("None"));
                println!("Tags: {}", entry.tags.join(", "));
                println!("Source: {}", entry.source.as_deref().unwrap_or("None"));
                println!("Created: {}", entry.created_at);
                println!("Updated: {}", entry.updated_at);
                println!("Access Count: {}", entry.access_count);
                println!("Learned Relevance: {:.3}", entry.learned_relevance);
                println!("\nContent:\n{}", entry.content);
            }
            None => {
                println!("Entry not found: {}", id);
            }
        },

        Commands::Delete { id } => {
            client.delete_entry(&id).await?;
            println!("Deleted entry: {}", id);
        }

        Commands::Feedback { entry_id, helpful } => {
            client.record_feedback(&entry_id, helpful).await?;
            println!(
                "Recorded {} feedback for entry {}",
                if helpful { "positive" } else { "negative" },
                entry_id
            );
        }

        Commands::Stats => {
            let stats = client.get_stats().await?;
            println!("Knowledge Base Statistics:");
            println!("  Total Entries: {}", stats.total_entries);
            println!("  Unique Categories: {}", stats.unique_categories);
            println!("  Unique Tags: {}", stats.unique_tags);
            println!("  Total Access Count: {}", stats.total_access_count);
            println!("  Embedding Dimensions: {}", stats.dimensions);
            println!("  Learning Enabled: {}", stats.learning_enabled);
        }

        Commands::Health => {
            let health = client.health().await?;
            println!(
                "Status: {}",
                if health.healthy {
                    "Healthy"
                } else {
                    "Unhealthy"
                }
            );
            println!("Version: {}", health.version);
            println!("Uptime: {} seconds", health.uptime_seconds);
        }

        Commands::Import { file } => {
            let data = tokio::fs::read_to_string(&file).await?;
            let entries: Vec<serde_json::Value> = serde_json::from_str(&data)?;

            let requests: Vec<AddEntryRequest> = entries
                .into_iter()
                .filter_map(|v| {
                    Some(AddEntryRequest {
                        title: v.get("title")?.as_str()?.to_string(),
                        content: v.get("content")?.as_str()?.to_string(),
                        category: v.get("category").and_then(|c| c.as_str()).map(String::from),
                        tags: v
                            .get("tags")
                            .and_then(|t| t.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        source: v.get("source").and_then(|s| s.as_str()).map(String::from),
                        metadata: HashMap::new(),
                    })
                })
                .collect();

            let ids = client.add_entries(requests).await?;
            println!("Imported {} entries", ids.len());
        }

        Commands::Export { file } => {
            // Note: This would need a list_all method on the client
            // For now, we'll just show a message
            println!("Export functionality requires the server to implement listing all entries.");
            println!("Output file: {}", file);
        }
    }

    Ok(())
}
