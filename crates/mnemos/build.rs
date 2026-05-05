fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/knowledge.proto")?;
    println!("cargo:rerun-if-changed=proto/knowledge.proto");
    Ok(())
}
