mod llama;
mod math;
mod reader;
mod transformer;

use anyhow::Result;
use clap::Parser;
use llama::Config;
use llama::RunOptions;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::mem::size_of;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Temperature
    #[arg(short, long, default_value_t = 0.)]
    temperature: f32,

    /// Path to model weights
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(args.model)?;
    let mut reader = BufReader::new(file);

    // Note(poudels14): can do size_of Config here since all fields are i32
    // and alignment is 4 for each field, so there's no round off
    let mut config = [0; size_of::<Config>()];
    reader.read_exact(&mut config)?;
    let config: Config = bincode::deserialize(&config)?;

    let weights = llama::init_checkpoint_weights(reader, &config)?;
    let mut state = llama::init_run_state(&config);

    llama::run(
        &config,
        &mut state,
        &weights,
        RunOptions {
            temperature: args.temperature,
        },
    );

    Ok(())
}
