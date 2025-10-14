//! Benchmark binary for Rust environment performance testing
//!
//! This binary provides comprehensive benchmarking capabilities for the
//! Rust environment implementations, measuring throughput, latency,
//! and memory usage across different configurations.

use dual_rl_poker_rust::{
    create_kuhn_batch, create_leduc_batch, BenchmarkResult, EnvBatch, EnvConfig, KuhnPokerEnv,
    LeducPokerEnv,
};
use std::time::{Duration, Instant};
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "benchmark_rust_env")]
struct BenchmarkOpts {
    /// Game to benchmark (kuhn_poker, leduc_poker, all)
    #[structopt(short, long, default_value = "all")]
    game: String,

    /// Batch size for parallel processing
    #[structopt(short, long, default_value = "1000")]
    batch_size: usize,

    /// Number of steps to run
    #[structopt(short, long, default_value = "100000")]
    num_steps: usize,

    /// Number of warmup steps
    #[structopt(short, long, default_value = "1000")]
    warmup_steps: usize,

    /// Random seed for reproducibility
    #[structopt(short, long, default_value = "42")]
    seed: u64,

    /// Enable detailed profiling
    #[structopt(short, long)]
    profile: bool,

    /// Output format (table, json, csv)
    #[structopt(short, long, default_value = "table")]
    output_format: String,

    /// Save results to file
    #[structopt(short, long)]
    save_file: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = BenchmarkOpts::from_args();

    println!("Rust Environment Benchmark");
    println!("=========================");
    println!("Game: {}", opts.game);
    println!("Batch Size: {}", opts.batch_size);
    println!("Steps: {}", opts.num_steps);
    println!("Seed: {}", opts.seed);
    println!();

    let mut results = Vec::new();

    match opts.game.as_str() {
        "kuhn_poker" => {
            results.push(benchmark_kuhn_poker(&opts)?);
        }
        "leduc_poker" => {
            results.push(benchmark_leduc_poker(&opts)?);
        }
        "all" => {
            results.push(benchmark_kuhn_poker(&opts)?);
            results.push(benchmark_leduc_poker(&opts)?);
        }
        _ => {
            return Err(format!("Unknown game: {}", opts.game).into());
        }
    }

    // Output results
    match opts.output_format.as_str() {
        "table" => print_table_results(&results),
        "json" => print_json_results(&results)?,
        "csv" => print_csv_results(&results)?,
        _ => {
            return Err(format!("Unknown output format: {}", opts.output_format).into());
        }
    }

    // Save results if requested
    if let Some(file_path) = opts.save_file {
        save_results(&results, &file_path)?;
        println!("\nResults saved to: {}", file_path);
    }

    Ok(())
}

fn benchmark_kuhn_poker(
    opts: &BenchmarkOpts,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("Benchmarking Kuhn Poker...");

    let config = EnvConfig {
        seed: Some(opts.seed),
        batch_size: opts.batch_size,
        ..Default::default()
    };

    let mut batch = create_kuhn_batch(opts.batch_size, config)?;

    // Warmup
    if opts.warmup_steps > 0 {
        println!("  Warming up...");
        run_warmup(&mut batch, opts.warmup_steps)?;
    }

    // Main benchmark
    println!("  Running benchmark...");
    let start_time = Instant::now();

    let mut total_steps = 0;
    let mut total_episodes = 0;
    let mut step_times = Vec::new();

    for step in 0..opts.num_steps {
        let step_start = Instant::now();

        let actions = batch.sample_legal_actions()?;
        let _results = batch.step(&actions)?;

        let step_time = step_start.elapsed();
        step_times.push(step_time);

        total_steps += opts.batch_size;

        // Count completed episodes
        let dones = batch.dones();
        total_episodes += dones.iter().filter(|&&done| done).count();

        // Progress reporting
        if (step + 1) % (opts.num_steps / 10) == 0 {
            let progress = (step + 1) * 100 / opts.num_steps;
            let elapsed = start_time.elapsed();
            let rate = total_steps as f64 / elapsed.as_secs_f64();
            println!("    Progress: {}% ({:.0} steps/sec)", progress, rate);
        }
    }

    let elapsed = start_time.elapsed();

    // Calculate statistics
    let step_times_ms: Vec<f64> = step_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    let avg_step_time = step_times_ms.iter().sum::<f64>() / step_times_ms.len() as f64;
    let min_step_time = step_times_ms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_step_time = step_times_ms.iter().fold(0.0, |a, &b| a.max(b));

    let result = BenchmarkResult {
        game_name: "kuhn_poker".to_string(),
        batch_size: opts.batch_size,
        num_steps: total_steps,
        num_episodes: total_episodes,
        elapsed_seconds: elapsed.as_secs_f64(),
        steps_per_second: total_steps as f64 / elapsed.as_secs_f64(),
        episodes_per_second: total_episodes as f64 / elapsed.as_secs_f64(),
    };

    println!("  Results:");
    println!("    Total Steps: {}", result.num_steps);
    println!("    Total Episodes: {}", result.num_episodes);
    println!("    Elapsed: {:.3}s", result.elapsed_seconds);
    println!("    Steps/sec: {:.0}", result.steps_per_second);
    println!("    Episodes/sec: {:.0}", result.episodes_per_second);
    println!("    Avg Step Time: {:.3}ms", avg_step_time);
    println!("    Min Step Time: {:.3}ms", min_step_time);
    println!("    Max Step Time: {:.3}ms", max_step_time);

    if opts.profile {
        println!("  Memory usage:");
        print_memory_usage();
    }

    println!();
    Ok(result)
}

fn benchmark_leduc_poker(
    opts: &BenchmarkOpts,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("Benchmarking Leduc Poker...");

    let config = EnvConfig {
        seed: Some(opts.seed),
        batch_size: opts.batch_size,
        ..Default::default()
    };

    let mut batch = create_leduc_batch(opts.batch_size, config)?;

    // Warmup
    if opts.warmup_steps > 0 {
        println!("  Warming up...");
        run_warmup(&mut batch, opts.warmup_steps)?;
    }

    // Main benchmark
    println!("  Running benchmark...");
    let start_time = Instant::now();

    let mut total_steps = 0;
    let mut total_episodes = 0;

    for step in 0..opts.num_steps {
        let actions = batch.sample_legal_actions()?;
        let _results = batch.step(&actions)?;

        total_steps += opts.batch_size;

        // Count completed episodes
        let dones = batch.dones();
        total_episodes += dones.iter().filter(|&&done| done).count();

        // Progress reporting
        if (step + 1) % (opts.num_steps / 10) == 0 {
            let progress = (step + 1) * 100 / opts.num_steps;
            let elapsed = start_time.elapsed();
            let rate = total_steps as f64 / elapsed.as_secs_f64();
            println!("    Progress: {}% ({:.0} steps/sec)", progress, rate);
        }
    }

    let elapsed = start_time.elapsed();

    let result = BenchmarkResult {
        game_name: "leduc_poker".to_string(),
        batch_size: opts.batch_size,
        num_steps: total_steps,
        num_episodes: total_episodes,
        elapsed_seconds: elapsed.as_secs_f64(),
        steps_per_second: total_steps as f64 / elapsed.as_secs_f64(),
        episodes_per_second: total_episodes as f64 / elapsed.as_secs_f64(),
    };

    println!("  Results:");
    println!("    Total Steps: {}", result.num_steps);
    println!("    Total Episodes: {}", result.num_episodes);
    println!("    Elapsed: {:.3}s", result.elapsed_seconds);
    println!("    Steps/sec: {:.0}", result.steps_per_second);
    println!("    Episodes/sec: {:.0}", result.episodes_per_second);

    if opts.profile {
        println!("  Memory usage:");
        print_memory_usage();
    }

    println!();
    Ok(result)
}

fn run_warmup<T: dual_rl_poker_rust::Environment>(
    batch: &mut EnvBatch<T>,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let actions = batch.sample_legal_actions()?;
        let _results = batch.step(&actions)?;
    }
    Ok(())
}

fn print_table_results(results: &[BenchmarkResult]) {
    println!("Benchmark Results Summary");
    println!("========================");
    println!();

    println!(
        "{:<12} {:<10} {:<12} {:<12} {:<15} {:<15}",
        "Game", "Batch", "Steps", "Episodes", "Steps/sec", "Episodes/sec"
    );
    println!("{}", "-".repeat(80));

    for result in results {
        println!(
            "{:<12} {:<10} {:<12} {:<12} {:<15.0} {:<15.0}",
            result.game_name,
            result.batch_size,
            result.num_steps,
            result.num_episodes,
            result.steps_per_second,
            result.episodes_per_second
        );
    }
}

fn print_json_results(results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    println!("{}", json);
    Ok(())
}

fn print_csv_results(results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    println!("game,batch_size,num_steps,num_episodes,elapsed_seconds,steps_per_second,episodes_per_second");
    for result in results {
        println!(
            "{},{},{},{},{},{},{}",
            result.game_name,
            result.batch_size,
            result.num_steps,
            result.num_episodes,
            result.elapsed_seconds,
            result.steps_per_second,
            result.episodes_per_second
        );
    }
    Ok(())
}

fn save_results(
    results: &[BenchmarkResult],
    file_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(file_path, json)?;
    Ok(())
}

fn print_memory_usage() {
    #[cfg(unix)]
    {
        use std::fs;

        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") || line.starts_with("VmSize:") {
                    println!("    {}", line);
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        println!("    Memory profiling not available on this platform");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_creation() {
        let opts = BenchmarkOpts {
            game: "kuhn_poker".to_string(),
            batch_size: 100,
            num_steps: 1000,
            warmup_steps: 100,
            seed: 42,
            profile: false,
            output_format: "table".to_string(),
            save_file: None,
        };

        // This would run the actual benchmark in the real implementation
        // For testing, we just verify the options are valid
        assert_eq!(opts.game, "kuhn_poker");
        assert_eq!(opts.batch_size, 100);
        assert_eq!(opts.num_steps, 1000);
    }
}
