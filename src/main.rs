
//! rmbg: standalone RMBG-2.0 (BiRefNet) inference engine.
//!
//! Pure-Rust CPU implementation plus a CUDA backend, both compiled into the
//! same binary; `--device cpu|gpu` picks one at run time. The CLI is designed
//! to be shelled out to from the existing rmbg-service FastAPI app.
mod deform;
mod engine;
mod forward;
mod graph;
mod image;
mod tensor;
mod weights;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
mod cuda_graph;

use std::time::Instant;

use tensor::{resize_bilinear, Tensor};

struct Args {
    weights: String,
    input: String,
    output: String,
    alpha_only: bool,
    dump_dir: Option<String>,
    steps: Vec<String>,
    device: String,
}

fn usage() -> ! {
    eprintln!(
        "usage: rmbg --weights model.safetensors -i in.png -o out.png [--alpha-only] [--dump-dir DIR] [--dump-step NAME]\n\n  --weights is the raw safetensors checkpoint; no index file is needed.\n\n  --device cpu|gpu selects the backend; both are in the default build. The gpu\n  backend additionally needs a working libcuda.so.1 at run time, and the\n  --no-default-features build has no gpu backend at all.\n  --cuda-selftest compares every CUDA op against the CPU implementation."
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut a = Args {
        weights: String::new(),
        input: String::new(),
        output: String::new(),
        alpha_only: false,
        dump_dir: None,
        steps: Vec::new(),
        device: "cpu".to_string(),
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--weights" | "--model" => {
                i += 1;
                a.weights = argv.get(i).cloned().unwrap_or_default();
            }
            "-i" | "--input" => {
                i += 1;
                a.input = argv.get(i).cloned().unwrap_or_default();
            }
            "-o" | "--output" => {
                i += 1;
                a.output = argv.get(i).cloned().unwrap_or_default();
            }
            "--alpha-only" => a.alpha_only = true,
            "--device" => {
                i += 1;
                a.device = argv.get(i).cloned().unwrap_or_default();
            }
            "--cpu" => a.device = "cpu".to_string(),
            "--gpu" | "--cuda" => a.device = "gpu".to_string(),
            "--cuda-selftest" => {
                #[cfg(feature = "cuda")]
                {
                    if let Err(e) = cuda_graph::selftest() {
                        die(&e);
                    }
                    std::process::exit(0);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    die("--cuda-selftest needs a build with the cuda feature (default); this is a --no-default-features CPU-only build");
                }
            }
            "--dump-dir" => {
                i += 1;
                a.dump_dir = argv.get(i).cloned();
            }
            "--dump-step" => {
                i += 1;
                if let Some(s) = argv.get(i) {
                    a.steps.push(s.clone());
                }
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument {}", other);
                usage();
            }
        }
        i += 1;
    }
    if a.weights.is_empty() || a.input.is_empty() || a.output.is_empty() {
        usage();
    }
    a
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn main() {
    let args = parse_args();
    let t0 = Instant::now();
    let t_index = Instant::now();
    let wts = match weights::Weights::open(&args.weights) {
        Ok(w) => w,
        Err(e) => die(&e),
    };
    let index_s = t_index.elapsed().as_secs_f64();
    eprintln!(
        "loaded index: {} tensors, {:.1} MB in {:.2}s",
        wts.tensors.len(),
        wts.total_bytes as f64 / 1e6,
        index_s
    );

    let loaded = match image::load_rgb(&args.input) {
        Ok(v) => v,
        Err(e) => die(&e),
    };
    let (in_h, in_w) = (loaded.h, loaded.w);
    eprintln!("input {}x{}", in_w, in_h);

    // ViTFeatureExtractor preprocessing: bilinear 1024x1024, rescale, normalize.
    let resized = resize_bilinear(&loaded, 1024, 1024, false);
    let mut input = Tensor::new(3, 1024, 1024);
    let hw = 1024 * 1024;
    for c in 0..3 {
        for i in 0..hw {
            input.data[c * hw + i] = (resized.data[c * hw + i] - MEAN[c]) / STD[c];
        }
    }

    let t2 = Instant::now();
    let logits = if args.device == "gpu" {
        #[cfg(feature = "cuda")]
        {
            let net = match cuda_graph::CudaNet::load(&wts) {
                Ok(n) => n,
                Err(e) => die(&e),
            };
            let l = match net.forward(&input, &args.dump_dir, &args.steps) {
                Ok(v) => v,
                Err(e) => die(&e),
            };
            eprintln!("gpu forward in {:.2}s", t2.elapsed().as_secs_f64());
            l
        }
        #[cfg(not(feature = "cuda"))]
        {
            die("--device gpu needs a build with the cuda feature (default); this is a --no-default-features CPU-only build");
        }
    } else {
        let t1 = Instant::now();
        let mut model = match engine::Rmbg::load(&wts) {
            Ok(m) => m,
            Err(e) => die(&e),
        };
        eprintln!("weights bound in {:.2}s", t1.elapsed().as_secs_f64());
        let l = match engine::forward_dump(&mut model, &input, &args.dump_dir, &args.steps) {
            Ok(v) => v,
            Err(e) => die(&e),
        };
        eprintln!("forward in {:.2}s", t2.elapsed().as_secs_f64());
        l
    };

    let mut prob = Tensor::new(1, logits.h, logits.w);
    for i in 0..logits.data.len() {
        prob.data[i] = 1.0 / (1.0 + (-logits.data[i]).exp());
    }
    let mask = resize_bilinear(&prob, in_h, in_w, false);

    if args.alpha_only {
        if let Err(e) = image::save_gray(&args.output, &mask) {
            die(&e);
        }
    } else if let Err(e) = image::save_rgba(&args.output, &loaded, &mask) {
        die(&e);
    }
    eprintln!("wrote {} ({:.2}s total)", args.output, t0.elapsed().as_secs_f64());
}

fn die(msg: &str) -> ! {
    eprintln!("rmbg: {}", msg);
    std::process::exit(1);
}
