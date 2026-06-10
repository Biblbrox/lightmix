use fastrand::Rng;

pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Sample from standard normal distribution using Box-Muller transform.
fn sample_normal(rng: &mut Rng) -> f64 {
    loop {
        let u1 = rng.f64();
        let u2 = rng.f64();
        if u1 > 1e-10 {
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

/// Sample from Gamma(α, 1) distribution using Marsaglia-Tsang method.
/// For α >= 1: Marsaglia-Tsang
/// For 0 < α < 1: use relation Gamma(α) = Gamma(α+1) * U^(1/α)
pub fn sample_gamma(rng: &mut Rng, alpha: f64) -> f64 {
    if alpha <= 0.0 {
        return 0.0;
    }
    if alpha < 1.0 {
        let u = rng.f64();
        return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }

    // Marsaglia-Tsang method for α >= 1
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        if x < 0.0 {
            return d * v;
        }
        let u = rng.f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if (u + u).ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample from Beta(α, β) distribution using Gamma ratios.
/// If X ~ Gamma(α, 1) and Y ~ Gamma(β, 1), then X/(X+Y) ~ Beta(α, β).
pub fn sample_beta(rng: &mut Rng, alpha: f64, beta: f64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        return 0.5;
    }
    let x = sample_gamma(rng, alpha);
    let y = sample_gamma(rng, beta);
    if x + y == 0.0 {
        return 0.5;
    }
    x / (x + y)
}
