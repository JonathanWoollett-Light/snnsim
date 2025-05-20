use std::iter::repeat;
use std::time::Duration;

fn main() {
    println!("Hello, world!");

    // Part 1
    // -------------------------------------------------------------------------

    // - `u`: The current membrane potential.
    // - `time_step`: The amount of time to step forward.
    // - `i`: Input current (e.g. incoming spike).
    // - `r`: Resistance (Ω).
    // - `c`: Capacative capacity.
    // fn leaky_integrate_neuron(u: f64, time_step: Duration, i: f64, r: f64, c: f64) -> f64 {
    //     let tau = r * c;
    //     return u + (time_step.as_secs_f64() / tau) * (i * r - u);
    // }

    // let num_steps = 100;
    // let potentials = (0..num_steps).scan(0.9f64, |a,_ |{
    //     let b = *a;
    //     *a = leaky_integrate_neuron(
    //         *a,
    //         Duration::from_secs_f64(1e-3_f64),
    //         0f64,
    //         5e7_f64,
    //         1e-10_f64,
    //     );
    //     Some(b)
    // }).collect::<Vec<_>>();
    // for (i,p) in potentials.iter().enumerate() {
    //     println!("{i}: {p:.8}");
    // }

    // Part 2
    // -------------------------------------------------------------------------

    // struct Lif {
    //     r: f64,
    //     c: f64,
    //     time_step: Duration,
    // }
    // impl Lif {
    //     fn step(&self, i: f64, u: f64) -> (bool, f64) {
    //         let membrane_potential = leaky_integrate_neuron(u, self.time_step, i, self.r, self.c);
    //         // Default `threshold` on `snn.Lapicque` is `1`.
    //         (membrane_potential > 1f64, membrane_potential)
    //     }
    // }

    // let lif1 = Lif{ r: 5f64, c: 1e-3_f64, time_step: Duration::from_secs_f64(1e-3_f64)};
    // let mut mem = 0.9;
    // let cur_in = (0..num_steps).map(|_| 0f64).collect::<Vec<_>>();
    // let mut mem_rec = vec![mem];
    // for step in 0..num_steps {
    //     let (_spk_out, new_mem) = lif1.step(cur_in[step], mem);
    //     mem = new_mem;
    //     mem_rec.push(mem);
    // }
    // for (i, p) in mem_rec.iter().enumerate() {
    //     println!("{i}: {p:.8}");
    // }

    // Part 3
    // -------------------------------------------------------------------------

    // let mut mem = 0f64;
    // let cur_in = (0..10).map(|_| 0f64).chain((0..190).map(|_|0.1f64)).collect::<Vec<_>>();
    // let mut mem_rec = vec![mem];
    // let mut spiked = Vec::new();
    // let num_steps = 200;
    // for step in 0..num_steps {
    //     let (spk_out, new_mem) = lif1.step(cur_in[step], mem);
    //     mem = new_mem;
    //     spiked.push(spk_out);
    //     mem_rec.push(mem);
    // }
    // for (i, p) in mem_rec.iter().enumerate() {
    //     println!("{i}: {p:.8}");
    // }
    // println!("{spiked:?}");
    // println!("The calculated value of input pulse [A] x resistance [Ω] is: {} V",cur_in[11]*lif1.r);
    // println!("The simulated value of steady-state membrane potential is: {} V",mem_rec[200]);

    // Part 4
    // -------------------------------------------------------------------------

    // let num_steps = 200;

    // let mut mem = 0f64;
    // let cur_in = repeat(0f64).take(10)
    //     .chain(repeat(0.1f64).take(20))
    //     .chain(repeat(0f64).take(170))
    //     .collect::<Vec<_>>();
    // let mut mem_rec = vec![mem];
    // for step in 0..num_steps {
    //     let (_spk_out, new_mem) = lif1.step(cur_in[step], mem);
    //     mem = new_mem;
    //     mem_rec.push(mem);
    // }
    // for (i, p) in mem_rec.iter().enumerate() {
    //     println!("{i}: {p:.8}");
    // }

    // let mut mem = 0f64;
    // let cur_in = repeat(0f64).take(10)
    //     .chain(repeat(0.147f64).take(5))
    //     .chain(repeat(0f64).take(185))
    //     .collect::<Vec<_>>();
    // let mut mem_rec = vec![mem];
    // let num_steps = 200;
    // for step in 0..num_steps {
    //     let (_spk_out, new_mem) = lif1.step(cur_in[step], mem);
    //     mem = new_mem;
    //     mem_rec.push(mem);
    // }
    // for (i, p) in mem_rec.iter().enumerate() {
    //     println!("{i}: {p:.8}");
    // }

    // Part 5
    // -------------------------------------------------------------------------

    fn leaky_integrate_neuron(
        mem: f64,
        cur: f64,
        threshold: f64,
        time_step: Duration,
        r: f64,
        c: f64,
    ) -> (f64, bool) {
        let tau = r * c;
        let spk = mem > threshold;
        let new_mem = mem + (time_step.as_secs_f64() / tau) * (cur * r - mem)
            - (spk as u8 as f64 * threshold);
        return (new_mem, spk);
    }

    struct Lif {
        r: f64,
        c: f64,
        time_step: Duration,
        threshold: f64,
    }
    impl Lif {
        fn new(r: f64, c: f64, time_step: Duration) -> Self {
            Self {
                r,
                c,
                time_step,
                threshold: 1f64,
            }
        }
        fn step(&self, mem: f64, cur: f64) -> (f64, bool) {
            leaky_integrate_neuron(mem, cur, self.threshold, self.time_step, self.r, self.c)
        }
    }

    let lif1 = Lif::new(5.1f64, 5e-3_f64, Duration::from_secs_f64(1e-3_f64));

    let mut mem = 0f64;
    let cur_in = repeat(0f64)
        .take(10)
        .chain(repeat(0.2f64).take(190))
        .collect::<Vec<_>>();
    let mut mem_rec = Vec::new();
    let mut spk_rec = Vec::new();
    let num_steps = 200;
    for step in 0..num_steps {
        let (new_mem, spk_out) = lif1.step(mem, cur_in[step]);
        mem = new_mem;
        mem_rec.push(mem);
        spk_rec.push(spk_out);
    }
    for (i, p) in mem_rec.iter().enumerate() {
        println!("{i}: {p:.8}");
    }
    for (i, p) in spk_rec.iter().enumerate() {
        println!("{i}: {p}");
    }
}
