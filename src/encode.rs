use ndarray::Axis;
use ndarray::{Array2, Array3};

// TODO This is both wrong (when 1st value should be 1 it will always be 0, e.g.
// 1 = [0,1,1,1], 0.5 = [0,0,1,0,1,0]) and slow  (it has a for loop). Speed it
// up and fix it.
/// Rate encodes floating-point spatial data into binary temporal-spatial data.
///
/// Expects `data` to be of shape `[samples x features]` and
/// returns an array of shape `[time steps x samples x features].`
pub fn rate_coding(data: Array2<f32>, time: usize) -> Array3<f32> {
    let steps = &data * time as f32;
    let mut counts = Array2::<f32>::zeros(data.dim());
    let mut base = Array3::zeros([0, data.dim().0, data.dim().1]);
    for _ in 0..time {
        // 0 when below step, 1 when above
        let step_done = (&counts - time as f32 + f32::EPSILON)
            .clamp(0f32, 1f32)
            .ceil();
        base.append(Axis(0), step_done.insert_axis(Axis(0)).view())
            .unwrap();
        counts %= time as f32;
        counts += &steps;
    }

    base
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        // [4,2,2]
        let a = ndarray::array![[1f32, 0f32], [0f32, 1f32], [1f32, 1f32], [0f32, 0f32]];
        // let b = array![1,1,0,0].into_dyn();
        // let c = b.
        println!("{a:?}");
        let b = rate_coding(a, 100);
        // let c = ArrayD::from_shape_fn(vec![1,2], |_|1f32);
        println!("b: {:?}", b);
        assert!(false);
    }
}
