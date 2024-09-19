use burn_tensor::{ops::ConvOptions, ElementConversion, Shape};
use cubecl::{
    tune::{local_tuner, LocalTuner},
    tune_set, AutotuneKey,
};
use serde::{Deserialize, Serialize};

use crate::{
    kernel::{
        conv::{Conv2dDirect, Conv2dIm2col},
        prng::random_uniform,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitAutotuneKey, JitRuntime, JitTuneId,
};

use super::Inputs;

/// Executes autotune on conv2d operations
pub fn conv2d_autotune<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let client = input.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<R>(&input.device),
        &client,
        Box::new(Conv2dOperations::<R, E, I>::new(
            input, weights, bias, options,
        )),
    )
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct Conv2dAutotuneKey {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    #[autotune(anchor)]
    pub height: usize,
    #[autotune(anchor)]
    pub width: usize,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
}

#[tune_set(operations(Conv2dDirect, Conv2dIm2col), create_key = create_key)]
pub fn conv2d_operations<R: JitRuntime, E: FloatElement, I: IntElement>(
    key: JitAutotuneKey,
    input: JitTensor<R, E, 4>,
    weights: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let (input, weights, bias) = test_inputs(key, &input.device);

    vec![
        Box::new(Conv2dDirect::<R, E, I>::new(
            input.clone(),
            weights.clone(),
            bias.clone(),
            options.clone(),
        )),
        Box::new(Conv2dIm2col::<R, E, I>::new(
            input.clone(),
            weights.clone(),
            bias.clone(),
            options.clone(),
        )),
    ]
}

fn create_key<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E, 4>,
    weights: &JitTensor<R, E, 4>,
    bias: &Option<JitTensor<R, E, 1>>,
    options: &ConvOptions<2>,
) -> JitAutotuneKey {
    let [batch_size, in_channels, height, width] = input.shape.dims;
    let [out_channels, _, kernel_h, kernel_w] = weights.shape.dims;
    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();
    JitAutotuneKey::Conv2d(Conv2dAutotuneKey::new(
        [kernel_h, kernel_w],
        stride,
        padding,
        dilation,
        groups,
        in_channels,
        out_channels,
        height,
        width,
        batch_size,
        bias.is_some(),
    ))
}

fn test_inputs<R: JitRuntime, E: FloatElement>(
    key: &JitAutotuneKey,
    device: &R::JitDevice,
) -> Inputs<R, E> {
    let key = match key {
        JitAutotuneKey::Conv2d(key) => key,
        _ => unreachable!(),
    };

    let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
    let input_shape = Shape::new([key.batch_size, key.in_channels, key.height, key.width]);
    let input = random_uniform(input_shape, device, random_bounds.0, random_bounds.1);
    let c_per_grp = key.in_channels / key.groups;
    let [kernel_h, kernel_w] = key.kernel_size;
    let weight_shape = Shape::new([key.out_channels, c_per_grp, kernel_h, kernel_w]);
    let weights = random_uniform(weight_shape, device, random_bounds.0, random_bounds.1);
    let bias_shape = Shape::new([key.out_channels]);
    let bias = key
        .has_bias
        .then(|| random_uniform(bias_shape, device, random_bounds.0, random_bounds.1));
    (input, weights, bias)
}
