use burn_tensor::{
    ops::{conv::calculate_conv_transpose_output_size, ConvTransposeOptions, FloatTensorOps as _},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, ones_device},
        reshape, swap_dims,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

use super::index;

/// Perform a 2D convolution transposition using the GEMM (col2im) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_transpose2d_col2im<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E, 4> {
    let client = input.client.clone();
    let device: <R as JitRuntime>::JitDevice = input.device.clone();
    let [input_channels, im_ch_per_group, kernel_h, kernel_w] = weight.shape.dims;
    let [batch_size, _, input_h, input_w] = input.shape.dims;
    let groups = options.groups;
    let input_ch_per_group = input_channels / groups;
    let ConvTransposeOptions {
        padding: [padding_h, padding_w],
        padding_out: [padding_out_h, padding_out_w],
        dilation: [dilation_h, dilation_w],
        stride: [stride_h, stride_w],
        ..
    } = options.clone();

    let im_h = calculate_conv_transpose_output_size(
        kernel_h,
        stride_h,
        padding_h,
        padding_out_h,
        dilation_h,
        input_h,
    );
    let im_w = calculate_conv_transpose_output_size(
        kernel_w,
        stride_w,
        padding_w,
        padding_out_w,
        dilation_w,
        input_w,
    );
    let im_channels = im_ch_per_group * groups;
    let im_shape = Shape::new([batch_size, im_channels, im_h, im_w]);

    let col_shape_0 = im_ch_per_group * kernel_h * kernel_w;
    let col_shape_1 = batch_size * input_h * input_w;

    let columns = if groups > 1 {
        let col_shape = Shape::new([groups, col_shape_0, col_shape_1]);
        let mut columns = empty_device(client.clone(), device.clone(), col_shape);

        let weight = reshape(
            weight.clone(),
            Shape::new([groups, input_ch_per_group, col_shape_0]),
        );

        let input = swap_dims(input, 0, 1);
        let input_shape = Shape::new([groups, input_ch_per_group, col_shape_1]);
        let input = reshape(input, input_shape);

        for group in 0..groups {
            let weight = swap_dims(index::<R, E, I>(weight.clone(), group), 0, 1);
            let input = index::<R, E, I>(input.clone(), group);
            let values = JitBackend::<R, E, I>::float_matmul(weight, input);
            let values = reshape(values, Shape::new([1, col_shape_0, col_shape_1]));
            columns = JitBackend::<R, E, I>::float_slice_assign(
                columns,
                [group..group + 1, 0..col_shape_0, 0..col_shape_1],
                values,
            );
        }
        reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]))
    } else {
        let weight = reshape(weight.clone(), Shape::new([input_channels, col_shape_0]));
        let weight = swap_dims(weight, 0, 1);
        let input = swap_dims(input, 0, 1);
        let input_shape = Shape::new([input_channels, col_shape_1]);
        let input = reshape(input, input_shape);
        JitBackend::<R, E, I>::float_matmul(weight, input)
    };

    let mut image = col2im(
        columns, im_shape, kernel_h, kernel_w, input_h, input_w, options,
    );

    if let Some(bias) = bias {
        let ones = ones_device(
            client.clone(),
            device.clone(),
            Shape::new([1, batch_size * im_h * im_w]),
        );
        let bias = reshape(bias, Shape::new([im_channels, 1]));
        let bias = JitBackend::<R, E, I>::float_matmul(bias, ones);
        let bias = reshape(bias, Shape::new([im_channels, batch_size, im_h, im_w]));
        let bias = swap_dims(bias, 0, 1);
        image = JitBackend::<R, E, I>::float_add(image, bias);
    };

    image
}

fn col2im<R: JitRuntime, E: FloatElement>(
    columns: JitTensor<R, E, 2>,
    im_shape: Shape<4>,
    kernel_h: usize,
    kernel_w: usize,
    input_h: usize,
    input_w: usize,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E, 4> {
    let [batch_size, out_channels, output_height, output_width] = im_shape.dims;

    let columns = into_contiguous(columns);

    let num_elems = im_shape.num_elements();
    let out = empty_device(
        columns.client.clone(),
        columns.device.clone(),
        im_shape.clone(),
    );

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        col2im_kernel::launch_unchecked::<E::FloatPrimitive, R>(
            &columns.client,
            cube_count,
            cube_dim,
            columns.as_handle_ref().as_tensor_arg(1),
            out.as_handle_ref().as_tensor_arg(1),
            Col2ImArgsLaunch::new(
                ScalarArg::new(batch_size as u32),
                ScalarArg::new(out_channels as u32),
                ScalarArg::new(output_height as u32),
                ScalarArg::new(output_width as u32),
                ScalarArg::new(input_h as u32),
                ScalarArg::new(input_w as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(options.padding[0] as i32),
                ScalarArg::new(options.padding[1] as i32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
            ),
        )
    };

    out
}

#[derive(CubeLaunch)]
struct Col2ImArgs {
    batch_size: UInt,
    im_channels: UInt,
    im_height: UInt,
    im_width: UInt,

    out_h: UInt,
    out_w: UInt,

    kernel_h: UInt,
    kernel_w: UInt,

    pad_h: I32,
    pad_w: I32,
    dilation_h: UInt,
    dilation_w: UInt,
    stride_h: UInt,
    stride_w: UInt,
}

#[cube(launch_unchecked)]
fn col2im_kernel<F: Float>(columns: &Tensor<F>, image: &mut Tensor<F>, args: &Col2ImArgs) {
    if ABSOLUTE_POS > image.len() {
        return;
    }

    let x_im = ABSOLUTE_POS % args.im_width + UInt::cast_from(args.pad_w);
    let y_im = ABSOLUTE_POS / args.im_width % args.im_height + UInt::cast_from(args.pad_h);
    let ch_im = ABSOLUTE_POS / (args.im_width * args.im_height) % args.im_channels;
    let batch = ABSOLUTE_POS / (args.im_width * args.im_height * args.im_channels);

    let kernel_extent_w = (args.kernel_w - 1) * args.dilation_w + 1;
    let kernel_extent_h = (args.kernel_h - 1) * args.dilation_h + 1;

    let mut val = F::new(0.0);

    let mut x_col_start = UInt::new(0);
    if x_im >= kernel_extent_w {
        x_col_start = (x_im - kernel_extent_w) / args.stride_w + 1;
    }
    let x_col_end = UInt::min(x_im / args.stride_w + 1, args.out_w);
    let mut y_col_start = UInt::new(0);
    if y_im >= kernel_extent_h {
        y_col_start = (y_im - kernel_extent_h) / args.stride_h + 1;
    }
    let y_col_end = UInt::min(y_im / args.stride_h + 1, args.out_h);

    for col_y in range(y_col_start, y_col_end, Comptime::new(false)) {
        let k_y = y_im - col_y * args.stride_h;
        for col_x in range(x_col_start, x_col_end, Comptime::new(false)) {
            let k_x = x_im - col_x * args.stride_w;

            if k_y % args.dilation_h == 0 && k_x % args.dilation_w == 0 {
                let kernel_y = k_y / args.dilation_h;
                let kernel_x = k_x / args.dilation_w;

                let col_shape_1 = args.batch_size * args.out_h * args.out_w;
                let col_pos = ch_im * args.kernel_h * args.kernel_w * col_shape_1
                    + kernel_y * args.kernel_w * col_shape_1
                    + kernel_x * col_shape_1
                    + batch * args.out_h * args.out_w
                    + col_y * args.out_w
                    + col_x;
                val += columns[col_pos];
            }
        }
    }
    image[ABSOLUTE_POS] = val;
}