// examples/naive_demo.rs
// Demonstration of the naive convolution implementation

use ndarray::Array;
use rust_conv_lib::{naive_conv_2d, naive_conv_4d, NaiveConv};

fn main() {
    println!("=== Naive Convolution Demonstration ===\n");

    // Example 1: Simple 2D convolution
    println!("Example 1: Simple 2D Convolution");
    let input_2d = Array::from_shape_vec((4, 4), vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ]).unwrap();
    
    let kernel_2d = Array::from_shape_vec((2, 2), vec![
        1.0, 0.0,
        0.0, -1.0,
    ]).unwrap();
    
    println!("Input (4x4):");
    for row in input_2d.outer_iter() {
        println!("  {:?}", row.to_vec());
    }
    
    println!("Kernel (2x2):");
    for row in kernel_2d.outer_iter() {
        println!("  {:?}", row.to_vec());
    }
    
    let result_2d = naive_conv_2d(&input_2d, &kernel_2d);
    
    println!("Result (3x3):");
    for row in result_2d.outer_iter() {
        println!("  {:?}", row.to_vec());
    }
    println!();

    // Example 2: Multi-channel 4D convolution
    println!("Example 2: Multi-channel 4D Convolution");
    
    // Create a 4D input: batch=1, channels=2, height=3, width=3
    let input_4d = Array::from_shape_vec((1, 2, 3, 3), vec![
        // Channel 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        // Channel 1
        9.0, 8.0, 7.0,
        6.0, 5.0, 4.0,
        3.0, 2.0, 1.0,
    ]).unwrap();
    
    // Create a 4D kernel: out_channels=1, in_channels=2, height=2, width=2
    let kernel_4d = Array::from_shape_vec((1, 2, 2, 2), vec![
        // For output channel 0, input channel 0
        1.0, 0.0,
        0.0, 1.0,
        // For output channel 0, input channel 1
        0.5, 0.5,
        0.5, 0.5,
    ]).unwrap();
    
    println!("Input shape: {:?}", input_4d.dim());
    println!("Kernel shape: {:?}", kernel_4d.dim());
    
    let result_4d = naive_conv_4d(&input_4d.view(), &kernel_4d.view(), (1, 1), (0, 0));
    
    println!("Result shape: {:?}", result_4d.dim());
    println!("Result values:");
    for batch in 0..result_4d.dim().0 {
        for channel in 0..result_4d.dim().1 {
            println!("  Batch {}, Channel {}:", batch, channel);
            for h in 0..result_4d.dim().2 {
                let mut row = Vec::new();
                for w in 0..result_4d.dim().3 {
                    row.push(result_4d[[batch, channel, h, w]]);
                }
                println!("    {:?}", row);
            }
        }
    }
    println!();

    // Example 3: Using the NaiveConv struct
    println!("Example 3: Using NaiveConv struct");
    let conv = NaiveConv::new();
    
    let simple_input = Array::from_shape_vec((3, 3), vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ]).unwrap();
    
    let edge_kernel = Array::from_shape_vec((3, 3), vec![
        -1.0, -1.0, -1.0,
        -1.0,  8.0, -1.0,
        -1.0, -1.0, -1.0,
    ]).unwrap();
    
    let edge_result = conv.conv_2d(&simple_input, &edge_kernel);
    
    println!("Edge detection kernel on uniform input:");
    println!("Input (3x3): all ones");
    println!("Kernel (3x3): edge detection filter");
    println!("Result: {:?}", edge_result);
    
    // The result should be 0.0 for uniform input with edge detection kernel
    assert_eq!(edge_result[[0, 0]], 0.0);
    println!("✓ Edge detection correctly produces 0 for uniform input");
}
