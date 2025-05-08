use std::error::Error;
use mnist_nn_rs::{backward_propagation, forward_propagation, get_accuracy, init_params, load_testing_data, load_training_data, update_params, write_to_csv};
use ndarray::Array2;
use polars::prelude::*;

#[derive(Debug)]
struct Config{
    fw1:Array2<f32>,
    fb1:Array2<f32>,
    fw2:Array2<f32>,
    fb2:Array2<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut final_configs = Config{
        fw1: Array2::zeros((10, 784)),
        fb1: Array2::zeros((10, 1)),
        fw2: Array2::zeros((10, 10)),
        fb2: Array2::zeros((10, 1))
    };

    let (mut training_data, mut training_label) = load_training_data()?;
    // let (mut testing_data, mut testing_label) = load_testing_data()?;
    let (mut w1,mut b1, mut w2, mut b2) = init_params();

    let iterations = 501;
    let alpha = 0.1;
    println!("{}", training_label);
    for i in 0..iterations{
        let (mut z1, mut a1, mut z2, mut a2) = forward_propagation(&mut w1, &mut b1, &mut w2, &mut b2, &mut training_data);
        let (dw1, db1, dw2, db2) = backward_propagation(&mut z1, &mut a1, &mut a2, &mut w2, &mut training_data, &mut training_label);
        update_params(&mut w1, &mut b1, &mut w2, &mut b2, &dw1, &db1, &dw2, &db2, alpha);
        if i%50 == 0{
            println!("Iteration: {}", i);
            let acc = get_accuracy(&a2, &training_label);
            println!("Accuracy: {:.2}%", acc * 100.0);
        }
        if i == iterations-1{
            final_configs = Config {
                fw1: w1.clone(),
                fb1: b1.clone(),
                fw2: w2.clone(),
                fb2: b2.clone()
            };
        }
    }

    write_to_csv(&final_configs.fw1, &final_configs.fb1, &final_configs.fw2, &final_configs.fb2);

    Ok(())
}