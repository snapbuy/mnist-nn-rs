use std::error::Error;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fs::File;

pub fn load_training_data() -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let q = LazyCsvReader::new("./mnistdata/mnist_train.csv")
    .with_has_header(true)
    .finish()?;

    // let df = q.clone().with_streaming(true).collect()?;
    let training_labels = q
        .clone()
        .with_streaming(true)
        .select([col("label")])
        .collect()?;

    let training_data = q
        .clone()
        .with_streaming(true)
        .drop([col("label")])
        .collect()?;


    // ----------------------------
    // to_ndarray docs of polars:
    // https://docs.rs/polars/0.47.0/polarsArrayBase<OwnedRepr<f32/prelude/struct.DataFrame.html#method.to_ndarray
    // ----------------------------
    let mut traning_data_ndarray = training_data
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let mut training_labels_ndarray = training_labels
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();

    // taking transpose  
    traning_data_ndarray = traning_data_ndarray.reversed_axes()/ 255.0;
    training_labels_ndarray = training_labels_ndarray.reversed_axes();

    let data_dimensions:&[usize] = traning_data_ndarray.shape();
    let labels_dimensions:&[usize] = training_labels_ndarray.shape();

    // println!("{}", traning_data_ndarray);
    // println!("{}", training_labels_ndarray);
    println!("DATA: {}, {}", data_dimensions[0], data_dimensions[1]);
    println!("LABELS: {}, {}", labels_dimensions[0], labels_dimensions[1]);
    Ok((traning_data_ndarray, training_labels_ndarray))
}

pub fn load_testing_data() -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let q = LazyCsvReader::new("./mnistdata/mnist_test.csv")
    .with_has_header(true)
    .finish()?;

    let testing_labels = q
        .clone()
        .with_streaming(true)
        .select([col("label")])
        .collect()?;

    let testing_data = q
        .clone()
        .with_streaming(true)
        .drop([col("label")])
        .collect()?;


    // ----------------------------
    // to_ndarray docs of polars:
    // https://docs.rs/polars/0.47.0/polarsArrayBase<OwnedRepr<f32/prelude/struct.DataFrame.html#method.to_ndarray
    // ----------------------------
    let mut testing_data_ndarray = testing_data
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();
    let mut testing_labels_ndarray = testing_labels
        .to_ndarray::<Float32Type>(IndexOrder::Fortran)
        .unwrap();

    // taking transpose  
    testing_data_ndarray = testing_data_ndarray.reversed_axes()/ 255.0;
    testing_labels_ndarray = testing_labels_ndarray.reversed_axes();

    let data_dimensions:&[usize] = testing_data_ndarray.shape();
    let labels_dimensions:&[usize] = testing_labels_ndarray.shape();
    
    // println!("{}", testing_data_ndarray);
    // println!("{}", testing_labels_ndarray);
    println!("DATA: {}, {}", data_dimensions[0], data_dimensions[1]);
    println!("LABELS: {}, {}", labels_dimensions[0], labels_dimensions[1]);
    Ok((testing_data_ndarray, testing_labels_ndarray))
}




pub fn init_params()->(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>){
    
    // ----------------------------
    // fn random of ndarray_rand crate:
    // https://docs.rs/ndarray-rand/0.15.0/ndarray_rand/trait.RandomExt.html#tymethod.random
    // ----------------------------

    let w1 = Array2::random((10, 784), Uniform::new(-0.5, 0.5));
    let b1 = Array2::random((10, 1), Uniform::new(-0.5, 0.5));
    let w2 = Array2::random((10, 10), Uniform::new(-0.5, 0.5));
    let b2 = Array2::random((10, 1), Uniform::new(-0.5, 0.5));
    
    (w1,b1,w2,b2)
}




pub fn relu(z:&mut Array2<f32>){
    z.mapv_inplace(|x| x.max(0.0))
}

// pub fn softmax(z:&mut Array2<f32>){
//     z.mapv_inplace(|x| x.exp());
//     let sum = z.sum();
//     z.mapv_inplace(|x| x / sum);
// }

pub fn softmax(z: &mut Array2<f32>) {
    for mut col in z.axis_iter_mut(Axis(1)) {
        // Subtract max for numerical stability
        let max = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        col.mapv_inplace(|x| (x - max).exp());

        let sum = col.sum();
        col.mapv_inplace(|x| x / sum);
    }
}

pub fn forward_propagation(
    w1:&mut Array2<f32>,
    b1:&mut Array2<f32>,
    w2:&mut Array2<f32>,
    b2:&mut Array2<f32>,
    x:&mut  Array2<f32>
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)
{
    let z1 = w1.dot(x) + &*b1;
    let mut a1 = z1.clone();
    relu(&mut a1);
    
    let z2 = w2.dot(&a1) + &*b2;
    let mut a2 = z2.clone();
    softmax(&mut a2);

    (z1,a1,z2,a2)
}




pub fn one_hot_encoded(y:&mut Array2<f32>, num_classes:usize) -> Array2<f32> {
    let ydash= y.flatten();
    let label_dimensions:&[usize] = ydash.shape();
    let mut one_hot_y = Array2::<f32>::zeros((label_dimensions[0], num_classes));

    for (row, &label) in ydash.iter().enumerate(){
        let class_index = label as usize;
        one_hot_y[(row, class_index)] = 1.0;
    }
    one_hot_y.reversed_axes()
}

pub fn deriv_relu(z:&mut Array2<f32>){
    z.mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
}

pub fn backward_propagation(
    z1:&mut Array2<f32>,
    a1:&mut Array2<f32>,
    a2:&mut Array2<f32>,
    w2:&mut Array2<f32>,
    x:&mut Array2<f32>,
    y:&mut Array2<f32>,
)->(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>){
    let m = y.len() as f32;
    let a1t = a1.view().reversed_axes();
    let w2t = w2.view().reversed_axes();
    let xt = x.view().reversed_axes();
    let one_hot_y = one_hot_encoded(y, 10);

    let dz2 = &*a2 - &one_hot_y;
    let dw2 = (1.0/m)*(dz2.dot(&a1t));
    let db2 = dz2.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1.0 / m);

    let mut z1_deriv = z1.clone();
    deriv_relu(&mut z1_deriv);
    let dz1 = w2t.dot(&dz2)*z1_deriv;
    let dw1 = (1.0/m)*(dz1.dot(&xt));
    let db1 = dz1.sum_axis(Axis(1)).insert_axis(Axis(1)) * (1.0 / m);

    (dw1, db1, dw2, db2)
}




pub fn update_params(
    w1: &mut Array2<f32>,
    b1: &mut Array2<f32>,
    w2: &mut Array2<f32>,
    b2: &mut Array2<f32>,
    dw1: &Array2<f32>,
    db1: &Array2<f32>,
    dw2: &Array2<f32>,
    db2: &Array2<f32>,
    alpha: f32,
) {
    *w1 -= &(alpha * dw1);
    *b1 -= &(alpha * db1);
    *w2 -= &(alpha * dw2);
    *b2 -= &(alpha * db2);
}




pub fn get_accuracy(predictions: &Array2<f32>, labels: &Array2<f32>) -> f32 {
    let pred_classes: Array1<usize> = predictions
        .axis_iter(Axis(1))
        .map(|col| {
            col.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect();

    let true_classes: Array1<usize> = labels.iter().map(|x| *x as usize).collect();

    let correct = pred_classes
        .iter()
        .zip(true_classes.iter())
        .filter(|(pred, truth)| pred == truth)
        .count();

    correct as f32 / labels.len() as f32
}






pub fn write_to_csv(w1: &Array2<f32>, b1: &Array2<f32>, w2: &Array2<f32>, b2: &Array2<f32>) {
    // Convert weights (w1, w2) and biases (b1, b2) to DataFrames
    let mut w1_df = Array2ToDataFrame(w1, "w1");
    let mut b1_df = Array2ToDataFrame(b1, "b1");
    let mut w2_df = Array2ToDataFrame(w2, "w2");
    let mut b2_df = Array2ToDataFrame(b2, "b2");

    let mut file = File::create("./final_config/w1.csv").expect("could not create file");
    CsvWriter::new(&mut file)
    .include_header(true)
    .with_separator(b',')
    .finish(&mut w1_df);

    let mut file = File::create("./final_config/b1.csv").expect("could not create file");
    CsvWriter::new(&mut file)
    .include_header(true)
    .with_separator(b',')
    .finish(&mut b1_df);

    let mut file = File::create("./final_config/w2.csv").expect("could not create file");
    CsvWriter::new(&mut file)
    .include_header(true)
    .with_separator(b',')
    .finish(&mut w2_df);

    let mut file = File::create("./final_config/b2.csv").expect("could not create file");
    CsvWriter::new(&mut file)
    .include_header(true)
    .with_separator(b',')
    .finish(&mut b2_df);
}

pub fn Array2ToDataFrame(array: &Array2<f32>, name: &str) -> DataFrame {
    let rows = array.shape()[0];
    let cols = array.shape()[1];

    let mut columns: Vec<Column> = Vec::new();

    for i in 0..cols {
        let col: Vec<f32> = array.column(i).to_vec();
        let col_name = format!("{}{}", name, i);
        columns.push(Column::new(PlSmallStr::from(col_name), col));
    }

    DataFrame::new(columns).expect("Failed to create DataFrame")
}



#[cfg(test)]
mod tests {
    // use super::*;
}
