use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct FileItem {
    pub image_path: PathBuf,
    pub label: usize,
}

#[derive(Clone, Debug)]
pub struct FileBatcher<B: Backend> {
    device: B::Device,
    threshold: f32,
}

#[derive(Clone, Debug)]
pub struct FileBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> FileBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            threshold: 0.3,
        }
    }
    pub fn with_threshold(device: B::Device, threshold: f32) -> Self {
        Self { device, threshold }
    }
}

impl<B: Backend> Batcher<FileItem, FileBatch<B>> for FileBatcher<B> {
    fn batch(&self, items: Vec<FileItem>) -> FileBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                // Open and resize the image
                let img = image::open(&item.image_path)
                    .expect("Failed to open image")
                    .resize_exact(28, 28, image::imageops::FilterType::Gaussian);

                // Convert to grayscale after resizing
                let img = img.to_luma8();

                // Verify dimensions
                assert_eq!(
                    img.dimensions(),
                    (28, 28),
                    "Image dimensions mismatch after resize"
                );

                // Apply threshold to eliminate noise
                let data: Vec<f32> = img
                    .into_raw()
                    .into_iter()
                    .map(|p| {
                        let normalized = p as f32 / 255.0;
                        // If pixel value is below threshold, consider it background (0)
                        // Otherwise, keep the original value
                        if normalized < self.threshold {
                            0.0
                        } else {
                            normalized
                        }
                    })
                    .collect();

                // Verify data length
                assert_eq!(
                    data.len(),
                    28 * 28,
                    "Unexpected data length after conversion"
                );

                let tensor_data = TensorData::new(data, vec![28, 28]).convert::<B::FloatElem>();

                let tensor =
                    Tensor::<B, 2>::from_data(tensor_data, &self.device).reshape([1, 28, 28]);

                tensor
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        FileBatch { images, targets }
    }
}

pub fn load_dataset(root_dir: &str) -> (Vec<FileItem>, HashMap<String, usize>) {
    let mut items = Vec::new();
    let mut label_map = HashMap::new();
    for (i, c) in ('A'..='Z').enumerate() {
        label_map.insert(c.to_string(), i);
    }
    // First, create a mapping of feature names to label indices
    for entry in fs::read_dir(root_dir).expect("Failed to read directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_dir() {
                if let Some(letter) = path.file_name().and_then(|n| n.to_str()) {
                    if let Some(&label) = label_map.get(letter) {
                        // Read all PNG files in the letter directory
                        for file in fs::read_dir(&path).expect("Failed to read letter directory") {
                            if let Ok(file) = file {
                                let file_path = file.path();
                                if file_path.extension().map_or(false, |ext| ext == "png") {
                                    items.push(FileItem {
                                        image_path: file_path,
                                        label,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (items, label_map)
}
// Helper function to print the label mapping
pub fn print_label_mapping(label_map: &HashMap<String, usize>) {
    println!("Feature to Label Mapping:");
    for (feature, label) in label_map.iter() {
        println!("Feature '{}' -> Label {}", feature, label);
    }
}
