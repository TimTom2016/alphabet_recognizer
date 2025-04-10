use shared::model::Model;

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};
use shared::data::{load_dataset, FileBatcher, FileItem};
use std::collections::HashMap;

static ARTIFACT_DIR: &str = "/tmp/alphabet_recognizer";

// Custom Dataset structure
pub struct CustomDataset {
    items: Vec<FileItem>,
}

impl CustomDataset {
    pub fn new(items: Vec<FileItem>) -> Self {
        Self { items }
    }
}

impl Dataset<FileItem> for CustomDataset {
    fn get(&self, index: usize) -> Option<FileItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Config)]
pub struct CustomTrainingConfig {
    #[config(default = 20)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 12)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);

    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = CustomTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    // Load dataset
    let (dataset, label_map) = load_dataset("Letters");
    println!("Label mapping:");
    for (feature, label) in &label_map {
        println!("{} -> {}", feature, label);
    }

    // Split dataset into train and test sets (e.g., 80-20 split)
    let total_size = dataset.len();
    let train_size = (total_size as f32 * 0.8) as usize;

    let mut dataset_shuffled = dataset;
    fastrand::seed(config.seed);
    fastrand::shuffle(&mut dataset_shuffled);

    let (train_data, test_data) = dataset_shuffled.split_at(train_size);

    // Create custom datasets
    let train_dataset = CustomDataset::new(train_data.to_vec());
    let test_dataset = CustomDataset::new(test_data.to_vec());

    // Data
    let batcher_train = FileBatcher::<B>::new(device.clone());
    let batcher_valid = FileBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(Model::new(&device), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save config and model
    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    // Save label mapping
    std::fs::write(
        format!("{ARTIFACT_DIR}/label_mapping.json"),
        serde_json::to_string_pretty(&label_map).unwrap(),
    )
    .expect("Failed to save label mapping");
}
