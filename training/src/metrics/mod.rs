use std::marker::PhantomData;

use burn::Tensor;
use burn::data::dataloader::Progress;
use burn::prelude::Backend;
use burn::tensor::Int;
use burn::train::ClassificationOutput;
use burn::train::logger::MetricLogger;
use burn::train::metric::store::{MetricsUpdate, NumericMetricUpdate, Split};
use burn::train::metric::{
    Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata, Numeric, NumericEntry,
};
use burn::train::renderer::{
    MetricState, MetricsRenderer, MetricsRendererTraining, ProgressType, TrainingProgress,
};

pub mod batchtime;
pub mod throughput;

// This trait should be implemented once per speciefic output type (classification and so on)
pub trait MetricOutput<B: Backend> {
    fn loss(&self) -> Tensor<B, 1>;
    fn output(&self) -> Tensor<B, 2>; // logits / class scores
    fn targets(&self) -> Tensor<B, 1, Int>;
}

impl<B: Backend> MetricOutput<B> for ClassificationOutput<B> {
    fn loss(&self) -> Tensor<B, 1> {
        self.loss.clone()
    }
    fn output(&self) -> Tensor<B, 2> {
        self.output.clone()
    }
    fn targets(&self) -> Tensor<B, 1, Int> {
        self.targets.clone()
    }
}

// The goal of this trait is to wrap metric into a trait without explicit InputType
// to make it possible to use in vec!. Sadly, Metric trait doesn't allow this =((
pub trait ErasedMetric<B: Backend> {
    fn update(
        &mut self,
        output: &dyn MetricOutput<B>,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry, NumericEntry);
    fn clear(&mut self);
    fn definition(&self) -> MetricDefinition;
}

struct WrappedMetric<M, B, F>
where
    M: Metric + Numeric,
    B: Backend,
    F: Fn(&dyn MetricOutput<B>) -> M::Input,
{
    metric: M,
    extractor: F,
    _backend: PhantomData<B>,
}

impl<M, B, F> ErasedMetric<B> for WrappedMetric<M, B, F>
where
    M: Metric + Numeric,
    B: Backend,
    F: Fn(&dyn MetricOutput<B>) -> M::Input,
{
    fn update(
        &mut self,
        output: &dyn MetricOutput<B>,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry, NumericEntry) {
        let input = (self.extractor)(output);
        let serialized = self.metric.update(&input, metadata);
        let entry = MetricEntry::new(MetricId::new(self.metric.name()), serialized);
        let numeric_entry = self.metric.value();
        let running_entry = self.metric.running_value();
        (entry, numeric_entry, running_entry)
    }

    fn clear(&mut self) {
        self.metric.clear();
    }

    fn definition(&self) -> MetricDefinition {
        MetricDefinition::new(MetricId::new(self.metric.name()), &self.metric)
    }
}

pub struct MetricsHandler<B: Backend> {
    metrics: Vec<Box<dyn ErasedMetric<B>>>,
}

impl<B: Backend> MetricsHandler<B> {
    pub fn new() -> Self {
        Self { metrics: vec![] }
    }

    pub fn metric_names(&self) -> Vec<String> {
        self.metrics
            .iter()
            .map(|m| m.definition().name.clone())
            .collect()
    }

    pub fn definitions(&self) -> Vec<MetricDefinition> {
        self.metrics.iter().map(|m| m.definition()).collect()
    }

    /// Builder-style: chain .add() calls for each metric
    pub fn add<M, F>(mut self, metric: M, extractor: F) -> Self
    where
        M: Metric + Numeric + 'static,
        F: Fn(&dyn MetricOutput<B>) -> M::Input + 'static,
    {
        self.metrics.push(Box::new(WrappedMetric {
            metric,
            extractor,
            _backend: PhantomData,
        }));
        self
    }

    pub fn register(&self, renderer: &mut impl MetricsRenderer) {
        for metric in &self.metrics {
            renderer.register_metric(metric.definition());
        }
    }

    pub fn clear(&mut self) {
        for metric in &mut self.metrics {
            metric.clear();
        }
    }

    pub fn render_train(
        &mut self,
        renderer: &mut impl MetricsRendererTraining,
        progress: &Progress,
        global_progress: &Progress,
        iteration: usize,
        epoch: i64,
    ) {
        renderer.render_train(
            TrainingProgress {
                progress: Some(progress.clone()),
                global_progress: global_progress.clone(),
                iteration: Some(iteration),
            },
            vec![
                ProgressType::Detailed {
                    tag: "Iteration".to_string(),
                    progress: progress.clone(),
                },
                ProgressType::Value {
                    tag: "Epoch".to_string(),
                    value: epoch as usize,
                },
            ],
        );
    }

    pub fn render_valid(
        &mut self,
        renderer: &mut impl MetricsRendererTraining,
        progress: &Progress,
        global_progress: &Progress,
        iteration: usize,
        epoch: i64,
    ) {
        renderer.render_valid(
            TrainingProgress {
                progress: Some(progress.clone()),
                global_progress: global_progress.clone(),
                iteration: Some(iteration),
            },
            vec![
                ProgressType::Detailed {
                    tag: "Iteration".to_string(),
                    progress: progress.clone(),
                },
                ProgressType::Value {
                    tag: "Epoch".to_string(),
                    value: epoch as usize,
                },
            ],
        );
    }

    pub fn update_train(
        &mut self,
        output: &dyn MetricOutput<B>,
        metadata: &MetricMetadata,
        renderer: &mut impl MetricsRendererTraining,
        logger: &mut impl MetricLogger,
        epoch: usize,
    ) {
        let (states, updates) = self.compute(output, metadata);
        for state in states {
            renderer.update_train(state);
        }
        logger.log(MetricsUpdate::new(vec![], updates), epoch, &Split::Train);
    }

    pub fn update_valid(
        &mut self,
        output: &dyn MetricOutput<B>,
        metadata: &MetricMetadata,
        renderer: &mut impl MetricsRendererTraining,
        logger: &mut impl MetricLogger,
        epoch: usize,
    ) {
        let (states, updates) = self.compute(output, metadata);
        for state in states {
            renderer.update_valid(state);
        }
        logger.log(MetricsUpdate::new(vec![], updates), epoch, &Split::Valid);
    }

    fn compute(
        &mut self,
        output: &dyn MetricOutput<B>,
        metadata: &MetricMetadata,
    ) -> (Vec<MetricState>, Vec<NumericMetricUpdate>) {
        let mut states = vec![];
        let mut updates = vec![];

        for metric in &mut self.metrics {
            let (entry, numeric_entry, running_entry) = metric.update(output, metadata);

            states.push(MetricState::Numeric(entry.clone(), numeric_entry.clone()));
            updates.push(NumericMetricUpdate::new(
                entry,
                numeric_entry, // current iteration value
                running_entry, // epoch running mean
            ));
        }

        (states, updates)
    }
}
