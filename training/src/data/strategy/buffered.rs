use std::{
    sync::{
        Mutex,
        mpsc::{Receiver, sync_channel},
    },
    thread::{self},
};

use polars::prelude::*;

use crate::data::{mapper::FrameMapper, strategy::FrameBatchStrategy};

pub struct BufferedBatchStrategy {
    recv: Option<Arc<Mutex<Receiver<Option<DataFrame>>>>>,
    mapper: Option<FrameMapper>,
    shuffle: Option<u64>,
    num_workers: usize,
    batch_size: usize,
    buffer_size: usize,
}
impl BufferedBatchStrategy {
    pub fn new(batch_size: usize, buffer_size: usize, num_workers: usize) -> Self {
        Self {
            recv: None,
            mapper: None,
            shuffle: None,
            batch_size,
            buffer_size,
            num_workers,
        }
    }

    pub fn with_mapper(mut self, mapper: FrameMapper) -> Self {
        self.mapper = Some(mapper);
        self
    }

    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }
}

impl Clone for BufferedBatchStrategy {
    fn clone(&self) -> Self {
        Self {
            recv: None,
            mapper: self.mapper.clone(),
            shuffle: self.shuffle,
            batch_size: self.batch_size,
            buffer_size: self.buffer_size,
            num_workers: self.num_workers,
        }
    }
}

impl FrameBatchStrategy for BufferedBatchStrategy {
    fn init(&mut self, stream: CollectBatches) {
        let source = Arc::new(Mutex::new(stream));

        let shuffle = self.shuffle;
        let mapper = self.mapper.clone();

        let (tx, rx) = sync_channel(self.buffer_size);
        self.recv = Some(Arc::new(Mutex::new(rx)));

        for _ in 0..self.num_workers {
            let source = Arc::clone(&source);
            let tx = tx.clone();
            let mapper = mapper.clone();

            thread::spawn(move || {
                loop {
                    let mut stream = source.as_ref().lock().unwrap();
                    let maybe_batch = stream.next().transpose().unwrap();
                    drop(stream);

                    if let Some(mut batch) = maybe_batch {
                        if let Some(ref map) = mapper {
                            batch = map(batch);
                        }

                        if let Some(seed) = shuffle {
                            batch = batch
                                .sample_n_literal(batch.height(), false, true, Some(seed))
                                .unwrap();
                        }

                        if tx.send(Some(batch)).is_err() {
                            break;
                        }
                    } else {
                        tx.send(None).unwrap_or(());
                        break;
                    }
                }
            });
        }

        //thread::spawn(move || {
        //    let mut stream = source.as_ref().unwrap().lock().unwrap();
        //    while let Some(mut batch) = stream.next().transpose().unwrap() {
        //        if let Some(ref map) = mapper {
        //            batch = map(batch);
        //        }

        //        if let Some(seed) = shuffle {
        //            batch = batch
        //                .sample_n_literal(batch.height(), false, true, Some(seed))
        //                .unwrap();
        //        }

        //        tx.send(Some(batch)).unwrap();
        //    }

        //    tx.send(None).unwrap();
        //});
    }

    fn batch(&mut self) -> Option<DataFrame> {
        let comm = self.recv.as_ref().unwrap().lock().unwrap();

        comm.recv().unwrap()
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn chunk_size(&self) -> usize {
        self.batch_size
    }

    fn clone_dyn(&self) -> Box<dyn FrameBatchStrategy> {
        Box::new(self.clone())
    }
}
