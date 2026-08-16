use crate::{Error, ErrorKind, Result};
use std::future::Future;
use std::pin::Pin;

/// A non-`Send` future suitable for both native and single-threaded WASM use.
pub type IoFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + 'a>>;

/// An asynchronous random-access byte source.
pub trait ByteSource {
    fn len(&self) -> Option<u64>;

    fn is_empty(&self) -> Option<bool> {
        self.len().map(|length| length == 0)
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize>;
}

/// An asynchronous sequential byte sink with optional seek support.
pub trait ByteSink {
    fn position(&self) -> u64;
    fn is_seekable(&self) -> bool {
        true
    }
    fn write<'a>(&'a mut self, bytes: &'a [u8]) -> IoFuture<'a, ()>;
    fn seek<'a>(&'a mut self, position: u64) -> IoFuture<'a, ()>;
    fn flush<'a>(&'a mut self) -> IoFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

/// An owned in-memory source useful for portable callers and deterministic tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemorySource {
    bytes: Vec<u8>,
}

impl MemorySource {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl ByteSource for MemorySource {
    fn len(&self) -> Option<u64> {
        u64::try_from(self.bytes.len()).ok()
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        Box::pin(async move {
            let offset = usize::try_from(offset)
                .map_err(|_| Error::new(ErrorKind::InvalidInput, "read offset is too large"))?;
            if offset > self.bytes.len() {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "read offset is beyond the source",
                ));
            }
            let length = destination.len().min(self.bytes.len() - offset);
            destination[..length].copy_from_slice(&self.bytes[offset..offset + length]);
            Ok(length)
        })
    }
}

/// A seekable in-memory sink useful for portable callers and deterministic tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemorySink {
    bytes: Vec<u8>,
    position: u64,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.bytes
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

impl ByteSink for MemorySink {
    fn position(&self) -> u64 {
        self.position
    }

    fn write<'a>(&'a mut self, bytes: &'a [u8]) -> IoFuture<'a, ()> {
        Box::pin(async move {
            let position = usize::try_from(self.position)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "sink position is too large"))?;
            let end = position
                .checked_add(bytes.len())
                .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "sink allocation overflow"))?;
            if end > self.bytes.len() {
                self.bytes.resize(end, 0);
            }
            self.bytes[position..end].copy_from_slice(bytes);
            self.position = u64::try_from(end).map_err(|_| {
                Error::new(
                    ErrorKind::ResourceLimit,
                    "sink position cannot be represented",
                )
            })?;
            Ok(())
        })
    }

    fn seek<'a>(&'a mut self, position: u64) -> IoFuture<'a, ()> {
        Box::pin(async move {
            usize::try_from(position)
                .map_err(|_| Error::new(ErrorKind::InvalidInput, "seek position is too large"))?;
            self.position = position;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{Context, Poll, Waker};

    fn ready<T>(mut future: IoFuture<'_, T>) -> Result<T> {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => result,
            Poll::Pending => panic!("memory I/O unexpectedly returned a pending future"),
        }
    }

    #[test]
    fn memory_source_reads_partially_at_end_of_source() {
        let source = MemorySource::new(b"abcdef".to_vec());
        let mut destination = [0_u8; 4];
        assert_eq!(ready(source.read_at(4, &mut destination)).unwrap(), 2);
        assert_eq!(&destination[..2], b"ef");
    }

    #[test]
    fn memory_source_rejects_offsets_beyond_end() {
        let source = MemorySource::new(b"abc".to_vec());
        let mut destination = [0_u8; 1];
        let error = ready(source.read_at(4, &mut destination)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn memory_sink_seeks_overwrites_and_zero_fills() {
        let mut sink = MemorySink::new();
        ready(sink.write(b"abc")).unwrap();
        ready(sink.seek(1)).unwrap();
        ready(sink.write(b"Z")).unwrap();
        ready(sink.seek(5)).unwrap();
        ready(sink.write(b"!")).unwrap();
        assert_eq!(sink.as_slice(), b"aZc\0\0!");
    }
}
