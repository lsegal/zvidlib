use crate::{Error, ErrorKind, Result};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
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
///
/// `is_seekable()` defaults to `true`, so implementers that cannot seek
/// (e.g. a streaming HTTP upload or a pipe) must override it to return
/// `false` *and* make `seek()` reject every call with
/// [`ErrorKind::Unsupported`]. Callers that require a seekable sink, such as
/// `crate::mp4::Mp4Muxer`, check `is_seekable()` up front and never call
/// `seek()` on a sink that reports `false`; a non-seekable sink's `seek()`
/// implementation exists only to fail safely if that contract is ever
/// violated.
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

/// A file on the local file system, read in place at the offsets asked for.
///
/// Every read completes before its future is first polled, so a synchronous
/// caller such as [`crate::codec::ExactFrameReader`] over an on-demand provider
/// can drive it directly. Clones share the open file, and no read moves a
/// cursor another one depends on.
#[cfg(any(unix, windows))]
#[derive(Clone, Debug)]
pub struct FileSource {
    file: std::sync::Arc<std::fs::File>,
    length: u64,
}

#[cfg(any(unix, windows))]
impl FileSource {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|error| {
            Error::new(
                ErrorKind::Io,
                format!("cannot open {}: {error}", path.display()),
            )
        })?;
        Self::from_file(file)
    }

    pub fn from_file(file: std::fs::File) -> Result<Self> {
        let length = file
            .metadata()
            .map_err(|error| Error::new(ErrorKind::Io, format!("cannot read file size: {error}")))?
            .len();
        Ok(Self {
            file: std::sync::Arc::new(file),
            length,
        })
    }

    fn read_some(&self, offset: u64, destination: &mut [u8]) -> std::io::Result<usize> {
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_at(&*self.file, destination, offset)
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::FileExt::seek_read(&*self.file, destination, offset)
        }
    }
}

#[cfg(any(unix, windows))]
impl ByteSource for FileSource {
    fn len(&self) -> Option<u64> {
        Some(self.length)
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        let result = if offset > self.length {
            Err(Error::new(
                ErrorKind::InvalidInput,
                "read offset is beyond the source",
            ))
        } else {
            // A positional read may return fewer bytes than asked for short of
            // the end, so read until the destination is full or the file ends.
            let mut filled = 0;
            loop {
                if filled == destination.len() {
                    break Ok(filled);
                }
                match self.read_some(offset + filled as u64, &mut destination[filled..]) {
                    Ok(0) => break Ok(filled),
                    Ok(read) => filled += read,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        break Err(Error::new(
                            ErrorKind::Io,
                            format!("cannot read file: {error}"),
                        ));
                    }
                }
            }
        };
        Box::pin(async move { result })
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

/// A sequential-only in-memory sink used to test the non-seekable `ByteSink`
/// contract: it reports `is_seekable() == false` and rejects every `seek()`
/// call with [`ErrorKind::Unsupported`].
#[doc(hidden)]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NonSeekableSink {
    bytes: Vec<u8>,
}

impl ByteSink for NonSeekableSink {
    fn position(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn write<'a>(&'a mut self, bytes: &'a [u8]) -> IoFuture<'a, ()> {
        Box::pin(async move {
            self.bytes.extend_from_slice(bytes);
            Ok(())
        })
    }

    fn seek<'a>(&'a mut self, _position: u64) -> IoFuture<'a, ()> {
        Box::pin(async move {
            Err(Error::new(
                ErrorKind::Unsupported,
                "this sink does not support seeking",
            ))
        })
    }
}

/// A [`ByteSource`] wrapper that caches fixed-size pages of the bytes it
/// reads, evicting the least-recently-used page once a configured byte
/// budget is exceeded.
///
/// This is how a [`crate::codec::SampleProvider`] backed by a container
/// track and a plain [`ByteSource`] (such as a file) keeps the memory for
/// compressed sample bytes bounded: samples near the playhead are read
/// through overlapping pages that stay cached, and pages behind the budget
/// are dropped rather than held for the life of the source.
///
/// Reads are not required to be page-aligned or page-sized; a read spanning
/// several pages fetches and caches each of them. Internal caching uses a
/// [`RefCell`], so a `CachingByteSource` is usable from one thread at a time,
/// matching how a reader built over it is used.
pub struct CachingByteSource<S> {
    inner: S,
    page_size: u64,
    budget_bytes: u64,
    pages: RefCell<HashMap<u64, Vec<u8>>>,
    lru: RefCell<VecDeque<u64>>,
    resident_bytes: RefCell<u64>,
}

impl<S: ByteSource> CachingByteSource<S> {
    /// `page_size` and `budget_bytes` must both be nonzero; a page larger
    /// than the budget is still cached (so a single read always succeeds)
    /// but is evicted again on the very next page fetch.
    pub fn new(inner: S, page_size: u64, budget_bytes: u64) -> Result<Self> {
        if page_size == 0 || budget_bytes == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "caching byte source requires a nonzero page size and budget",
            ));
        }
        Ok(Self {
            inner,
            page_size,
            budget_bytes,
            pages: RefCell::new(HashMap::new()),
            lru: RefCell::new(VecDeque::new()),
            resident_bytes: RefCell::new(0),
        })
    }

    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Total bytes currently cached, for tests and diagnostics.
    pub fn resident_bytes(&self) -> u64 {
        *self.resident_bytes.borrow()
    }

    fn touch(&self, page: u64) {
        let mut lru = self.lru.borrow_mut();
        lru.retain(|candidate| *candidate != page);
        lru.push_back(page);
    }

    fn evict_until_within_budget(&self) {
        loop {
            if *self.resident_bytes.borrow() <= self.budget_bytes {
                return;
            }
            let Some(evicted) = self.lru.borrow_mut().pop_front() else {
                return;
            };
            if let Some(bytes) = self.pages.borrow_mut().remove(&evicted) {
                *self.resident_bytes.borrow_mut() -= bytes.len() as u64;
            }
        }
    }

    /// Reads one page, from the cache if present, filling it from the inner
    /// source otherwise.
    async fn page(&self, page_index: u64) -> Result<Vec<u8>> {
        if let Some(cached) = self.pages.borrow().get(&page_index) {
            self.touch(page_index);
            return Ok(cached.clone());
        }
        let mut bytes = vec![0_u8; self.page_size as usize];
        let read = self
            .inner
            .read_at(page_index * self.page_size, &mut bytes)
            .await?;
        bytes.truncate(read);
        *self.resident_bytes.borrow_mut() += bytes.len() as u64;
        self.pages.borrow_mut().insert(page_index, bytes.clone());
        self.touch(page_index);
        self.evict_until_within_budget();
        Ok(bytes)
    }
}

impl<S: ByteSource> ByteSource for CachingByteSource<S> {
    fn len(&self) -> Option<u64> {
        self.inner.len()
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        Box::pin(async move {
            let mut produced = 0_usize;
            while produced < destination.len() {
                let position = offset + produced as u64;
                let page_index = position / self.page_size;
                let page_start = page_index * self.page_size;
                let in_page_offset = (position - page_start) as usize;
                let page = self.page(page_index).await?;
                if in_page_offset >= page.len() {
                    break;
                }
                let available = (page.len() - in_page_offset).min(destination.len() - produced);
                destination[produced..produced + available]
                    .copy_from_slice(&page[in_page_offset..in_page_offset + available]);
                produced += available;
                if page.len() < self.page_size as usize {
                    // The inner source ended partway through this page.
                    break;
                }
            }
            Ok(produced)
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

    #[test]
    fn non_seekable_sink_writes_but_rejects_seek() {
        let mut sink = NonSeekableSink::default();
        assert!(!sink.is_seekable());
        ready(sink.write(b"abc")).unwrap();
        assert_eq!(sink.bytes, b"abc");
        let error = ready(sink.seek(0)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    /// A source that counts every byte actually read from it, so a test can
    /// tell a cache hit from a fetch.
    struct CountingSource {
        inner: MemorySource,
        reads: std::cell::Cell<u64>,
    }

    impl ByteSource for CountingSource {
        fn len(&self) -> Option<u64> {
            self.inner.len()
        }

        fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
            Box::pin(async move {
                let read = self.inner.read_at(offset, destination).await?;
                self.reads.set(self.reads.get() + read as u64);
                Ok(read)
            })
        }
    }

    fn large_source(bytes: u64) -> CountingSource {
        CountingSource {
            inner: MemorySource::new((0..bytes).map(|value| value as u8).collect::<Vec<u8>>()),
            reads: std::cell::Cell::new(0),
        }
    }

    #[test]
    fn caching_byte_source_reads_match_the_inner_source_across_page_boundaries() {
        let cache = CachingByteSource::new(large_source(1000), 64, 1_000_000).unwrap();
        let mut destination = [0_u8; 100];
        assert_eq!(ready(cache.read_at(30, &mut destination)).unwrap(), 100);
        let expected: Vec<u8> = (30..130).map(|value| value as u8).collect();
        assert_eq!(&destination, expected.as_slice());
    }

    #[test]
    fn caching_byte_source_answers_a_repeated_read_without_fetching_again() {
        let cache = CachingByteSource::new(large_source(1000), 64, 1_000_000).unwrap();
        let mut destination = [0_u8; 10];
        ready(cache.read_at(0, &mut destination)).unwrap();
        let reads_after_first = cache.inner.reads.get();
        assert!(reads_after_first > 0);
        ready(cache.read_at(0, &mut destination)).unwrap();
        assert_eq!(
            cache.inner.reads.get(),
            reads_after_first,
            "a repeated read fetched from the inner source again instead of the cache"
        );
    }

    /// Reading all over a track much larger than the cache budget - the
    /// shape of scrubbing and sequential playback through a long file - must
    /// never grow resident bytes past the configured budget.
    #[test]
    fn caching_byte_source_stays_within_its_budget_while_scanning_a_large_source() {
        let total = 1_000_000_u64;
        let page_size = 4096_u64;
        let budget = 64 * 1024_u64;
        let cache = CachingByteSource::new(large_source(total), page_size, budget).unwrap();
        let mut destination = [0_u8; 1024];
        let mut offset = 0_u64;
        while offset + destination.len() as u64 <= total {
            ready(cache.read_at(offset, &mut destination)).unwrap();
            assert!(
                cache.resident_bytes() <= budget,
                "resident bytes {} exceeded the budget {budget} at offset {offset}",
                cache.resident_bytes()
            );
            // Jump far enough each time that pages do not all stay resident
            // through overlap alone.
            offset += 97 * 1024;
        }
    }

    #[test]
    fn caching_byte_source_rejects_a_zero_page_size_or_budget() {
        assert_eq!(
            CachingByteSource::new(MemorySource::default(), 0, 1024)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            CachingByteSource::new(MemorySource::default(), 1024, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
    }
}
