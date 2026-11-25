//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: AtomicBool,
    /// a critical dispatch or indirect bind waited past its grace budget without finishing
    critical_late: AtomicBool,
    /// lock-free mirror of `demand.wanted` for polling from worker queues without lock nesting
    wanted_flag: AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: AtomicBool,
    demand: Mutex<Demand>,
}

/// Whether one of the title's threads waits for the pipeline, and the worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// an unbounded waiter (`want`) requires this build until completion
    unbounded_wanted: bool,
    /// active timed waiters (cleared on timeout so late builds do not bypass throttling)
    timed_waiters: usize,
    /// thread id of the worker building it (preload or queue worker), 0 = none
    worker_thread: u32,
    /// base thread priority to restore when no longer wanted
    base_priority: i32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.wanted_flag.load(Ordering::Acquire)))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `wanted_flag`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.wanted_flag.load(Ordering::Acquire));
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

fn set_thread_priority_by_id(tid: u32, priority: i32) {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
    unsafe {
        let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, tid);
        if !h.is_null() {
            SetThreadPriority(h, priority);
            CloseHandle(h);
        }
    }
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetCurrentThreadId, SetThreadPriority,
        THREAD_PRIORITY_BELOW_NORMAL, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL,
    };
    let base_priority = if preload {
        THREAD_PRIORITY_LOWEST
    } else if crate::state::boot().is_some() {
        THREAD_PRIORITY_BELOW_NORMAL
    } else {
        THREAD_PRIORITY_NORMAL
    };
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    {
        let mut d = p.demand.lock().unwrap();
        d.worker_thread = unsafe { GetCurrentThreadId() };
        d.base_priority = base_priority;
        let target = if d.wanted { THREAD_PRIORITY_NORMAL } else { base_priority };
        unsafe { SetThreadPriority(GetCurrentThread(), target) };
    }
    complete(p, job());
    {
        // (under the lock: `want` / timed-wait transitions do not race priority restore)
        let mut d = p.demand.lock().unwrap();
        d.worker_thread = 0;
        unsafe { SetThreadPriority(GetCurrentThread(), base_priority) };
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads waits unconditionally for it: boost worker and wake pools.
    fn want(&self) {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        let boost = {
            let mut d = self.demand.lock().unwrap();
            d.unbounded_wanted = true;
            let first = !std::mem::replace(&mut d.wanted, true);
            self.wanted_flag.store(true, Ordering::Release);
            (first && d.worker_thread != 0 && d.base_priority != THREAD_PRIORITY_NORMAL).then_some(d.worker_thread)
        };
        if let Some(tid) = boost {
            set_thread_priority_by_id(tid, THREAD_PRIORITY_NORMAL);
        }
        crate::gpu::recomp::dxc::wake_pool();
        wake_queue();
    }

    /// A title thread starts a bounded wait for this build: boost worker while actively waiting.
    fn begin_timed_wait(&self) {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        let boost = {
            let mut d = self.demand.lock().unwrap();
            d.timed_waiters += 1;
            let first = !std::mem::replace(&mut d.wanted, true);
            self.wanted_flag.store(true, Ordering::Release);
            (first && d.worker_thread != 0 && d.base_priority != THREAD_PRIORITY_NORMAL).then_some(d.worker_thread)
        };
        if let Some(tid) = boost {
            set_thread_priority_by_id(tid, THREAD_PRIORITY_NORMAL);
        }
        crate::gpu::recomp::dxc::wake_pool();
        wake_queue();
    }

    /// Bounded wait finished (`timed_out`: clear wanted/priority boost once last waiter times out
    /// so late cold builds cannot bypass throttling or starve foreground threads).
    fn end_timed_wait(&self, timed_out: bool) {
        use windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
        let restore = {
            let mut d = self.demand.lock().unwrap();
            d.timed_waiters = d.timed_waiters.saturating_sub(1);
            if timed_out && d.timed_waiters == 0 && !d.unbounded_wanted {
                d.wanted = false;
                self.wanted_flag.store(false, Ordering::Release);
                (d.worker_thread != 0 && d.base_priority != THREAD_PRIORITY_NORMAL)
                    .then_some((d.worker_thread, d.base_priority))
            } else {
                None
            }
        };
        if let Some((tid, prio)) = restore {
            set_thread_priority_by_id(tid, prio);
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count
    frame: u64,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: 0, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64) {
        if frame != self.frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
    }

    fn begin(&mut self, id: u64, now: std::time::Instant) {
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(since);
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(*since) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

enum QueueItem {
    Pipeline(Arc<Pending>),
    Helper(Box<dyn FnOnce() + Send>),
}

impl QueueItem {
    fn is_orphan(&self) -> bool {
        matches!(self, Self::Pipeline(p) if p.orphan.load(Ordering::Acquire))
    }

    fn is_wanted(&self) -> bool {
        matches!(self, Self::Pipeline(p) if p.wanted_flag.load(Ordering::Acquire))
    }

    fn is_late(&self) -> bool {
        matches!(self, Self::Pipeline(p) if p.late.load(Ordering::Acquire))
    }

    fn is_compute(&self) -> bool {
        matches!(self, Self::Pipeline(p) if p.compute)
    }

    fn is_helper(&self) -> bool {
        matches!(self, Self::Helper(_))
    }
}

struct WorkQueueState {
    list: std::collections::VecDeque<QueueItem>,
    late_running: usize,
}

struct WorkQueue {
    state: Mutex<WorkQueueState>,
    cv: Condvar,
}

struct LateSlotGuard(&'static WorkQueue);

impl Drop for LateSlotGuard {
    fn drop(&mut self) {
        let mut st = self.0.state.lock().unwrap();
        st.late_running -= 1;
        self.0.cv.notify_all();
    }
}

static Q: OnceLock<WorkQueue> = OnceLock::new();

/// Pick the highest-priority non-orphan build from `list`:
/// 1. `wanted` (a title thread is actively waiting for it),
/// 2. non-late compute,
/// 3. non-late graphics,
/// 4. late builds only while `late_running < max_late` (keeping lanes open for fresh/warm PSOs).
fn pick_work_item<T>(
    list: &mut std::collections::VecDeque<T>,
    late_running: usize,
    max_late: usize,
    is_orphan: impl Fn(&T) -> bool,
    is_wanted: impl Fn(&T) -> bool,
    is_late: impl Fn(&T) -> bool,
    is_compute: impl Fn(&T) -> bool,
) -> Option<(T, bool)> {
    while list.front().is_some_and(&is_orphan) {
        list.pop_front();
    }
    if let Some(i) = list.iter().position(|x| !is_orphan(x) && is_wanted(x)) {
        let item = list.remove(i)?;
        return Some((item, false));
    }
    if let Some(i) = list.iter().position(|x| !is_orphan(x) && !is_late(x) && is_compute(x)) {
        let item = list.remove(i)?;
        return Some((item, false));
    }
    if let Some(i) = list.iter().position(|x| !is_orphan(x) && !is_late(x)) {
        let item = list.remove(i)?;
        return Some((item, false));
    }
    if late_running < max_late {
        if let Some(i) = list
            .iter()
            .position(|x| !is_orphan(x) && is_compute(x))
            .or_else(|| list.iter().position(|x| !is_orphan(x)))
        {
            let item = list.remove(i)?;
            return Some((item, true));
        }
    }
    None
}

fn pick_queue_item(st: &mut WorkQueueState, max_late: usize) -> Option<(QueueItem, bool)> {
    st.list.retain(|x| !x.is_orphan());
    if let Some(i) = st.list.iter().position(QueueItem::is_helper) {
        return st.list.remove(i).map(|item| (item, false));
    }
    pick_work_item(
        &mut st.list,
        st.late_running,
        max_late,
        QueueItem::is_orphan,
        QueueItem::is_wanted,
        QueueItem::is_late,
        QueueItem::is_compute,
    )
}

fn queue() -> &'static WorkQueue {
    Q.get_or_init(|| {
        let q = WorkQueue {
            state: Mutex::new(WorkQueueState {
                list: std::collections::VecDeque::new(),
                late_running: 0,
            }),
            cv: Condvar::new(),
        };
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        let max_late = n.saturating_sub(2).max(1);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || {
                let q = Q.wait();
                loop {
                    let (item, _late_guard) = {
                        let mut st = q.state.lock().unwrap();
                        loop {
                            if let Some((item, is_late)) = pick_queue_item(&mut st, max_late) {
                                let guard = if is_late {
                                    st.late_running += 1;
                                    Some(LateSlotGuard(q))
                                } else {
                                    None
                                };
                                break (item, guard);
                            }
                            st = q.cv.wait(st).unwrap();
                        }
                    };
                    match item {
                        QueueItem::Helper(work) => work(),
                        QueueItem::Pipeline(p) => {
                            // (None: a bind took it and builds it itself)
                            let job = p.job.lock().unwrap().take();
                            if let Some(job) = job {
                                build(&p, job, false);
                            }
                        }
                    }
                }
            });
        }
        q
    })
}

fn enqueue_work(p: Arc<Pending>) {
    let q = queue();
    q.state.lock().unwrap().list.push_back(QueueItem::Pipeline(p));
    q.cv.notify_all();
}

fn wake_queue() {
    if let Some(q) = Q.get() {
        let _g = q.state.lock().unwrap();
        q.cv.notify_all();
    }
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = std::sync::mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, job)) = job else { return };
        let _ = send.send((i, job()));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    let q = queue();
    {
        let mut st = q.state.lock().unwrap();
        for _ in 1..count.min(crate::gpu::recomp::dxc::compile_threads()) {
            let (drain, parent) = (drain.clone(), parent.clone());
            st.list.push_front(QueueItem::Helper(Box::new(move || {
                let before = BUILDING.with(|b| b.replace(parent));
                drain();
                BUILDING.with(|b| *b.borrow_mut() = before);
            })));
        }
    }
    q.cv.notify_all();
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) {
        results[i] = result;
    }
    results
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: AtomicBool::new(send),
        real: AtomicUsize::new(0),
        driver: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        critical_late: Default::default(),
        wanted_flag: AtomicBool::new(false),
        spared: Default::default(),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue_work(p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;
    use std::sync::mpsc;

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, began);
        calls.begin(2, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` the same bind
    /// waits until the pipeline is built.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: the bind waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, None), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    #[test]
    fn frame_wait_budget_caps_burst_and_resets_on_next_frame() {
        use std::time::{Duration, Instant};
        let mut fb = FrameBudget::default();
        let per_pso = Duration::from_millis(35);
        let per_frame = Duration::from_millis(75);

        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let (tx3, rx3) = mpsc::channel();
        let (tx4, rx4) = mpsc::channel();
        let (p1, p2, p3, p4) = (0x7E57_C001usize, 0x7E57_C002usize, 0x7E57_C003usize, 0x7E57_C004usize);
        submit(p1, true, Box::new(move || { let _ = rx1.recv_timeout(Duration::from_secs(5)); 0xA1 }));
        submit(p2, true, Box::new(move || { let _ = rx2.recv_timeout(Duration::from_secs(5)); 0xA2 }));
        submit(p3, true, Box::new(move || { let _ = rx3.recv_timeout(Duration::from_secs(5)); 0xA3 }));
        submit(p4, true, Box::new(move || { let _ = rx4.recv_timeout(Duration::from_secs(5)); 0xA4 }));

        let e1 = entry(p1).unwrap();
        let e2 = entry(p2).unwrap();
        let e3 = entry(p3).unwrap();
        let e4 = entry(p4).unwrap();

        let t0 = Instant::now();
        assert_eq!(resolve_within_with_frame(&e1, p1, per_pso, per_frame, 1, &mut fb, false), (p1, true));
        assert_eq!(resolve_within_with_frame(&e2, p2, per_pso, per_frame, 1, &mut fb, false), (p2, true));
        // Even with remaining budget (~0-5 ms), p3 + p4 must finish within per_frame + timer slack,
        // and once spent >= per_frame, p4 does not wait and neither p3 nor p4 is marked late.
        let _ = resolve_within_with_frame(&e3, p3, per_pso, per_frame, 1, &mut fb, false);
        let t4 = Instant::now();
        assert_eq!(resolve_within_with_frame(&e4, p4, per_pso, per_frame, 1, &mut fb, false), (p4, true));
        assert!(t4.elapsed() < Duration::from_millis(20), "exhausted frame budget must skip wait ({:?})", t4.elapsed());
        assert!(t0.elapsed() < Duration::from_millis(250), "burst waited {:?}", t0.elapsed());
        assert!(!e3.late.load(Ordering::Acquire), "frame-truncated PSO wait must not be marked late");
        assert!(!e4.late.load(Ordering::Acquire), "budget-skipped PSO must not be marked late");

        // Moving to frame 2 resets the budget so p4 gets a fresh wait window.
        let t_next = Instant::now();
        assert_eq!(resolve_within_with_frame(&e4, p4, per_pso, per_frame, 2, &mut fb, false), (p4, true));
        assert!(t_next.elapsed() >= Duration::from_millis(20), "new frame must reset wait budget ({:?})", t_next.elapsed());
        assert!(e4.late.load(Ordering::Acquire), "timed-out PSO in fresh frame becomes late");

        let _ = tx1.send(());
        let _ = tx2.send(());
        let _ = tx3.send(());
        let _ = tx4.send(());
        let start = Instant::now();
        for p in [p1, p2, p3, p4] {
            while is_pending(p) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(!is_pending(p));
            let _ = released(p);
        }
    }

    #[test]
    fn critical_wait_uses_grace_budget_and_truncated_remainder_stays_non_late() {
        use std::time::{Duration, Instant};
        assert_eq!(critical_frame_budget_for(Duration::from_millis(250)), Duration::from_millis(2_000));
        assert_eq!(critical_frame_budget_for(Duration::from_millis(1_000)), Duration::from_millis(2_000));
        assert_eq!(critical_frame_budget_for(Duration::from_millis(3_500)), Duration::from_millis(3_500));

        let mut fb = FrameBudget::default();
        let normal_per_pso = Duration::from_millis(25);
        let normal_per_frame = Duration::from_millis(50);
        let crit_per_pso = Duration::from_millis(500);
        let crit_per_frame = Duration::from_millis(1_000);

        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let (tx3, rx3) = mpsc::channel();
        let (p1, p2, p3) = (0x7E57_C021usize, 0x7E57_C022usize, 0x7E57_C023usize);
        submit(p1, true, Box::new(move || { let _ = rx1.recv_timeout(Duration::from_secs(5)); 0xC1 }));
        submit(p2, true, Box::new(move || { let _ = rx2.recv_timeout(Duration::from_secs(5)); 0xC2 }));
        submit_lazy(p3, true, Box::new(move || { let _ = rx3.recv_timeout(Duration::from_secs(5)); 0xC3 }), 0);

        let e1 = entry(p1).unwrap();
        let e2 = entry(p2).unwrap();
        let e3 = entry(p3).unwrap();

        // Exhaust the normal per-frame budget via p1 and p2 (which becomes frame-truncated, not late).
        assert_eq!(resolve_within_with_frame(&e1, p1, normal_per_pso, normal_per_frame, 1, &mut fb, false), (p1, true));
        fb.record(1, Duration::from_millis(15));
        assert_eq!(resolve_within_with_frame(&e2, p2, normal_per_pso, normal_per_frame, 1, &mut fb, false), (p2, true));
        assert!(e1.late.load(Ordering::Acquire));
        assert!(!e2.late.load(Ordering::Acquire), "truncated frame wait must not set late");
        assert!(!e1.critical_late.load(Ordering::Acquire), "normal wait must not set critical_late");

        // e1 timed out on normal wait (`e1.late == true`) but still gets one critical grace attempt.
        let t_crit = Instant::now();
        assert_eq!(resolve_within_with_frame(&e1, p1, Duration::from_millis(30), crit_per_frame, 1, &mut fb, true), (p1, true));
        assert!(t_crit.elapsed() >= Duration::from_millis(15), "critical path must retry after normal late ({:?})", t_crit.elapsed());
        assert!(e1.critical_late.load(Ordering::Acquire));

        // Release p1 and p2 workers before running p3 so concurrent worker-saturating tests cannot
        // form a circular wait against p1/p2 on 3-worker runners.
        let _ = tx1.send(());
        let _ = tx2.send(());

        // p3 takes ~40ms (longer than normal 25ms, well inside critical 500ms): even though the
        // normal frame budget is exhausted, the critical grace frame budget lets it finish.
        let wake_p3 = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            let _ = tx3.send(());
        });
        assert_eq!(resolve_within_with_frame(&e3, p3, crit_per_pso, crit_per_frame, 1, &mut fb, true), (0xC3, false));
        wake_p3.join().unwrap();

        let start = Instant::now();
        for p in [p1, p2, p3] {
            while is_pending(p) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert!(!is_pending(p));
            let _ = released(p);
        }
    }

    #[test]
    fn timed_out_wait_clears_wanted_flag() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C010usize;
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            started_tx.send(wanted_here()).unwrap();
            let _ = done_rx.recv_timeout(Duration::from_secs(5));
            0xB1B1
        }));
        let wanted_probe = started_rx.recv().unwrap();
        let p = entry(standin).unwrap();
        p.begin_timed_wait();
        assert!(wanted_probe(), "active timed wait must raise wanted_flag");
        p.end_timed_wait(true);
        assert!(!wanted_probe(), "timed-out wait must clear wanted_flag");
        let mut fb = FrameBudget::default();
        assert_eq!(
            resolve_within_with_frame(&p, standin, Duration::from_millis(25), Duration::from_millis(200), 1, &mut fb, false),
            (standin, true)
        );
        assert!(!wanted_probe(), "timed-out wait must clear wanted_flag so late build yields to throttling");
        assert!(p.late.load(Ordering::Acquire));
        let _ = done_tx.send(());
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(released(standin), 0xB1B1);
    }

    #[test]
    fn work_queue_prioritizes_wanted_and_reserves_lanes_for_fresh_builds() {
        use std::collections::VecDeque;
        #[derive(Debug, PartialEq, Eq)]
        struct Item {
            id: u32,
            orphan: bool,
            wanted: bool,
            late: bool,
            compute: bool,
        }
        let mut q = VecDeque::from([
            Item { id: 1, orphan: true, wanted: false, late: false, compute: false },
            Item { id: 2, orphan: false, wanted: false, late: true, compute: false },
            Item { id: 3, orphan: false, wanted: false, late: false, compute: false },
            Item { id: 4, orphan: false, wanted: false, late: false, compute: true },
            Item { id: 5, orphan: false, wanted: true, late: true, compute: false },
            Item { id: 6, orphan: false, wanted: false, late: true, compute: true },
        ]);
        let pick = |q: &mut VecDeque<Item>, late_running, max_late| {
            pick_work_item(q, late_running, max_late, |x| x.orphan, |x| x.wanted, |x| x.late, |x| x.compute)
                .map(|(x, is_late)| (x.id, is_late))
        };
        assert_eq!(pick(&mut q, 0, 2), Some((5, false)), "wanted jumps ahead of everything");
        assert_eq!(pick(&mut q, 0, 2), Some((4, false)), "non-late compute precedes non-late graphics");
        assert_eq!(pick(&mut q, 0, 2), Some((3, false)), "non-late graphics precedes late items");
        assert_eq!(pick(&mut q, 2, 2), None, "saturated late cap keeps lanes open for fresh PSOs");
        assert_eq!(pick(&mut q, 1, 2), Some((6, true)), "under late cap, late compute precedes late graphics");
        assert_eq!(pick(&mut q, 1, 2), Some((2, true)), "finally late graphics");
        assert_eq!(pick(&mut q, 0, 2), None);
    }

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            let stand = p.stand(pso);
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue_work(p);
            }
            stand
        }
        r => r,
    }
}

/// Default per-PSO bind wait when running a title: warm compute PSOs from the disk driver
/// cache finish in 0-111 ms, while cold compiles take 261 ms to 65 s.
const DEFAULT_TITLE_WAIT_MS: u64 = 250;
/// Maximum cumulative wait per thread within one PresentX frame, preventing a burst of
/// cold PSOs from exceeding a title's 30 s hang watchdog.
const DEFAULT_FRAME_WAIT_BUDGET_MS: u64 = 2_000;

#[derive(Clone, Copy)]
enum EnvWaitBudget {
    Unset,
    Unbounded,
    Millis(u64),
}

const CRITICAL_FRAME_WAIT_CAP_MS: u64 = 2_000;

fn env_wait_budget() -> EnvWaitBudget {
    static B: OnceLock<EnvWaitBudget> = OnceLock::new();
    *B.get_or_init(|| match std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(0) => EnvWaitBudget::Unbounded,
        Some(ms) => EnvWaitBudget::Millis(ms),
        None => EnvWaitBudget::Unset,
    })
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; `0`: until built;
/// unset: bounded to `DEFAULT_TITLE_WAIT_MS` when running a title so cold bursts cannot trip its
/// hang watchdog, unbounded otherwise).
fn wait_budget() -> Option<std::time::Duration> {
    match env_wait_budget() {
        EnvWaitBudget::Unbounded => None,
        EnvWaitBudget::Millis(ms) => Some(std::time::Duration::from_millis(ms)),
        EnvWaitBudget::Unset => crate::state::boot().is_some().then(|| std::time::Duration::from_millis(DEFAULT_TITLE_WAIT_MS)),
    }
}

fn critical_wait_budget(base: Option<std::time::Duration>) -> Option<std::time::Duration> {
    let base = base?;
    if matches!(env_wait_budget(), EnvWaitBudget::Millis(_)) {
        Some(base)
    } else {
        Some(base.saturating_mul(4))
    }
}

#[derive(Default)]
struct FrameBudget {
    frame: u64,
    spent: std::time::Duration,
}

impl FrameBudget {
    fn remaining(&mut self, frame: u64, per_pso: std::time::Duration, per_frame: std::time::Duration) -> std::time::Duration {
        if self.frame != frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
        per_pso.min(per_frame.saturating_sub(self.spent))
    }

    fn record(&mut self, frame: u64, waited: std::time::Duration) {
        if self.frame != frame {
            self.frame = frame;
            self.spent = std::time::Duration::ZERO;
        }
        self.spent = self.spent.saturating_add(waited);
    }
}

thread_local! {
    static FRAME_BUDGET: std::cell::RefCell<FrameBudget> = std::cell::RefCell::new(FrameBudget::default());
}

fn frame_budget_for(per_pso: std::time::Duration) -> std::time::Duration {
    std::time::Duration::from_millis(DEFAULT_FRAME_WAIT_BUDGET_MS).max(per_pso)
}

fn critical_frame_budget_for(per_pso: std::time::Duration) -> std::time::Duration {
    let scaled = frame_budget_for(per_pso).saturating_mul(4);
    let cap = std::time::Duration::from_millis(CRITICAL_FRAME_WAIT_CAP_MS);
    scaled.min(cap).max(per_pso)
}

/// `resolve`, waiting for a pipeline still being built – built here if unbounded and no worker
/// has started it yet, or waited on workers up to the per-PSO and per-frame budget.
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_checked_for(pso, wait_budget()).0
}

/// Like `resolve_wait`, also returning whether `pso` is still building (timed out or late).
pub fn resolve_wait_checked(pso: usize) -> (usize, bool) {
    resolve_wait_checked_for(pso, wait_budget())
}

/// Critical wait for `Dispatch` / `ExecuteIndirect`: grants a 4x per-PSO grace within the
/// 2_000 ms (`CRITICAL_FRAME_WAIT_CAP_MS`) per-frame cap when `XCL_PSO_WAIT_MS` is unset, and
/// allows one critical retry even if an earlier non-critical bind already marked `late`.
pub fn resolve_wait_critical(pso: usize) -> (usize, bool) {
    resolve_wait_checked_with(pso, critical_wait_budget(wait_budget()), true)
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
#[cfg(test)]
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    resolve_wait_checked_for(pso, budget).0
}

#[cfg(test)]
pub(super) fn resolve_wait_checked_for_test(pso: usize, budget: std::time::Duration) -> (usize, bool) {
    resolve_wait_checked_for(pso, Some(budget))
}

fn resolve_wait_checked_for(pso: usize, budget: Option<std::time::Duration>) -> (usize, bool) {
    resolve_wait_checked_with(pso, budget, false)
}

fn resolve_wait_checked_with(pso: usize, budget: Option<std::time::Duration>, critical: bool) -> (usize, bool) {
    let Some(p) = entry(pso) else { return (pso, false) };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        let r = resolve_spared(&p, pso);
        let pending = p.real.load(Ordering::Acquire) == 0 && !*p.done.lock().unwrap();
        return (r, pending);
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    if let Some(budget) = budget {
        let frame = super::present::frame_generation();
        let per_frame = if critical { critical_frame_budget_for(budget) } else { frame_budget_for(budget) };
        return FRAME_BUDGET.with(|fb| resolve_within_with_frame(&p, pso, budget, per_frame, frame, &mut fb.borrow_mut(), critical));
    }
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(&p, job());
    }
    p.want();
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    let real = match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    (real, false)
}

/// `resolve_wait` for at most `per_pso` (and `per_frame` cumulatively in `frame`): the build goes
/// to the workers (never this thread), and a pipeline not built after waiting the full `per_pso`
/// is marked late – its binds go on without it until a worker finishes it. When the per-frame
/// budget truncates or skips the wait (`wait < per_pso`), the pipeline is NOT marked late so it
/// keeps its chance on the next frame (or on a critical bind).
fn resolve_within_with_frame(
    p: &Arc<Pending>,
    pso: usize,
    per_pso: std::time::Duration,
    per_frame: std::time::Duration,
    frame: u64,
    fb: &mut FrameBudget,
    critical: bool,
) -> (usize, bool) {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    let r = p.real.load(Ordering::Acquire);
    if r != 0 {
        return (r, false);
    }
    if *p.done.lock().unwrap() {
        return (built(p), false);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue_work(p.clone());
    }
    let late_flag = if critical { &p.critical_late } else { &p.late };
    if late_flag.load(Ordering::Acquire) {
        let r = p.real.load(Ordering::Acquire);
        if r != 0 {
            return (r, false);
        }
        let finished = *p.done.lock().unwrap();
        return (if finished { built(p) } else { p.stand(pso) }, !finished);
    }
    let wait = fb.remaining(frame, per_pso, per_frame);
    if wait.is_zero() {
        let finished = *p.done.lock().unwrap();
        return (if finished { built(p) } else { p.stand(pso) }, !finished);
    }
    p.begin_timed_wait();
    let t0 = std::time::Instant::now();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), wait, |d| !*d).unwrap();
    let finished = *done;
    drop(done);
    fb.record(frame, t0.elapsed());
    p.end_timed_wait(!finished);
    if finished {
        return (built(p), false);
    }
    if wait >= per_pso {
        late_flag.store(true, Ordering::Release);
        if critical {
            p.late.store(true, Ordering::Release);
        }
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)",
                wait.as_millis(),
                n + 1
            );
        }
    }
    (p.stand(pso), true)
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.real.load(Ordering::Acquire) {
        0 => {
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue_work(p.clone());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// `pso` is a stand-in whose background build finished without a real pipeline (`0`).
pub fn is_failed_standin(pso: usize) -> bool {
    entry(pso).is_some_and(|p| *p.done.lock().unwrap() && p.real.load(Ordering::Acquire) == 0)
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
