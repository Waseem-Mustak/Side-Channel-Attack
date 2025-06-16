/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  // Allocate buffer of size n * LINESIZE
  const buffer = new ArrayBuffer(n * LINESIZE);
  const view = new Uint8Array(buffer);
  
  // Array to store time measurements
  const times = [];
  
  // Read the buffer 10 times
  for (let iteration = 0; iteration < 10; iteration++) {
    const start = performance.now();
    
    // Read through buffer at intervals of LINESIZE
    for (let i = 0; i < n * LINESIZE; i += LINESIZE) {
      // Force a read of each cache line by accessing the byte
      const value = view[i];
    }
    
    const end = performance.now();
    times.push(end - start);
  }
  
  // Calculate median time
  times.sort((a, b) => a - b);
  const median = times.length % 2 === 0
    ? (times[times.length / 2 - 1] + times[times.length / 2]) / 2
    : times[Math.floor(times.length / 2)];
    
  return median;
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};
    
    // Test with n = 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000
    const testSizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000];
    
    for (const n of testSizes) {
      try {
        results[n] = readNlines(n);
      } catch (error) {
        console.error(`Failed for n=${n}:`, error);
        break;
      }
    }

    self.postMessage(results);
  }
});
