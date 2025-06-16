/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
const LLCSIZE = 16 * 1024 * 1024; // 32MB
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10; 

function sweep(P) {
    // Number of samples
    const K = Math.floor(TIME / P);
    // Allocate buffer of size LLCSIZE
    const buffer = new ArrayBuffer(LLCSIZE);
    const view = new Uint8Array(buffer);
    // Array to store sweep counts
    const counts = [];

    for (let k = 0; k < K; k++) {
        let count = 0;
        const start = performance.now();
        while (performance.now() - start < P) {
            // Sweep through the buffer at cache line intervals
            for (let i = 0; i < LLCSIZE; i += LINESIZE) {
                const value = view[i];
            }
            count++;
        }
        counts.push(count);
    }
    return counts;
}   

self.addEventListener('message', function(e) {
    if (e.data === 'start') {
        const result = sweep(P);
        self.postMessage(result);
    }
});