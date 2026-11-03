package he

import (
	"runtime"
	"sync"
)

// Workers is the goroutine count for per-ciphertext loops. Each worker builds its own
// Lattigo encoder, encryptor or protocol instance, so nothing that draws randomness is
// shared between goroutines.
var Workers = runtime.NumCPU()

// parallelFor runs body(state, k) for k in [0, n) across Workers goroutines. Each
// goroutine calls newState once. It returns the first error by index order.
func parallelFor[S any](n int, newState func() (S, error), body func(S, int) error) error {
	w := min(Workers, n)
	if w < 1 {
		return nil
	}
	errs := make([]error, n)
	next := make(chan int, n)
	for k := 0; k < n; k++ {
		next <- k
	}
	close(next)
	var wg sync.WaitGroup
	var initErr error
	var once sync.Once
	for g := 0; g < w; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			st, err := newState()
			if err != nil {
				once.Do(func() { initErr = err })
				return
			}
			for k := range next {
				errs[k] = body(st, k)
			}
		}()
	}
	wg.Wait()
	if initErr != nil {
		return initErr
	}
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
