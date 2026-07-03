package llama

import (
	"fmt"
	"sort"
	"sync"
)

// Pool manages multiple independently loaded models addressed by name, so an
// application can serve several models (or several instances of one model)
// concurrently. Each model runs on its own context; predictions against
// different names run in parallel, while calls to the same name are
// serialized by that instance's own lock.
type Pool struct {
	mu     sync.RWMutex
	models map[string]*LLama
}

func NewPool() *Pool {
	return &Pool{models: make(map[string]*LLama)}
}

// Load opens the model at path and registers it under name. The name must be
// unused; load a second instance of the same file under a different name for
// parallel inference on one model.
func (p *Pool) Load(name, path string, opts ...ModelOption) (*LLama, error) {
	p.mu.Lock()
	if _, ok := p.models[name]; ok {
		p.mu.Unlock()
		return nil, fmt.Errorf("model %q is already loaded", name)
	}
	// Reserve the name before the (slow) load so concurrent Loads of the same
	// name fail fast instead of loading twice.
	p.models[name] = nil
	p.mu.Unlock()

	model, err := New(path, opts...)
	if err != nil {
		p.mu.Lock()
		delete(p.models, name)
		p.mu.Unlock()
		return nil, fmt.Errorf("load model %q: %w", name, err)
	}

	p.mu.Lock()
	p.models[name] = model
	p.mu.Unlock()
	return model, nil
}

// Get returns the named model, or false when it is unknown or still loading.
func (p *Pool) Get(name string) (*LLama, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	model, ok := p.models[name]
	return model, ok && model != nil
}

// Names lists the registered model names in sorted order.
func (p *Pool) Names() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	names := make([]string, 0, len(p.models))
	for name, model := range p.models {
		if model != nil {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// Predict runs a prediction on the named model.
func (p *Pool) Predict(name, text string, opts ...PredictOption) (string, error) {
	model, ok := p.Get(name)
	if !ok {
		return "", fmt.Errorf("model %q is not loaded", name)
	}
	return model.Predict(text, opts...)
}

// Embeddings computes embeddings on the named model.
func (p *Pool) Embeddings(name, text string, opts ...PredictOption) ([]float32, error) {
	model, ok := p.Get(name)
	if !ok {
		return nil, fmt.Errorf("model %q is not loaded", name)
	}
	return model.Embeddings(text, opts...)
}

// Remove frees the named model and drops it from the pool.
func (p *Pool) Remove(name string) {
	p.mu.Lock()
	model := p.models[name]
	delete(p.models, name)
	p.mu.Unlock()

	if model != nil {
		model.Free()
	}
}

// Free releases every model in the pool.
func (p *Pool) Free() {
	p.mu.Lock()
	models := p.models
	p.models = make(map[string]*LLama)
	p.mu.Unlock()

	for _, model := range models {
		if model != nil {
			model.Free()
		}
	}
}
