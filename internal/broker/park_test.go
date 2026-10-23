// Copyright 2026 Trevin Tindall (T-Crypt)
// SPDX-License-Identifier: Apache-2.0

package broker

// Park, driven through the lease table: admission over fake RAM readings,
// the parked transition on a later card read, grants beside a small
// residual, eviction of a big one, and the eviction table's three reasons.

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/T-Crypt/berth/internal/gpu"
	"github.com/T-Crypt/berth/internal/procs"
	"github.com/T-Crypt/berth/internal/proto"
)

// parkSetup is a broker whose RAM readings the test holds, with the park
// floor small so admission is the variable under test.
func parkSetup(t *testing.T, tree procs.Tree, avail uint64, pressure float64) (*Broker, *card, *reports) {
	t.Helper()
	b, c, r := setup(t, tree)
	b.cfg.ParkFloor = 1 * GiB
	b.cfg.ParkResidualMax = 2 * GiB
	b.cfg.MemAvailable = func() uint64 { return avail }
	b.cfg.MemPressure = func() float64 { return pressure }
	return b, c, r
}

// parked takes a lease through acquire, its measured peak, a drain, and
// admission into parking; the residual lands on the card and the lease
// becomes parked. The foreground waiter that triggered the drain is granted
// once the memory freed, and releases, so the queue ends empty.
func warmLease(t *testing.T, b *Broker, c *card, engine string, peak uint64, root, eng int) *Lease {
	t.Helper()
	l := must(t, acquire(b, Request{Engine: engine, Needs: peak}))
	l.Attach(root)
	c.set(gpu.Proc{PID: eng, Name: engine + "-engine", Used: peak})
	b.Tick(context.Background())

	w := acquire(b, Request{Engine: "next", Priority: Foreground, Needs: 20 * GiB})
	pending(t, w)
	drained(t, l)
	if ok, why := l.Park(); !ok {
		t.Fatalf("park refused: %s", why)
	}
	c.set(gpu.Proc{PID: eng, Name: engine + "-engine", Used: 512 << 20})
	l.Parked()
	b.Tick(context.Background())
	lw := must(t, w)
	lw.Release()
	b.Tick(context.Background())
	return l
}

func parkTree(root, eng int) procs.Fake {
	return procs.Fake{
		Parents: map[int]int{eng: root, root: 1},
		Starts:  map[int]uint64{root: 1, eng: 1},
	}
}

// Admission: the readings admit a parking lease, and a refusal carries its
// why line while the drain runs as today.
func TestParkAdmission(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	warmLease(t, b, c, "vllm", 8*GiB, 100, 101)
	if st := b.Status(); len(st.Leases) != 1 || st.Leases[0].State != "parked" {
		t.Fatalf("leases = %+v, want parked", st.Leases)
	}

	// A second park into RAM that cannot take it: refused with the why.
	tree.Parents[102] = 100
	tree.Starts[102] = 1
	m := must(t, acquire(b, Request{Engine: "sglang", Needs: 4 * GiB}))
	m.Attach(100)
	c.set(
		gpu.Proc{PID: 101, Name: "vllm-engine", Used: 512 << 20},
		gpu.Proc{PID: 102, Name: "sglang-engine", Used: 4 * GiB},
	)
	b.Tick(context.Background())
	// The waiter only triggers the drain; its patience must outlast a slow runner.
	w := acquire(b, Request{Engine: "next2", Priority: Foreground, Wait: 2 * time.Second})
	pending(t, w)
	drained(t, m)
	b.cfg.MemAvailable = func() uint64 { return 4 * GiB }
	ok, why := m.Park()
	if ok || !strings.Contains(why, "park refused") || !strings.Contains(why, "available above the floor") {
		t.Fatalf("park = %v %q, want a refusal with the why line", ok, why)
	}
	st := b.Status()
	var refused *proto.Lease
	for i := range st.Leases {
		if st.Leases[i].Engine == "sglang" {
			refused = &st.Leases[i]
		}
	}
	if refused == nil || refused.State != "draining" {
		t.Fatalf("leases = %+v, want the refused lease back on today's ladder", st.Leases)
	}
}

// A park whose footprint was never measured is refused: there is nothing
// to hold in RAM.
func TestParkRefusedWithoutAFootprint(t *testing.T) {
	b, _, _ := parkSetup(t, procs.Map{}, 8*GiB, 0)
	l := must(t, acquire(b, Request{Engine: "vllm"}))
	w := acquire(b, Request{Engine: "next", Priority: Foreground})
	pending(t, w)
	drained(t, l)
	ok, why := l.Park()
	if ok || !strings.Contains(why, "never held card memory") {
		t.Fatalf("park = %v %q", ok, why)
	}
	evs, _, _ := b.Events(0, false)
	if len(evs) == 0 || evs[len(evs)-1].Kind != proto.EvParkRefused {
		t.Fatalf("last event = %+v, want the refusal", evs)
	}
}

// The parked transition needs a card read after the park call; the residual
// counts as held from then on, the engine is never a stowaway, and status
// carries the footprint and the parked age.
func TestParkingBecomesParkedOnALaterRead(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, r := parkSetup(t, tree, 16*GiB, 0)
	l := must(t, acquire(b, Request{Engine: "vllm", Needs: 8 * GiB}))
	l.Attach(100)
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 8 * GiB})
	b.Tick(context.Background())
	w := acquire(b, Request{Engine: "next", Priority: Foreground, Needs: 20 * GiB})
	pending(t, w)
	drained(t, l)
	if ok, why := l.Park(); !ok {
		t.Fatalf("park refused: %s", why)
	}

	// A card read before the park call parks nothing.
	b.Tick(context.Background())
	if st := b.Status(); st.Leases[0].State != "parking" {
		t.Fatalf("state = %s, want parking before the call", st.Leases[0].State)
	}

	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 512 << 20})
	l.Parked()
	b.Tick(context.Background())
	st := b.Status()
	if st.Leases[0].State != "parked" || st.Leases[0].ParkedSince <= 0 {
		t.Fatalf("lease = %+v, want parked with an age", st.Leases[0])
	}
	if st.ParkedBytes != 8*GiB {
		t.Fatalf("parked bytes = %d, want the 8 GiB footprint", st.ParkedBytes)
	}
	if len(st.Squatters) != 0 || len(r.seen) != 0 {
		t.Fatalf("squatters = %+v, reports = %v", st.Squatters, r.seen)
	}

	// The residual is held, never free: a request that cannot fit beside
	// it waits.
	big := acquire(b, Request{Engine: "big", Needs: 24 * GiB})
	b.Tick(context.Background())
	pending(t, big)
}

// A park that leaves too much on the card is given up after the grace: the
// runner is told to stop the engine the usual way.
func TestParkResidualTooBigGivesUp(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	b.cfg.ReleaseGrace = time.Millisecond

	l := must(t, acquire(b, Request{Engine: "vllm", Needs: 8 * GiB}))
	l.Attach(100)
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 8 * GiB})
	b.Tick(context.Background())
	w := acquire(b, Request{Engine: "next", Priority: Foreground, Needs: 20 * GiB})
	pending(t, w)
	drained(t, l)
	if ok, why := l.Park(); !ok {
		t.Fatalf("park refused: %s", why)
	}

	// The card read after the park call still shows 4 GiB, past the max.
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 4 * GiB})
	l.Parked()
	time.Sleep(5 * time.Millisecond)
	b.Tick(context.Background())

	if why := drained(t, l); !strings.Contains(why, "park did not free the card") {
		t.Fatalf("drain reason %q", why)
	}
	if st := b.Status(); len(st.Leases) != 1 || st.Leases[0].State != "draining" {
		t.Fatalf("leases = %+v, want it back on the ladder", st.Leases)
	}
}

// A sized waiter builds beside a parked engine's residual.
func TestSizedWaiterFitsBesideParked(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	warmLease(t, b, c, "vllm", 8*GiB, 100, 101)
	must(t, acquire(b, Request{Engine: "chat", Needs: 20 * GiB}))
}

// A whole-card waiter is granted beside a small residual, with the grant
// event naming what it sits next to; a big residual is evicted first.
func TestWholeCardWaiterBesideParked(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := warmLease(t, b, c, "vllm", 8*GiB, 100, 101)
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 1 * GiB})
	b.Tick(context.Background())

	_, live, stop := b.Events(0, true)
	next := acquire(b, Request{Engine: "next"})
	var granted *proto.Event
	for e := range live {
		if e.Kind == proto.EvGranted && e.Engine == "next" {
			granted = &e
			break
		}
	}
	stop()
	if granted == nil {
		t.Fatal("the whole-card waiter was never granted")
	}
	if !strings.Contains(granted.Detail, "beside parked vllm (1024 MiB)") {
		t.Fatalf("grant detail = %q", granted.Detail)
	}
	must(t, next).Release()
	b.Tick(context.Background())

	// A residual past the max: the waiter evicts, least recently parked
	// first, and the parked lease is drained off the card.
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 3 * GiB})
	b.Tick(context.Background())
	warm := acquire(b, Request{Engine: "warm"})
	pending(t, warm)
	select {
	case why := <-l.Drains():
		if !strings.Contains(why, "needs the card") {
			t.Fatalf("eviction drain reason %q", why)
		}
	case <-time.After(time.Second):
		t.Fatal("the big residual was not evicted")
	}
	if st := b.Status(); st.Leases[0].State != "draining" {
		t.Fatalf("state = %s, want the evicted lease draining", st.Leases[0].State)
	}
}

// The eviction table: past the ttl, under the RAM floor, and under memory
// pressure. Each evicted lease is drained with its why.
func TestParkedEnginesAreEvicted(t *testing.T) {
	tree := parkTree(100, 101)
	tests := []struct {
		name     string
		age      time.Duration
		avail    uint64
		pressure float64
		want     string // a substring of the eviction why
	}{
		{"past the ttl", 45 * time.Minute, 16 * GiB, 0, "past the ttl"},
		{"under the RAM floor", time.Minute, 0, 0, "under the floor"},
		{"memory pressure", time.Minute, 16 * GiB, 11.0, "pressure"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, c, _ := parkSetup(t, tree, 16*GiB, 0)
			l := warmLease(t, b, c, "vllm", 8*GiB, 100, 101)
			b.cfg.MemAvailable = func() uint64 { return tt.avail }
			b.cfg.MemPressure = func() float64 { return tt.pressure }
			b.mu.Lock()
			l.l.parkedAt = time.Now().Add(-tt.age)
			b.mu.Unlock()

			b.Tick(context.Background())
			select {
			case why := <-l.Drains():
				if !strings.Contains(why, "evicted") || !strings.Contains(why, tt.want) {
					t.Fatalf("eviction drain reason %q, want %q", why, tt.want)
				}
			case <-time.After(time.Second):
				t.Fatal("the parked engine was not evicted")
			}
		})
	}
}

// An engine's park_ttl stops it sooner than the node's ttl: ComfyUI kept up
// on a drain gives its context back minutes later, not half an hour.
func TestParkedEngineOwnTTL(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := warmLease(t, b, c, "comfyui", 8*GiB, 100, 101)
	b.cfg.MemAvailable = func() uint64 { return 16 * GiB }
	b.cfg.MemPressure = func() float64 { return 0 }
	b.mu.Lock()
	l.l.parkTTL = 5 * time.Minute
	l.l.parkedAt = time.Now().Add(-6 * time.Minute)
	b.mu.Unlock()

	b.Tick(context.Background())
	select {
	case why := <-l.Drains():
		if !strings.Contains(why, "past the ttl") {
			t.Fatalf("eviction drain reason %q, want past the ttl", why)
		}
	case <-time.After(time.Second):
		t.Fatal("the parked engine outlived its own ttl")
	}
}

// A resume puts the parked lease back in the queue at its own priority,
// and the grant closes the resume signal for the runner's connection.
func TestResumeGrantsTheParkedLease(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := warmLease(t, b, c, "vllm", 8*GiB, 100, 101)

	// An engine that is not parked is refused, so the caller starts it
	// the normal way.
	if err := b.Resume("nobody"); err == nil || !strings.Contains(err.Error(), "not parked") {
		t.Fatalf("resume = %v", err)
	}

	// Priorities still apply: a foreground holder keeps the card from a
	// batch resume.
	game := must(t, acquire(b, Request{Engine: "game", Priority: Foreground}))
	if err := b.Resume("vllm"); err != nil {
		t.Fatal(err)
	}
	if st := b.Status(); len(st.Waiting) != 1 || st.Waiting[0].Engine != "vllm" {
		t.Fatalf("waiting = %+v, want the resumed lease queued", st.Waiting)
	}
	game.Release()
	b.Tick(context.Background())
	select {
	case <-l.Resumed():
	case <-time.After(time.Second):
		t.Fatal("the resumed lease was never granted")
	}
	if st := b.Status(); len(st.Leases) != 1 || st.Leases[0].State != "active" {
		t.Fatalf("leases = %+v", st.Leases)
	}
}

// A whole-card lease (no needs, as ComfyUI runs) resumes too: the resumed
// lease is still in the lease table while it waits, and it must not wait
// for its own memory to leave the card.
func TestResumeGrantsAWholeCardParkedLease(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := must(t, acquire(b, Request{Engine: "comfyui", Priority: Batch}))
	l.Attach(100)
	c.set(gpu.Proc{PID: 101, Name: "comfyui-engine", Used: 8 * GiB})
	b.Tick(context.Background())

	w := acquire(b, Request{Engine: "chat", Priority: Interactive})
	pending(t, w)
	drained(t, l)
	if ok, why := l.Park(); !ok {
		t.Fatalf("park refused: %s", why)
	}
	c.set(gpu.Proc{PID: 101, Name: "comfyui-engine", Used: 470 << 20})
	l.Parked()
	b.Tick(context.Background())
	must(t, w).Release()
	b.Tick(context.Background())

	if err := b.Resume("comfyui"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-l.Resumed():
	case <-time.After(time.Second):
		t.Fatalf("the resumed whole-card lease was never granted: %+v", b.Status().Waiting)
	}
	if st := b.Status(); len(st.Leases) != 1 || st.Leases[0].State != "active" {
		t.Fatalf("leases = %+v", st.Leases)
	}
}

// Park and resume cycle as often as switches need: the resume signal is one
// token per grant, sent again on the next cycle, and never a closed
// channel that would double-close on the second one.
func TestParkAndResumeCycleTwice(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := warmLease(t, b, c, "vllm", 8*GiB, 100, 101)

	for cycle := 1; cycle <= 2; cycle++ {
		if err := b.Resume("vllm"); err != nil {
			t.Fatalf("cycle %d: resume: %v", cycle, err)
		}
		select {
		case <-l.Resumed():
		case <-time.After(time.Second):
			t.Fatalf("cycle %d: the resumed lease was never granted", cycle)
		}

		// One grant, one token: a second receive must not answer.
		select {
		case <-l.Resumed():
			t.Fatalf("cycle %d: the grant token was received twice", cycle)
		case <-time.After(50 * time.Millisecond):
		}

		// Back on the card at full footprint, the next whole-card waiter
		// evicts the lease, and it parks again.
		c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 8 * GiB})
		b.Tick(context.Background())
		w := acquire(b, Request{Engine: "switch" + strconv.Itoa(cycle), Priority: Foreground, Needs: 20 * GiB})
		pending(t, w)
		drained(t, l)
		if ok, why := l.Park(); !ok {
			t.Fatalf("cycle %d: park refused: %s", cycle, why)
		}
		c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 512 << 20})
		l.Parked()
		b.Tick(context.Background())

		// The waiter holds the card beside the parked residual; it leaves
		// so the next cycle's grant can land.
		must(t, w).Release()
	}
}

// A waiter that arrives while an engine is still parking waits for the park
// to finish: the parking engine still holds its weights, and draining it
// mid-park would throw the park away.
func TestWaiterDoesNotEvictAParkingEngine(t *testing.T) {
	tree := parkTree(100, 101)
	b, c, _ := parkSetup(t, tree, 16*GiB, 0)
	l := must(t, acquire(b, Request{Engine: "vllm", Needs: 8 * GiB}))
	l.Attach(100)
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 8 * GiB})
	b.Tick(context.Background())
	// A whole-card foreground waiter drains the holder, which parks.
	w := acquire(b, Request{Engine: "whole", Priority: Foreground})
	pending(t, w)
	drained(t, l)
	if ok, why := l.Park(); !ok {
		t.Fatalf("park refused: %s", why)
	}

	// While it parks, the card still holds its 8 GiB: the waiter waits.
	b.Tick(context.Background())
	pending(t, w)
	select {
	case why := <-l.Drains():
		t.Fatalf("the parking engine was drained mid-park: %q", why)
	case <-time.After(200 * time.Millisecond):
	}
	if st := b.Status(); st.Leases[0].State != "parking" {
		t.Fatalf("state = %s, want still parking", st.Leases[0].State)
	}

	// Once parked with a small residual, the waiter is granted beside it.
	c.set(gpu.Proc{PID: 101, Name: "vllm-engine", Used: 512 << 20})
	l.Parked()
	b.Tick(context.Background())
	must(t, w)
}
