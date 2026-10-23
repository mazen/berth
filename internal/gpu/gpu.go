// Copyright 2026 Trevin Tindall (T-Crypt)
// SPDX-License-Identifier: Apache-2.0

// Package gpu reads what the card reports: total, used and free memory, and
// the processes holding it. Probes observe the driver without opening an
// inference context: NVIDIA through nvidia-smi, AMD through Linux sysfs/DRM.
package gpu

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"
)

const MiB = 1 << 20
const GiB = 1 << 30

// Proc is one compute process holding memory on the card.
type Proc struct {
	PID   int
	Name  string
	Used  uint64 // bytes
	Start uint64 // process start time, when sampled by the probe; 0 = broker resolves it
	GTT   uint64 // APU GPU-accessible system RAM, kept separate from Used (VRAM)
}

// Device is one card as the driver reports it. Telemetry the driver does
// not report ("[N/A]") reads -1, never a made-up zero.
type Device struct {
	Index    int
	Name     string
	BusID    string // pci.bus_id, e.g. "00000000:01:00.0"; empty when not queried
	Total    uint64
	Used     uint64
	Free     uint64
	APU      bool
	GTTTotal uint64 // driver capacity, not extra/dedicated VRAM or a reservation
	GTTUsed  uint64

	Util       float64 // %
	Temp       float64 // C
	Power      float64 // W
	PowerLimit float64 // W
	Fan        float64 // %
	ClockSM    float64 // MHz
	ClockMem   float64 // MHz
	PCIeGen    float64
	PCIeWidth  float64
}

// Query is the per-card field list, memory first.
const Query = "index,name,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current"

// Snapshot is one reading: the scheduled card in full, every card's memory
// in Devices, so a multi-GPU box is inventoried with its real limits.
type Snapshot struct {
	Devices  []Device
	Index    int
	Name     string
	Total    uint64
	Used     uint64
	Free     uint64
	Procs    []Proc
	At       time.Time
	APU      bool
	GTTTotal uint64
	GTTUsed  uint64
	// Guarded probes need fresh, successful observations for admission and
	// card-wide confirmation of release: a /proc entry can disappear before
	// the driver has reclaimed its buffers. NVIDIA retains its existing policy.
	Guarded bool
	MaxAge  time.Duration
	// Unobserved names same-user PIDs whose fdinfo was inaccessible. They
	// are not an empty process list: the broker retains prior ownership and
	// uses card-wide counters to guard admissions and release confirmation.
	Unobserved []int
}

// Baseline is memory in use that no compute process accounts for: the
// desktop, the driver, graphics clients nvidia-smi does not list.
func (s Snapshot) Baseline() uint64 {
	var procs uint64
	for _, p := range s.Procs {
		procs += p.Used
	}
	if procs >= s.Used {
		return 0
	}
	return s.Used - procs
}

// GTTBaseline is APU shared memory not privately attributed to a visible client.
// It must return independently of the reserved-UMA baseline before release.
func (s Snapshot) GTTBaseline() uint64 {
	var procs uint64
	for _, p := range s.Procs {
		procs += p.GTT
	}
	if procs >= s.GTTUsed {
		return 0
	}
	return s.GTTUsed - procs
}

// graphicsProcBytes is how much dedicated VRAM a process nvidia-smi does
// not list must hold before it joins the snapshot's process list. Desktop
// compositors sit well under it; a game, a renderer or a stray compute
// process does not.
const graphicsProcBytes = 64 * MiB

// luid is one GPU adapter's identity in a "GPU Process Memory" instance
// name: the high and low 32-bit parts of the adapter LUID.
type luid struct {
	hi, lo uint32
}

// valid is true when a "luid_..." part was actually parsed, as opposed to an
// instance name that carries none.
func (l luid) valid() bool { return l.hi != 0 || l.lo != 0 }

// parseInstance reads the PID and the adapter LUID out of a "GPU Process
// Memory" instance name, which the engine dresses as
// "pid_1234_luid_0x00000000_0x00015adb_phys_0". ok is false when the name
// carries no usable PID; luid is zero when it carries no "luid_..." part.
func parseInstance(instance string) (pid int, l luid, ok bool) {
	const prefix = "pid_"
	if !strings.HasPrefix(instance, prefix) {
		return 0, luid{}, false
	}
	digits := strings.TrimPrefix(instance, prefix)
	if i := strings.IndexByte(digits, '_'); i >= 0 {
		digits = digits[:i]
	}
	pid, err := strconv.Atoi(digits)
	if err != nil || pid <= 0 {
		return 0, luid{}, false
	}
	if i := strings.Index(instance, "_luid_0x"); i >= 0 {
		l, _ = parseLuid(instance[i+len("_luid_0x"):])
	}
	return pid, l, true
}

// parseLuid reads "0x<hi>_0x<lo>[_...]" into its two parts, high part first.
func parseLuid(s string) (luid, bool) {
	parts := strings.SplitN(s, "_", 3)
	if len(parts) < 2 {
		return luid{}, false
	}
	hi, ok := hexPart(parts[0])
	if !ok {
		return luid{}, false
	}
	lo, ok := hexPart(strings.TrimPrefix(parts[1], "0x"))
	if !ok {
		return luid{}, false
	}
	return luid{hi: hi, lo: lo}, true
}

func hexPart(s string) (uint32, bool) {
	v, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return 0, false
	}
	return uint32(v), true
}

// filterByLuid keeps the instances that sit on the adapters in want, so a
// multi-GPU box reports only the card it governs. A nil want keeps every
// instance, as when the governed card's adapter could not be pinned down.
func filterByLuid(items []dedicatedUsage, want map[luid]bool) []dedicatedUsage {
	if want == nil {
		return items
	}
	out := make([]dedicatedUsage, 0, len(items))
	for _, it := range items {
		_, l, ok := parseInstance(it.Instance)
		if !ok || !l.valid() || !want[l] {
			continue
		}
		out = append(out, it)
	}
	return out
}

// dedicatedUsage is one PDH sample: the instance name and its Dedicated
// Usage in bytes.
type dedicatedUsage struct {
	Instance string
	Bytes    uint64
}

// BusID is the governed card's pci.bus_id, from this read's device list.
func BusID(s Snapshot) string {
	for _, d := range s.Devices {
		if d.Index == s.Index {
			return d.BusID
		}
	}
	return ""
}

// pick finds the scheduled card: by bus id once it is known, so a card that
// took the index of one passed through to a VM is never read in its place,
// and by index before that. A bus id matches in nvidia-smi's form or in
// sysfs form, as --gpu takes it.
func pick(devs []Device, index int, busID string) (Device, error) {
	for _, d := range devs {
		if busID != "" && sameBus(d.BusID, busID) || busID == "" && d.Index == index {
			return d, nil
		}
	}
	if busID != "" {
		return Device{}, fmt.Errorf("no GPU at %s (found %d)", busID, len(devs))
	}
	return Device{}, fmt.Errorf("no GPU with index %d (found %d)", index, len(devs))
}

// sameBus is whether two bus ids name one PCI function, whatever the width
// of their domain.
func sameBus(a, b string) bool {
	if x, y := SysfsAddr(a), SysfsAddr(b); x != "" && y != "" {
		return x == y
	}
	return strings.EqualFold(a, b)
}

// parseBusID splits a pci.bus_id like "00000000:01:00.0" into its PCI bus,
// device and function. The domain is not matched: the adapter address a
// Windows query returns carries no domain, so a multi-domain box falls back
// to keeping every adapter.
func parseBusID(s string) (bus, device, function uint32, ok bool) {
	i := strings.IndexByte(s, ':')
	if i < 0 {
		return 0, 0, 0, false
	}
	rest := s[i+1:]
	j := strings.IndexByte(rest, ':')
	if j < 0 {
		return 0, 0, 0, false
	}
	k := strings.IndexByte(rest[j+1:], '.')
	if k < 0 {
		return 0, 0, 0, false
	}
	busHex, devHex, fnHex := rest[:j], rest[j+1:j+1+k], rest[j+1+k+1:]
	b, e1 := strconv.ParseUint(busHex, 16, 32)
	d, e2 := strconv.ParseUint(devHex, 16, 32)
	f, e3 := strconv.ParseUint(fnHex, 16, 32)
	if e1 != nil || e2 != nil || e3 != nil {
		return 0, 0, 0, false
	}
	return uint32(b), uint32(d), uint32(f), true
}

// dedicatedByPid sums one read's samples by the PID their instance names
// carry. One process can hold both cards of a multi-adapter board, so a
// PID's figures come together. An instance that is not a process (none is
// known) and a zero sample contribute nothing.
func dedicatedByPid(items []dedicatedUsage) map[int]uint64 {
	out := map[int]uint64{}
	for _, it := range items {
		pid, _, ok := parseInstance(it.Instance)
		if !ok || it.Bytes == 0 {
			continue
		}
		out[pid] += it.Bytes
	}
	return out
}

// cappedRead is one PID whose raw PDH sum exceeded the card's total memory.
type cappedRead struct {
	pid   int
	raw   uint64
	total uint64
}

// capToTotal limits every PID's dedicated sum to the card's total memory, so
// a driver's committed reservation cannot read as more than the card holds,
// and returns the PIDs it capped, for the caller to log.
func capToTotal(dedicated map[int]uint64, total uint64) (map[int]uint64, []cappedRead) {
	if total == 0 {
		return dedicated, nil
	}
	out := make(map[int]uint64, len(dedicated))
	var reads []cappedRead
	for pid, v := range dedicated {
		if v > total {
			reads = append(reads, cappedRead{pid: pid, raw: v, total: total})
			v = total
		}
		out[pid] = v
	}
	return out, reads
}

// mergeProcs folds PDH's dedicated-memory figures into the compute-app
// list. A listed process whose Used is 0 (WDDM reports [N/A]) takes the
// PDH figure, and nvidia-smi's own number wins when it has one. Under WDDM
// nvidia-smi lists every graphics client too, with [N/A], so a process
// either list names joins only when its dedicated use clears
// graphicsProcBytes. The compositor never joins: it is the desktop, and its
// PDH figure is a committed reservation far above what it holds (issue 149),
// so its memory stays in the baseline as Xorg's does on Linux. PDH does not
// say a PID's name, so the process table does when it can (names, read once
// per probe read) and a PID the table does not know keeps an empty Name.
// Card totals still come from the driver.
func mergeProcs(procs []Proc, dedicated map[int]uint64, names map[int]string) []Proc {
	if len(dedicated) == 0 {
		return procs
	}
	listed := make(map[int]bool, len(procs))
	out := make([]Proc, 0, len(procs)+len(dedicated))
	for _, p := range procs {
		listed[p.PID] = true
		// nvidia-smi names dwm.exe "[Insufficient Permissions]"; the
		// process table names it either way.
		if compositor(p.Name) || compositor(names[p.PID]) {
			continue
		}
		if p.Used == 0 {
			if d := dedicated[p.PID]; d > graphicsProcBytes {
				p.Used = d
			} else {
				continue
			}
		}
		out = append(out, p)
	}
	var fresh []int
	for pid, d := range dedicated {
		if !listed[pid] && d > graphicsProcBytes && !compositor(names[pid]) {
			fresh = append(fresh, pid)
		}
	}
	sort.Ints(fresh)
	for _, pid := range fresh {
		out = append(out, Proc{PID: pid, Name: names[pid], Used: dedicated[pid]})
	}
	return out
}

// compositor is whether name, a bare exe name or a full Windows path, is
// the Desktop Window Manager.
func compositor(name string) bool {
	return strings.EqualFold(name[strings.LastIndexAny(name, `\/`)+1:], "dwm.exe")
}

// Probe reads the card.
type Probe interface {
	Read(ctx context.Context) (Snapshot, error)
}

// Pauser is a probe that keeps a process on the card between reads (the
// nvidia-smi loop) and can stop it while the card's driver is switched.
type Pauser interface {
	Pause(bool)
}

// logger returns l, or the default logger when l is nil, so a probe without a
// configured logger still records its PDH notes.
func logger(l *slog.Logger) *slog.Logger {
	if l == nil {
		return slog.Default()
	}
	return l
}

// NvidiaSMI probes one GPU through nvidia-smi: the card at BusID when it
// is set, otherwise the card at Index.
type NvidiaSMI struct {
	Index int
	BusID string // pci.bus_id, as nvidia-smi prints it
	Bin   string // default "nvidia-smi"
	Log   *slog.Logger
}

func (n NvidiaSMI) bin() string {
	if n.Bin == "" {
		return "nvidia-smi"
	}
	return n.Bin
}

func (n NvidiaSMI) run(ctx context.Context, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, n.bin(), args...).Output()
	if err != nil {
		return "", fmt.Errorf("%s %s: %w", n.bin(), strings.Join(args, " "), err)
	}
	return string(out), nil
}

func (n NvidiaSMI) Read(ctx context.Context) (Snapshot, error) {
	mem, err := n.run(ctx, "--query-gpu="+Query, "--format=csv,noheader,nounits")
	if err != nil {
		return Snapshot{}, err
	}
	devs, err := ParseDevices(mem)
	if err != nil {
		return Snapshot{}, err
	}
	d, err := pick(devs, n.Index, n.BusID)
	if err != nil {
		return Snapshot{Devices: devs}, err
	}
	selector := n.BusID
	if selector == "" {
		selector = strconv.Itoa(d.Index)
	}
	apps, err := n.run(ctx, "-i", selector, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
	if err != nil {
		return Snapshot{}, err
	}
	s, err := Parse(d.Index, mem, apps, time.Now())
	if err != nil {
		return s, err
	}
	return withPDH(s, logger(n.Log)), nil
}

// ParseDevices reads "index, name[, bus_id], total, used, free[, telemetry...]"
// lines (memory in MiB), one per card, in the order of Query. A line can carry
// the full Query, Query without its telemetry tail, or just the memory fields.
func ParseDevices(mem string) ([]Device, error) {
	var out []Device
	for _, line := range strings.Split(mem, "\n") {
		if line = strings.TrimSpace(line); line == "" {
			continue
		}
		f := splitCSV(line)
		if len(f) != 5 && len(f) != 6 && len(f) != 15 {
			return nil, fmt.Errorf("gpu query: want 5, 6 or 15 fields, got %q", line)
		}
		idx, err := strconv.Atoi(f[0])
		if err != nil {
			return nil, fmt.Errorf("memory query index %q: %w", f[0], err)
		}
		d := Device{Index: idx, Name: f[1]}
		memField := 2 // first memory field: index, name[, bus_id], total, used, free
		if len(f) >= 6 {
			d.BusID = f[2]
			memField = 3
		}
		for i, dst := range []*uint64{&d.Total, &d.Used, &d.Free} {
			n, err := strconv.ParseUint(f[memField+i], 10, 64)
			if err != nil {
				return nil, fmt.Errorf("memory query %q field %d: %w", line, memField+i, err)
			}
			*dst = n * MiB
		}
		tel := []*float64{&d.Util, &d.Temp, &d.Power, &d.PowerLimit, &d.Fan, &d.ClockSM, &d.ClockMem, &d.PCIeGen, &d.PCIeWidth}
		for i, dst := range tel {
			*dst = -1
			if len(f) == 15 {
				if v, err := strconv.ParseFloat(f[6+i], 64); err == nil {
					*dst = v
				}
			}
		}
		out = append(out, d)
	}
	return out, nil
}

// Parse builds the snapshot for the card at index from nvidia-smi's
// csv,noheader,nounits output: mem lists every card, apps that card's processes.
func Parse(index int, mem, apps string, at time.Time) (Snapshot, error) {
	s := Snapshot{At: at}
	devs, err := ParseDevices(mem)
	if err != nil {
		return s, err
	}
	s.Devices = devs
	found := false
	for _, d := range devs {
		if d.Index == index {
			s.Index, s.Name, s.Total, s.Used, s.Free = d.Index, d.Name, d.Total, d.Used, d.Free
			found = true
		}
	}
	if !found {
		return s, fmt.Errorf("no GPU with index %d (found %d)", index, len(devs))
	}

	for _, line := range strings.Split(apps, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "No running") {
			continue
		}
		// The name may itself contain commas: the PID is before the first
		// one and the memory figure after the last.
		i, j := strings.IndexByte(line, ','), strings.LastIndexByte(line, ',')
		if i < 0 || i == j {
			return s, fmt.Errorf("compute-apps line %q", line)
		}
		pid, err := strconv.Atoi(strings.TrimSpace(line[:i]))
		if err != nil {
			return s, fmt.Errorf("compute-apps pid in %q: %w", line, err)
		}
		used, err := strconv.ParseUint(strings.TrimSpace(line[j+1:]), 10, 64)
		if err != nil {
			// "[N/A]" on some drivers and in some containers.
			used = 0
		}
		s.Procs = append(s.Procs, Proc{PID: pid, Name: strings.TrimSpace(line[i+1 : j]), Used: used * MiB})
	}
	return s, nil
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}
