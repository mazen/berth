// Copyright 2026 Trevin Tindall (T-Crypt)
// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fixture is a small llama-swap config with the shapes the walker must handle:
// a folded block, a literal block, a quoted name with a space, a proxied model
// (no cmd), a single-line cmd, comments everywhere, and a top-level key after
// models that must keep its distance.
const fixture = `# llama-swap config
macros:
  server: /usr/local/bin/llama-server
models:
  # a note that stays
  "Alpha-7B":
    aliases: ["alpha"]
    cmd: >
      ${server} -m /models/alpha.gguf
      --ctx-size 4096
    ttl: 300
  "Beta 9B":
    cmd: |
      ${server} -m /models/beta.gguf
  "Gamma":
    proxy: http://127.0.0.1:8000
  "Delta":
    cmd: ${server} -m /models/delta.gguf
after:
  key: 1
`

// fixtureWired is fixture after one wire with options() below.
const fixtureWired = `# llama-swap config
macros:
  server: /usr/local/bin/llama-server
models:
  # a note that stays
  "Alpha-7B":
    aliases: ["alpha"]
    cmd: >
      /usr/bin/berth run --as Alpha-7B --group llama-swap --priority foreground --drain-post http://127.0.0.1:9090/api/models/unload/Alpha-7B --
      ${server} -m /models/alpha.gguf
      --ctx-size 4096
    ttl: 300
  "Beta 9B":
    cmd: |
      /usr/bin/berth run --as 'Beta 9B' --group llama-swap --priority foreground --drain-post http://127.0.0.1:9090/api/models/unload/Beta%209B --
      ${server} -m /models/beta.gguf
  "Gamma":
    proxy: http://127.0.0.1:8000
  "Delta":
    cmd: ${server} -m /models/delta.gguf
after:
  key: 1
`

func options() Options {
	return Options{
		Bin:      "/usr/bin/berth",
		Group:    "llama-swap",
		Priority: "foreground",
		APIBase:  "http://127.0.0.1:9090",
	}
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLlamaSwapWiresTheWholeRoster(t *testing.T) {
	r, err := LlamaSwap(writeTemp(t, fixture), options())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"Alpha-7B": "add",
		"Beta 9B":  "add",
		"Gamma":    "skip",
		"Delta":    "skip",
	}
	for _, m := range r.Models {
		if want[m.Name] != m.Action {
			t.Errorf("%s: action = %q, want %q (reason %q)", m.Name, m.Action, want[m.Name], m.Reason)
		}
	}
	if !r.Changed {
		t.Error("Changed = false, want true")
	}
	if got := r.NewContent(); got != fixtureWired {
		t.Errorf("content mismatch:\n--- got ---\n%s\n--- want ---\n%s", got, fixtureWired)
	}
	if r.Count("add") != 2 || r.Count("skip") != 2 {
		t.Errorf("counts = add %d, skip %d; want 2, 2", r.Count("add"), r.Count("skip"))
	}
}

func TestLlamaSwapIsIdempotent(t *testing.T) {
	r, err := LlamaSwap(writeTemp(t, fixtureWired), options())
	if err != nil {
		t.Fatal(err)
	}
	if r.Changed {
		t.Errorf("Changed = true on already wired input (models: %+v)", r.Models)
	}
	if r.NewContent() != fixtureWired {
		t.Errorf("content changed on second pass:\n%s", r.NewContent())
	}
	for _, m := range r.Models {
		if m.Action != "wired" && m.Action != "skip" {
			t.Errorf("%s: action = %q, want wired or skip", m.Name, m.Action)
		}
	}
}

func TestLlamaSwapUpdateRealignsFlags(t *testing.T) {
	// A hand-wrapped line with different flags: without Update it counts as
	// wired and stays; with Update it is rewritten to the requested flags.
	in := strings.Replace(fixtureWired,
		"/usr/bin/berth run --as Alpha-7B --group llama-swap --priority foreground --drain-post http://127.0.0.1:9090/api/models/unload/Alpha-7B --",
		"/opt/old/berth run --as Alpha-7B --priority interactive --drain-post http://127.0.0.1:9090/api/models/unload/Alpha-7B --", 1)
	o := options()

	r, err := LlamaSwap(writeTemp(t, in), o)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range r.Models {
		if m.Name == "Alpha-7B" && m.Action != "wired" {
			t.Errorf("without Update: Alpha-7B action = %q, want wired", m.Action)
		}
	}

	o.Update = true
	r, err = LlamaSwap(writeTemp(t, in), o)
	if err != nil {
		t.Fatal(err)
	}
	var updated *Model
	for i, m := range r.Models {
		if m.Name == "Alpha-7B" {
			updated = &r.Models[i]
		}
	}
	if updated == nil || updated.Action != "update" {
		t.Fatalf("with Update: Alpha-7B = %+v, want update", updated)
	}
	out := r.NewContent()
	if strings.Contains(out, "/opt/old/berth") {
		t.Error("old wrapper line survived --update")
	}
	if !strings.Contains(out, "/usr/bin/berth run --as Alpha-7B --group llama-swap --priority foreground") {
		t.Errorf("updated line missing:\n%s", out)
	}
	// And the update is stable.
	r2, err := LlamaSwap(writeTemp(t, out), o)
	if err != nil {
		t.Fatal(err)
	}
	if r2.Changed {
		t.Error("--update is not idempotent")
	}
}

func TestLlamaSwapExclude(t *testing.T) {
	o := options()
	o.Exclude = []string{"Beta *", "Delta"}
	r, err := LlamaSwap(writeTemp(t, fixture), o)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range r.Models {
		switch m.Name {
		case "Beta 9B":
			if m.Action != "skip" || !strings.Contains(m.Reason, "excluded") {
				t.Errorf("Beta 9B = %+v, want excluded skip", m)
			}
		case "Delta":
			if m.Action != "skip" || !strings.Contains(m.Reason, "excluded") {
				t.Errorf("Delta = %+v, want exclusion to win over the cmd shape", m)
			}
		}
	}
	if strings.Contains(r.NewContent(), "run --as 'Beta 9B'") {
		t.Error("excluded model was wired")
	}
}

func TestLlamaSwapKeepsCRLF(t *testing.T) {
	in := strings.ReplaceAll(fixture, "\n", "\r\n")
	r, err := LlamaSwap(writeTemp(t, in), options())
	if err != nil {
		t.Fatal(err)
	}
	out := r.NewContent()
	if strings.Contains(out, "\n\n") || strings.Contains(strings.ReplaceAll(out, "\r\n", ""), "\r") {
		t.Errorf("mixed line endings after edit:\n%q", out)
	}
	if !strings.Contains(out, "--drain-post http://127.0.0.1:9090/api/models/unload/Alpha-7B --\r\n") {
		t.Error("inserted line does not carry the file's CRLF ending")
	}
	r2, err := LlamaSwap(writeTemp(t, out), options())
	if err != nil || r2.Changed {
		t.Errorf("CRLF pass is not idempotent: changed=%v err=%v", r2 != nil && r2.Changed, err)
	}
}

func TestLlamaSwapErrors(t *testing.T) {
	if _, err := LlamaSwap(filepath.Join(t.TempDir(), "missing.yaml"), options()); err == nil {
		t.Error("missing file: want error")
	}
	if _, err := LlamaSwap(writeTemp(t, "a: [}\n"), options()); err == nil {
		t.Error("bad YAML: want error")
	}
	if _, err := LlamaSwap(writeTemp(t, "macros:\n  server: x\n"), options()); err == nil {
		t.Error("no models key: want error")
	}
	if _, err := LlamaSwap(writeTemp(t, "models: {}\n"), options()); err == nil {
		t.Error("empty models: want error")
	}
}

func TestOptionsValidate(t *testing.T) {
	bad := []Options{
		{Bin: ""},
		{Bin: "x", APIBase: "http://127.0.0.1:9090", Priority: "urgent"},
		{Bin: "x", APIBase: "not a url", Priority: "foreground"},
		{Bin: "x", APIBase: "http://127.0.0.1:9090", Priority: "", Exclude: []string{"[bad"}},
	}
	for i, o := range bad {
		if err := o.Validate(); err == nil {
			t.Errorf("case %d: want error", i)
		}
	}
	ok := Options{Bin: "x", APIBase: "http://127.0.0.1:9090", Priority: "foreground", Group: "g", Exclude: []string{"a*"}}
	if err := ok.Validate(); err != nil {
		t.Errorf("valid options rejected: %v", err)
	}
}

func TestWriteBacksUpAndReplaces(t *testing.T) {
	p := writeTemp(t, fixture)
	if err := os.Chmod(p, 0o600); err != nil {
		t.Fatal(err)
	}
	r, err := LlamaSwap(p, options())
	if err != nil {
		t.Fatal(err)
	}
	backup, err := Write(r)
	if err != nil {
		t.Fatal(err)
	}
	if backup == "" || !strings.Contains(backup, ".bak-wire-") {
		t.Fatalf("backup = %q, want a .bak-wire path", backup)
	}
	old, err := os.ReadFile(backup)
	if err != nil || string(old) != fixture {
		t.Errorf("backup content: err=%v, matches=%v", err, string(old) == fixture)
	}
	got, err := os.ReadFile(p)
	if err != nil || string(got) != fixtureWired {
		t.Errorf("written content: err=%v, matches=%v", err, string(got) == fixtureWired)
	}
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && fi.Mode().Perm() != 0o600 {
		// Windows carries no Unix mode bits; the file's existence and
		// content are the check there.
		t.Errorf("mode = %v, want 0600 preserved", fi.Mode().Perm())
	}

	// A result with nothing to change writes nothing.
	clean, err := LlamaSwap(p, options())
	if err != nil {
		t.Fatal(err)
	}
	if b, err := Write(clean); err != nil || b != "" {
		t.Errorf("Write on a clean result = (%q, %v), want (\"\", nil)", b, err)
	}
}

// TestMain pins the POSIX rules the fixtures are written in, on every OS;
// a test that wants the Windows rules opts in with withWindowsArgs.
func TestMain(m *testing.M) {
	windowsArgs = false
	os.Exit(m.Run())
}

// withWindowsArgs runs a test under the Windows command-line rules on any
// OS, the way llama-swap splits a cmd on a Windows host.
func withWindowsArgs(t *testing.T) {
	t.Helper()
	was := windowsArgs
	windowsArgs = true
	t.Cleanup(func() { windowsArgs = was })
}

// On Windows the wrapper names berth by a backslash path, often with a
// space in it. The line must read back as the same words under the rules
// llama-swap splits with there, and a second pass must find it wired.
func TestLlamaSwapWindowsPaths(t *testing.T) {
	withWindowsArgs(t)
	for _, bin := range []string{`D:\AI\berth\bin\berth.exe`, `D:\AI Stuff\berth\bin\berth.exe`, `C:\Tools\BERTH.EXE`} {
		o := options()
		o.Bin = bin
		r, err := LlamaSwap(writeTemp(t, fixture), o)
		if err != nil {
			t.Fatalf("%s: %v", bin, err)
		}
		if r.Count("add") != 2 {
			t.Fatalf("%s: add = %d, want 2", bin, r.Count("add"))
		}
		line := wrapperLine("Beta 9B", o, 0, "")
		if strings.Contains(line, "'") {
			t.Errorf("%s: wrapper line uses POSIX quotes: %s", bin, line)
		}
		if f := shellFields(line); len(f) < 4 || f[0] != bin || f[3] != "Beta 9B" {
			t.Errorf("%s: wrapper line reads back as %q", bin, f)
		}
		again, err := LlamaSwap(writeTemp(t, r.NewContent()), o)
		if err != nil {
			t.Fatalf("%s: second pass: %v", bin, err)
		}
		if again.Changed {
			t.Errorf("%s: second pass wants changes: %+v", bin, again.Models)
		}
	}
}

func TestWindowsFields(t *testing.T) {
	cases := map[string][]string{
		`D:\AI\berth.exe run --as x`:       {`D:\AI\berth.exe`, "run", "--as", "x"},
		`"D:\AI Stuff\berth.exe" run`:      {`D:\AI Stuff\berth.exe`, "run"},
		`"C:\dir\\" next`:                  {`C:\dir\`, "next"},
		`a\"b 'c d'`:                       {`a"b`, "'c", "d'"},
		`"" x`:                             {"", "x"},
		`"say \"hi\"" \server\share\a.exe`: {`say "hi"`, `\server\share\a.exe`},
	}
	for in, want := range cases {
		got := windowsFields(in)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("windowsFields(%q) = %q, want %q", in, got, want)
		}
	}
	for _, w := range []string{`D:\AI\berth.exe`, `D:\AI Stuff\x`, `C:\dir with space\`, `a"b`, ""} {
		if got := windowsFields(windowsWord(w) + " tail"); len(got) != 2 || got[0] != w {
			t.Errorf("windowsWord(%q) = %s reads back as %q", w, windowsWord(w), got)
		}
	}
}
