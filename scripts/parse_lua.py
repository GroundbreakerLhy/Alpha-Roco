#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse the Lua-subset data tables used by biligame rocom wiki modules."""
import re


class LuaParser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def skip_ws(self):
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c.isspace():
                self.i += 1
            elif c == "-" and s[self.i:self.i + 2] == "--":
                j = s.find("\n", self.i)
                self.i = len(s) if j < 0 else j + 1
            else:
                break

    def parse(self):
        self.skip_ws()
        if self.s.startswith("return", self.i):
            self.i += len("return")
        return self.parse_value()

    def parse_string(self):
        s = self.s
        if s[self.i] == '"':
            self.i += 1
            out = []
            while self.i < len(s):
                c = s[self.i]
                if c == "\\":
                    n = s[self.i + 1:self.i + 2]
                    out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"',
                                "\\": "\\", "'": "'", "0": "\0"}.get(n, n))
                    self.i += 2
                elif c == '"':
                    self.i += 1
                    return "".join(out)
                else:
                    out.append(c)
                    self.i += 1
            raise ValueError("unterminated string")
        if s.startswith("[[", self.i):
            end = s.find("]]", self.i + 2)
            if end < 0:
                raise ValueError("unterminated long string")
            out = s[self.i + 2:end]
            self.i = end + 2
            return out
        raise ValueError("not a string at %d" % self.i)

    def parse_number(self):
        m = re.match(r"-?\d+(?:\.\d+)?", self.s[self.i:])
        if not m:
            raise ValueError("not a number at %d" % self.i)
        tok = m.group(0)
        self.i += len(tok)
        return float(tok) if "." in tok else int(tok)

    def parse_table(self):
        assert self.s[self.i] == "{"
        self.i += 1
        tbl = {}
        n = 0
        while True:
            self.skip_ws()
            if self.i >= len(self.s):
                raise ValueError("unterminated table")
            if self.s[self.i] == "}":
                self.i += 1
                break
            key = None
            self.skip_ws()
            if self.s[self.i] == "[":
                self.i += 1
                self.skip_ws()
                if self.s[self.i] == '"':
                    key = self.parse_string()
                else:
                    key = self.parse_number()
                self.skip_ws()
                assert self.s[self.i] == "]"
                self.i += 1
                self.skip_ws()
                assert self.s[self.i] == "="
                self.i += 1
            else:
                m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", self.s[self.i:])
                if m:
                    save = self.i
                    name = m.group(0)
                    j = self.i + len(name)
                    while j < len(self.s) and self.s[j].isspace():
                        j += 1
                    if j < len(self.s) and self.s[j] == "=":
                        key = name
                        self.i = j + 1
                    else:
                        self.i = save
            self.skip_ws()
            val = self.parse_value()
            if key is None:
                n += 1
                tbl[n] = val
            else:
                tbl[key] = val
            self.skip_ws()
            if self.i < len(self.s) and self.s[self.i] in ",;":
                self.i += 1
        if n and all(isinstance(k, int) and 1 <= k <= n for k in tbl):
            return [tbl[k] for k in sorted(tbl)]
        return tbl

    def parse_value(self):
        self.skip_ws()
        c = self.s[self.i]
        if c == "{":
            return self.parse_table()
        if c == '"' or self.s.startswith("[[", self.i):
            return self.parse_string()
        if c == "-" or c.isdigit():
            return self.parse_number()
        for word, val in (("true", True), ("false", False), ("nil", None)):
            if self.s.startswith(word, self.i):
                self.i += len(word)
                return val
        raise ValueError("unexpected char %r at %d" % (c, self.i))


def load_lua(path):
    with open(path, encoding="utf-8") as f:
        return LuaParser(f.read()).parse()


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        d = load_lua(p)
        print(p, "->", type(d).__name__, "len", len(d) if hasattr(d, "__len__") else "-")
