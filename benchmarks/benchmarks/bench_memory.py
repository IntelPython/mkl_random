# Copyright (c) 2026, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Peak-memory benchmarks.

Peak RSS includes setup, so setup allocates nothing large.
"""

from ._utils import _make_state

_N = 10_000_000
_REPEATS = 10_000


class PeakMemFill:
    """Peak RSS of large fills, to catch new temporary buffers."""

    def setup(self):
        self.rs = _make_state()

    def peakmem_standard_normal(self):
        self.rs.standard_normal(_N)

    def peakmem_randint_int64_small(self):
        self.rs.randint(0, 100, _N, dtype="int64")

    def peakmem_randint_uint64_wide(self):
        self.rs.randint(0, 3 * 2**32, _N, dtype="uint64")

    def peakmem_randint_uint8(self):
        self.rs.randint(0, 2**8, _N, dtype="uint8")

    def peakmem_noncentral_chisquare(self):
        self.rs.noncentral_chisquare(3.0, 2.0, _N)

    def peakmem_zipf(self):
        self.rs.zipf(2.0, _N)


class PeakMemRepeat:
    """Peak RSS of repeated small calls, to catch per-call leaks."""

    def setup(self):
        self.rs = _make_state()
        self.state = self.rs.get_state()

    def peakmem_set_state(self):
        for _ in range(_REPEATS):
            self.rs.set_state(self.state)

    def peakmem_logseries(self):
        for _ in range(_REPEATS):
            self.rs.logseries(0.9, 1_000)
